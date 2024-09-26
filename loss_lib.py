# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""A library for losses."""

import gin.tf
import tensorflow as tf


@gin.configurable
class NumetricLoss:
    """Losses for comparing between two tensors of the same shape."""

    def __init__(
        self,
        weight_l1_loss: float = 0.0,
        weight_l2_loss: float = 1.0,
        weight_s3im_loss: float = 0.0,
        weight_lpips_loss: float = 0.0,
        weight_temporal_reg_loss: float = 0.0,
    ):
        """Initializes the loss class.

        Args:
          weight_l1_loss: A float number indicating the weight of the l1 loss.
          weight_l2_loss: A float number indicating the weight of the l2 loss.
          weight_s3im_loss: A float number indicating the weight of the S3IM loss
            [ICCV'23], https://arxiv.org/abs/2308.07032.
          weight_lpips_loss: A float number indicating the weight of the lpips loss.
          weight_temporal_reg_loss: A float number indicating the weight of the
            temporal regularization loss.
        """
        self._weight_l1_loss = weight_l1_loss
        self._weight_l2_loss = weight_l2_loss
        self._weight_s3im_loss = weight_s3im_loss
        self._weight_lpips_loss = weight_lpips_loss
        self._weight_temporal_reg_loss = weight_temporal_reg_loss

    def __call__(
        self,
        model_targets: dict[str, tf.Tensor],
        model_outputs: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Computes a loss dictionary for the given outputs and targets.

        The loss is calculate by comparing the output 'results' element to the
        target 'expected' element. An element 'total_loss' is required to present.

        Args:
          model_targets: A dictionary of model targets.
          model_outputs: A dictionary of model outputs.

        Returns:
          A dictionary with string keys and tensor values.
        """
        losses = {"total_loss": tf.constant(0.0, dtype=tf.float32)}

        if self._weight_l1_loss > 0.0:
            target_tensor = model_targets["target"]
            output_tensor = model_outputs["output"]
            target_tensor = tf.reshape(target_tensor, (target_tensor.shape[0], -1))
            output_tensor = tf.reshape(output_tensor, (output_tensor.shape[0], -1))
            l1_distance = tf.reduce_sum(
                tf.reduce_mean(tf.math.abs(output_tensor - target_tensor), axis=1),
                axis=0,
            )  # l1 distance. Note the loss is summarized (not averaged) along the
            # batch_size axis, because the averaging will be done later.
            losses["l1_loss"] = l1_distance
            losses["total_loss"] += l1_distance * self._weight_l1_loss

        if self._weight_l2_loss > 0.0:
            target_tensor = model_targets["target"]
            output_tensor = model_outputs["output"]
            target_tensor = tf.reshape(target_tensor, (target_tensor.shape[0], -1))
            output_tensor = tf.reshape(output_tensor, (output_tensor.shape[0], -1))
            l2_distance = tf.reduce_sum(
                tf.reduce_mean(tf.math.square(output_tensor - target_tensor), axis=1),
                axis=0,
            )  # l2 distance. Note the loss is summarized (not averaged) along the
            # batch_size axis, because the averaging will be done later.
            losses["l2_loss"] = l2_distance
            losses["total_loss"] += l2_distance * self._weight_l2_loss

        if self._weight_lpips_loss > 0.0:
            h, w = model_outputs["height"], model_outputs["width"]
            lpips_model = model_outputs["lpips_model"]
            target_tensor = model_targets["target"]
            output_tensor = model_outputs["output"]
            num_images = target_tensor.shape[0]
            target_tensor = tf.reshape(target_tensor, (num_images, h, w, 3))
            output_tensor = tf.reshape(output_tensor, (num_images, h, w, 3))
            lpips_loss = tf.reduce_sum(
                lpips_model(output_tensor, target_tensor), axis=0
            )
            losses["lpips_loss"] = lpips_loss
            losses["total_loss"] += lpips_loss * self._weight_lpips_loss

        # Refer to https://github.com/Madaoer/S3IM-Neural-Fields/blob/main/
        # model_components/s3im.py
        # The original S3IM loss is applied to *stochastic* patches (i.e., random
        # pixels). Here we apply it to the real patch (a whole image). Also, the
        # original S3IM loss uses multiple patches (M=10 in their paper), while per
        # their paper, M=1 works closely to M=10. For simplicity, here we use M=1.
        if self._weight_s3im_loss > 0.0:
            h, w = model_outputs["height"], model_outputs["width"]
            target_tensor = model_targets["target"]
            output_tensor = model_outputs["output"]
            batch_size = target_tensor.shape[0]
            target_tensor = tf.reshape(target_tensor, (batch_size, h, w, 3))
            output_tensor = tf.reshape(output_tensor, (batch_size, h, w, 3))
            s3im_loss = tf.reduce_sum(
                1
                - tf.image.ssim(
                    img1=target_tensor, img2=output_tensor, max_val=1.0, filter_size=4
                ),  # filter_size=4 per S3IM paper.
                axis=0,
            )
            losses["s3im_loss"] = s3im_loss
            losses["total_loss"] += s3im_loss * self._weight_s3im_loss

        if self._weight_temporal_reg_loss > 0.0 and "input" in model_outputs:
            input_tensor = model_outputs["input"]
            output_tensor = model_outputs["output"]
            num_images_per_example = model_outputs["num_images_per_example"]
            num_images = output_tensor.shape[0]
            batch_size = num_images // num_images_per_example
            input_tensor = tf.reshape(
                input_tensor, (batch_size, num_images_per_example, -1)
            )
            output_tensor = tf.reshape(
                output_tensor, (batch_size, num_images_per_example, -1)
            )
            in1, in2 = (
                input_tensor[:, 0, ...],
                input_tensor[:, num_images_per_example - 1, ...],
            )
            out1, out2 = (
                output_tensor[:, 0, ...],
                output_tensor[:, num_images_per_example - 1, ...],
            )
            temporal_reg_loss = tf.norm(out2 - out1, ord=1) / tf.norm(in2 - in1, ord=1)
            losses["temporal_reg_loss"] = temporal_reg_loss
            losses["total_loss"] += temporal_reg_loss * self._weight_temporal_reg_loss

        return losses
