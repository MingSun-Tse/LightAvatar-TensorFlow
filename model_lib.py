"""Model library for efficient realvatar."""

import math
import re
from typing import Any, List, Optional, Tuple

from absl import logging
import gin
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion as q
import tensorflow_hub as hub

from google3.pyglib import gfile

RAY_ORIGIN_KEY = "rays_o"
RAY_DIRECTION_KEY = "rays_d"
RAY_COLOR_KEY = "rays_c"
RAY_EXPRESSION_KEY = "rays_e"
RAY_ROTATION_KEY = "rays_r"
RAY_KID_KEY = "rays_k"
EXP_DIM = 157
ROT_DIM = 12
C2W_KEY = "c2w"
C2W_WRT_SHOULDER_KEY = "c2w_wrt_shoulder"
RGB_KEY = "rgb"
EXP_KEY = "exp"
ROT_KEY = "rot"
KID_KEY = "kid"
FID_KEY = "fid"  # Frame ID
RAY_DATA_KEY = "ray_data"  # Ray format, each ray is a tf example.
FLAME_DATASET_INFO = {  # Summary of the dataset of different subjects.
    "subject0": {
        "near": 3.3,
        "far": 5.2,
        "height": 512,
        "width": 364,
        "n_real_train_frames": 1560,
    },
    "subject1": {
        "near": 3.3,
        "far": 5.3,
        "height": 512,
        "width": 472,
        "n_real_train_frames": 1012,
    },
    "subject2": {
        "near": 2.7,
        "far": 4.3,
        "height": 512,
        "width": 390,
        "n_real_train_frames": 1480,
    },
    "subject3": {
        "near": 3.5,
        "far": 5.2,
        "height": 500,
        "width": 512,
        "n_real_train_frames": 5310,
    },
    "subject4": {
        "near": 3.2,
        "far": 5.5,
        "height": 512,
        "width": 374,
        "n_real_train_frames": 1440,
    },
    "subject7": {
        "near": 1.9,
        "far": 3.9,
        "height": 512,
        "width": 348,
        "n_real_train_frames": 1420,
    },
    "subject8": {
        "near": 5.9,
        "far": 8.3,
        "height": 512,
        "width": 370,
        "n_real_train_frames": 1360,
    },
    "subject11": {
        "near": 3.25,
        "far": 5.1,
        "height": 512,
        "width": 406,
        "n_real_train_frames": 1450,
    },
    "subject14": {
        "near": 6.7,
        "far": 10.2,
        "height": 512,
        "width": 372,
        "n_real_train_frames": 2655,
    },
    "subject15": {
        "near": 6.7,
        "far": 10.2,
        "height": 512,
        "width": 452,
        "n_real_train_frames": 1818,
    },
    "subject17": {
        "near": 7.2,
        "far": 9.3,
        "height": 512,
        "width": 512,
        "n_real_train_frames": 3912,
    },
    "subject18": {
        "near": 6.0,
        "far": 8.3,
        "height": 512,
        "width": 344,
        "n_real_train_frames": 2656,
    },
    "subject19": {
        "near": 9.5,
        "far": 11.3,
        "height": 512,
        "width": 512,
        "n_real_train_frames": 2049,
    },
}


@gin.configurable()
class PointSampler(tf.keras.layers.Layer):
    """Point sampling using NeRF default sampling scheme.

    Refers to R2L [ECCV, 2022]: https://arxiv.org/abs/2203.17261.
    """

    def __init__(
        self,
        n_sampled_points: int = 16,
        near: float = 2.05,
        far: float = 4.05,
        perturb: bool = True,
    ):
        """Point Sampling Constructor.

        Args:
          n_sampled_points: Num of sampled points on a ray.
          near: Distance of the near plane from world origin.
          far: Distance of the far plane from world origin.
          perturb: Add perturbation to sampled point location.
        """
        super().__init__()
        self.n_sampled_points = n_sampled_points
        self.near = near
        self.far = far
        self.perturb = perturb

    def call(
        self, rays_o: tf.Tensor, rays_d: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        # Decide where to sample along each ray. Under the logic, all rays will be
        # sampled at the same times.
        t_vals = tf.linspace(0.0, 1.0, self.n_sampled_points)  # [n_sampled_points]

        # Linearly sample points between 'near' and 'far'.
        # Same linear sampling coefficients (t_vals) will be used for all rays.
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals

        # Perturb sampling points along each ray during training
        if self.perturb and training:
            # Get intervals between samples
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = tf.concat([mids, z_vals[None, -1]], -1)
            lower = tf.concat([z_vals[None, 0], mids], -1)  # [n_sampled_points]
            # Stratified samples in those intervals
            t_rand = tf.random.uniform(tf.shape(z_vals))
            z_vals = lower + (upper - lower) * t_rand  # [n_sampled_points]

        # Get points
        z_vals = z_vals[None]
        # [num_rays, 1, 3] + [num_rays, 1, 3] * [1, n_sampled_points, 1]
        # -> [num_rays, n_sampled_points, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = tf.reshape(pts, [pts.shape[0], -1])  # Concat all points along a ray
        return pts


@gin.configurable
class PositionalEncoder(tf.keras.layers.Layer):
    """Sinusoidal Positional Encodings."""

    def __init__(
        self,
        num_freqs: int = 10,
        max_freq_log2: Optional[float] = None,
        scale: Optional[float] = 1.0,
    ):
        """Sinusoidal Positional Encodings Constructor.

        Args:
          num_freqs: Number of frequency bands to use.
          max_freq_log2: Maximum log2 frequency band.
          scale: The scale applied to input features.
        """
        super().__init__()

        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1.0

        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** tf.linspace(0.0, max_freq_log2, num_freqs)
        self.scale = scale

    @property
    def dim_out(self) -> int:
        # +1 for identity.
        return 2 * self.num_freqs + 1

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """A vectorized sinusoidal encoding.

        Args:
          x: the input features to encode with shape [..., C].

        Returns:
          A tensor containing the encoded features.
        """
        freqs = self.freq_bands
        batch_shape = x.shape[:-1]
        batch_ones = [1] * len(batch_shape)

        freqs = tf.reshape(freqs, (*batch_ones, self.num_freqs, 1))  # (*, F, 1).
        x_expanded = tf.expand_dims(x, axis=-2)  # (*, 1, C).
        # Will be broadcasted to shape (*B, F, C).
        angles = self.scale * x_expanded * freqs

        # The shape of the features is (*B, F, 2, C) so that when we reshape it
        # it matches the ordering of the original NeRF code.
        features = tf.stack((tf.sin(angles), tf.cos(angles)), axis=-2)

        batch_shape_tf = tf.shape(x)[:-1]
        feats_shape = tf.concat(
            [batch_shape_tf, [2 * self.num_freqs * x.shape[-1]]], axis=0
        )
        features = tf.reshape(features, feats_shape)

        # Prepend the original signal for the identity.
        features = tf.concat([x, features], axis=-1)
        return features


def fc(width: int) -> tf.keras.layers.Layer:
    """Feedforward layer."""
    return tf.keras.layers.Dense(units=width)


def conv(width: int) -> tf.keras.layers.Layer:
    """Standard 3x3 conv layer with SAME padding."""
    return tf.keras.layers.Conv2D(filters=width, kernel_size=3, padding="same")


def leaky_relu(alpha: float) -> tf.keras.layers.Layer:
    """Leaky ReLU layer."""
    return tf.keras.layers.LeakyReLU(alpha=alpha)


def relu() -> tf.keras.layers.Layer:
    """ReLU layer."""
    return tf.keras.layers.ReLU()


def bn(synchronized: bool) -> tf.keras.layers.Layer:
    """Batch Normalization layer."""
    return tf.keras.layers.BatchNormalization(synchronized=synchronized)


class ResMLP(tf.keras.Model):
    """Residual MLP block.

    Refers to R2L [ECCV, 2022]: https://arxiv.org/abs/2203.17261.
    """

    def __init__(
        self,
        width: int,
        depth: int = 2,
        activation: str = "LeakyReLU",
        residual: bool = True,
        **kwargs,
    ):
        """Builds the ResMLP instance.

        Args:
          width: Num of neurons in each layer.
          depth: Num of layers.
          activation: Activation function type.
          residual: Whether to use residual. Default: True.
          **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        activation = activation.lower()
        assert activation in ["leakyrelu", "relu"]
        modules = [fc(width)]
        for _ in range(depth - 1):
            if activation == "leakyrelu":
                modules.append(leaky_relu(alpha=0.2))
            elif activation == "relu":
                modules.append(relu())
            modules.append(fc(width))
        self._body = tf.keras.Sequential(modules)
        self._residual = residual

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self._residual:
            return self._body(x) + x
        else:
            return self._body(x)


class ResBlock(tf.keras.Model):
    """Residual block. Refers to EDSR [2017, CVPRW].

    https://github.com/sanghyun-son/EDSR-PyTorch/blob/8dba5581a7502b92de9641eb431130d6c8ca5d7f/src/model/common.py#L37
    """

    def __init__(
        self,
        width: int,
        depth: int = 2,
        norm: Optional[str] = "bn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        modules = [conv(width)]
        if norm == "bn":
            modules.append(bn(synchronized=True))
        for _ in range(depth - 1):
            modules += [leaky_relu(0.2), conv(width)]
            if norm == "bn":
                modules.append(bn(synchronized=True))
        self._body = tf.keras.Sequential(modules)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._body(x) + x


class PixelShuffle(tf.keras.layers.Layer):
    """Pixel shuffle. A typical module for upsampling in SR. [CVPR'16].

    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.depth_to_space(x, block_size=self.scale)


def transpose_conv(width: int) -> tf.keras.layers.Layer:
    """Transpose 3x3 conv layer with SAME padding. Used for SR x2 upsampling."""
    return tf.keras.layers.Conv2DTranspose(
        filters=width, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )  # x2 upsampling, so strides set to 2.


@gin.configurable
class Student(tf.keras.Model):
    """NeLF student model for distilling NeRF-based avatar model.

    Refers to R2L [ECCV, 2022]: https://arxiv.org/abs/2203.17261.
    """

    def __init__(
        self,
        dim_out: int = 3,
        n_resblocks: int = 40,
        width: int = 128,
        n_rays_train: int = 4096,
        use_ray_position: bool = True,
        use_expressions: bool = False,
        use_jaw_eyes_pose: bool = False,
        apply_rotation_limits: bool = False,
        expression_scheme: str = "brute_force",
        n_local_feats: int = 1024,
        local_feat_dim: int = 32,
        local_feat_width: int = 128,
        local_feat_n_resblocks: int = 2,
        spatial_attention_width: int = 128,
        spatial_attention_n_resblocks: int = 2,
        spatial_attention_use_expressions: bool = True,
        image_size: str = "512x557",
        activation: str = "LeakyReLU",
        activation_out: str = "LeakyReLU",
        n_resblocks_warp_network: int = -1,
        freeze_warp_network: bool = False,
        freeze_student: bool = False,
        use_rotations: bool = False,
        n_real_train_frames: int = 3470,
        warp_condition_latent_dim: int = 64,
        warp_ablation_study: bool = False,
        use_warp: str = "",
        use_pseudo_frameid: bool = False,
        no_warp_for_pseudo: bool = False,
        set_up_lpips_model: bool = False,
        data: str = "GNOME",
        eval_chunk_size: int = 4096,
        norm: Optional[str] = None,
        input_noise_std: Optional[float] = None,
        input_noise_position: str = "nelf_input",
        input_noise_target: Tuple[str, ...] = ("points", "expression"),
        lpips_model: str = "vgg",
        return_internal_results: bool = False,
        n_resblocks_expression_adapter: int = -1,
        base_rays_d_path: Optional[str] = None,
        num_freqs_point: int = 10,
        num_freqs_expression: int = 10,
        n_resblocks_sr: int = -1,
        width_sr: int = 128,
        sr_scale: int = 4,
        sr_upsample: str = "PixelShuffle",
        sr_norm: Optional[str] = None,
        temporal_reg: bool = False,
        cam_params: Optional[List[Any]] = None,
        **kwargs,
    ):
        """Builds the Student instance.

        Args:
          dim_out: Output dimension, 3 in default because output is typically RGB.
          n_resblocks: Num of residual blocks in the NeLF model.
          width: Num of neurons in each layer.
          n_rays_train: Num of total rays per iteration during training.
          use_ray_position: Use ray position as NeLF input.
          use_expressions: Use expressions as a part of student input.
          use_jaw_eyes_pose: Use jaw eyes pose as a part of student NeLF input.
          apply_rotation_limits: Apply rotation limits when using jaw and eyes
            rotations.
          expression_scheme: Different schemes to use expression in the model.
          n_local_feats: Num of local features when expression scheme is
            local_feat_v1 or local_feat_v2.
          local_feat_dim: Dimension of local feature.
          local_feat_width: Width of local feature network.
          local_feat_n_resblocks: Num of residual blocks of local feature network.
          spatial_attention_width: Width of spatial attention network.
          spatial_attention_n_resblocks: Num of residual blocks of spatial attention
            network.
          spatial_attention_use_expressions: Use expression as input for spatial
            attention network.
          image_size: Image size of the training & testing data.
          activation: Activation function type of the backbone.
          activation_out: Activation function type of the output layer.
          n_resblocks_warp_network: Num of residual blocks in the warp network. -1
            means there is no warp network.
          freeze_warp_network: Not optimize the warp network.
          freeze_student: Not optimize the student network.
          use_rotations: Use GNOME rotations as the input of student.
          n_real_train_frames: Num of real training frames.
          warp_condition_latent_dim: condition latent dim for the warping network.
          warp_ablation_study: Conduct ablation study to warp network: remove the
            warp network but only concat the warp condition latent to the NeLF
            input.
          use_warp: A string to indicate using warp network during train or test.
          use_pseudo_frameid: Bool, all frame ids use the same pseudo id.
          no_warp_for_pseudo: Bool, do not use warp for pseudo rays.
          set_up_lpips_model: Bool, set up lpips model during init.
          data: String, data set name. Choices = ['GNOME', 'FLAME'].
          eval_chunk_size: Num of rays in a batch during eval to avoid OOM.
          norm: Type of normalization layers, default None.
          input_noise_std: Add noise to input to avoid overfitting, default None.
          input_noise_position: String. Add noise to the input at which position.
          input_noise_target: Add noise to which input, choices: Any combination of
            ["points", "expression"].
          lpips_model: Model name to calculate LPIPS, default: "vgg", choices:
            ["vgg", "alexnet"].
          return_internal_results: Return features of intermediate layers for
            analyses, default: False.
          n_resblocks_expression_adapter: Num of residual blocks of expression
            adapter network to adjust the raw input expression, in the hopes of
            mitigating the flickering problem.
          base_rays_d_path: Path to base_rays_d, used for FLAME data.
          num_freqs_point: Num of frequencies in positional encoding for sampled
            points.
          num_freqs_expression: Num of frequencies in positional encoding for
            sampled expressions.
          n_resblocks_sr: Num of resblocks for SR module.
          width_sr: Num of neurons in each layer of the SR network.
          sr_scale: SR upsampling scale.
          sr_upsample: SR upsampling scheme. PixelShuffle or TransposeConv.
          sr_norm: Norm layer type for SR network.
          temporal_reg: Use temporal regularization or not.
          cam_params: Camera params.
          **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.expression_scheme = expression_scheme
        self.n_verts_per_ray = 64  # 64 kNN points used.
        self.n_local_feats = n_local_feats
        self.local_feat_dim = local_feat_dim
        self.n_rays_train = n_rays_train
        self.use_ray_position = use_ray_position
        self.use_expressions = use_expressions
        self.use_rotations = use_rotations
        self.use_jaw_eyes_pose = use_jaw_eyes_pose
        self.apply_rotation_limits = apply_rotation_limits
        self.spatial_attention_use_expressions = spatial_attention_use_expressions
        self.warp_condition_latent_dim = warp_condition_latent_dim
        self.warp_ablation_study = warp_ablation_study
        self.use_warp = use_warp
        self.use_pseudo_frameid = use_pseudo_frameid
        self.no_warp_for_pseudo = no_warp_for_pseudo
        self.eval_chunk_size = eval_chunk_size
        self.norm = norm
        self.input_noise_std = input_noise_std
        self.input_noise_position = input_noise_position
        self.input_noise_target = input_noise_target
        self.lpips_model = lpips_model
        self.return_internal_results = return_internal_results
        self.n_resblocks_expression_adapter = n_resblocks_expression_adapter
        self.n_resblocks_sr = n_resblocks_sr
        self.sr_scale = sr_scale
        self.sr_upsample = sr_upsample
        self.sr_norm = sr_norm
        self.temporal_reg = temporal_reg
        self.cam_params = cam_params
        self.dataset = data.split("-")[0] if "-" in data else data
        self.subject = data.split("-")[1] if "-" in data else None
        assert activation in ["LeakyReLU", "ReLU"]
        assert activation_out in ["LeakyReLU", "ReLU", "sigmoid"]

        # Set up LPIPS model.
        assert lpips_model in ["vgg", "alexnet"], (
            "LPIPS model '%s' is not supported." % lpips_model
        )
        self.set_up_lpips_model = set_up_lpips_model
        self._lpips_model = None
        if set_up_lpips_model:
            lpips_tfhub_path = "@spectra/metrics/lpips/net-lin_vgg_v0.1/4"
            if lpips_model == "alexnet":
                lpips_tfhub_path = "@neural-rendering/lpips/distance/1"
            self._lpips_model = hub.load(lpips_tfhub_path)
            logging.info("==> LPIPS model loaded: %s", lpips_model)

        # Set up some attributes based on the data.
        assert self.dataset in ["GNOME", "FLAME"], "Data '%s' is not supported." % data
        if self.dataset == "GNOME":
            self.exp_dim = 157
            self.rot_dim = 12
            point_dim = 6  # For GNOME data, two sets of points are sampled.

            # Set up camera params, which will be used to get rays.
            if cam_params is None:
                if image_size == "256x256":
                    self.cam_params = [250, 250, 128, 128, 256, 256]
                elif image_size == "512x557":
                    self.cam_params = [557, 557, 278, 255.5, 512, 557]
                elif image_size == "470x512":
                    self.cam_params = [470.43045, 470.43045, 255.5, 234.5, 470, 512]
            else:
                self.cam_params = cam_params
                self.cam_params[-2:] = [int(x) for x in self.cam_params[-2:]]

            # Change camera params if SR is used.
            if n_resblocks_sr > 0:
                assert sr_scale > 1
                fx, fy, cx, cy, h, w = self.cam_params
                h = int(np.ceil(h / sr_scale))
                w = int(np.ceil(w / sr_scale))
                fx /= sr_scale
                fy /= sr_scale
                cx /= sr_scale
                cy /= sr_scale
                self.cam_params = [fx, fy, cx, cy, h, w]
                logging.info(
                    "==> Use SR X%s. Camera params updated: %s",
                    sr_scale,
                    self.cam_params,
                )

            # Get near and far.
            near, far = 2.05, 4.05
            n_real_train_frames = 3470
            h, w = self.cam_params[-2:]

        elif self.dataset == "FLAME":
            self.exp_dim = 109 if use_jaw_eyes_pose else 100
            self.rot_dim = 0  # For FLAME data, not used this.
            point_dim = 3
            # Get dataset info (near and far, image size, etc.).
            data_info = FLAME_DATASET_INFO["subject" + self.subject]
            near, far = data_info["near"], data_info["far"]
            n_real_train_frames = data_info["n_real_train_frames"]
            # Get base rays direction, which will be used to get rays.
            if base_rays_d_path is None:
                data_folder = (
                    "/cns/li-d/home/huanwangx/efficient_realvatar/Data/FLAME_Train_Pseudo_subject%s"
                    % self.subject
                )
                base_rays_d_path = "%s/base_rays_d.npy" % data_folder
            with gfile.Open(base_rays_d_path, "rb") as f:
                self.base_rays_d = np.load(f)
                logging.info(
                    "==> Load given base_rays_d: %s. Shape: %s",
                    base_rays_d_path,
                    self.base_rays_d.shape,
                )
            h, w = self.base_rays_d.shape[:2]
            self.cam_params = [None, None, None, None, h, w]

        logging.info(
            "==> Dataset %s subject %s, near = %s, far = %s, image size = %s*%s,"
            " n_real_train_frames = %s",
            self.dataset,
            self.subject,
            near,
            far,
            h,
            w,
            n_real_train_frames,
        )

        # Preprocessing.
        self.point_sampler = PointSampler(near=near, far=far)
        self.positional_encoder_point = PositionalEncoder(num_freqs=num_freqs_point)
        self.positional_encoder_expression = PositionalEncoder(
            num_freqs=num_freqs_expression
        )

        # Get the input dimension of the student based on different settings.
        n_sampled_points = self.point_sampler.n_sampled_points
        pe_expansion_point = self.positional_encoder_point.dim_out
        pe_expansion_expression = self.positional_encoder_expression.dim_out
        self.dim_in = 0
        if use_ray_position:
            self.dim_in += (n_sampled_points * point_dim) * pe_expansion_point
        if use_expressions:
            self.dim_in += self.exp_dim * pe_expansion_expression
        if use_rotations:
            self.dim_in += self.rot_dim * pe_expansion_expression
        if expression_scheme in ["local_feats_v1", "local_feats_v2"]:
            self.dim_in += self.local_feat_dim
        if warp_ablation_study:
            self.dim_in += warp_condition_latent_dim * pe_expansion_point
        logging.info(
            "==> The input dimension of student is: %d (sampled_points: %s,"
            " point_dim: %s, PE_expansion_point: %s, PE_expansion_exp: %s,"
            " use_ray_position: %s, use_expressions: %s, use_rotations: %s,"
            " expression_scheme: %s)",
            self.dim_in,
            n_sampled_points,
            point_dim,
            pe_expansion_point,
            pe_expansion_expression,
            use_ray_position,
            use_expressions,
            use_rotations,
            expression_scheme,
        )

        # Build the local feature model and spatial attention model.
        if expression_scheme in ["local_feats_v1", "local_feats_v2"]:
            if expression_scheme == "local_feats_v1":
                self.local_feat_model = self._build_mlp(
                    dim_in=self.exp_dim * pe_expansion_expression,
                    dim_out=local_feat_dim * n_local_feats,
                    n_resblocks=local_feat_n_resblocks,
                    width=local_feat_width,
                    activation=activation,
                )
                logging.info("==> Build local feature model done!")
            elif expression_scheme == "local_feats_v2":
                initial_value = tf.random.normal(shape=(n_local_feats, local_feat_dim))
                self.local_feats = tf.Variable(
                    initial_value,
                    trainable=True,
                    name="local_feats_%s_%s" % (self.dataset, self.subject),
                )
                logging.info("==> Initialize local feature bank done!")

            # Build spatial attention model.
            sa_dim_in = (n_sampled_points * point_dim) * pe_expansion_point
            if spatial_attention_use_expressions:
                sa_dim_in += self.exp_dim * pe_expansion_expression
            self.spatial_attention_model = self._build_mlp(
                dim_in=sa_dim_in,
                dim_out=n_local_feats,
                n_resblocks=spatial_attention_n_resblocks,
                width=spatial_attention_width,
                activation=activation,
                activation_out="sigmoid",  # The output is weights, so use sigmoid.
            )
            logging.info("==> Build spatial attention model done!")

        # Build the head, a layer of FC.
        head = [
            tf.keras.layers.Dense(
                width, activation="LeakyReLU", input_shape=(self.dim_in,)
            ),
        ]

        self.head = tf.keras.Sequential(head)

        # Build the body, multiple residual blocks.
        body = []
        for _ in range(n_resblocks):
            body.append(ResMLP(width=width, depth=2, activation=activation))
        self.body = tf.keras.Sequential(body)

        # Build the tail, a layer of FC.
        tail = tf.keras.layers.Dense(
            dim_out, activation="LeakyReLU", input_shape=(width,)
        )
        self.tail = tf.keras.Sequential(tail)

        if freeze_student:
            self.head.trainable = False
            self.body.trainable = False
            self.tail.trainable = False
            logging.info("==> Student model frozen!")

        # Set up warp network.
        self.n_resblocks_warp_network = n_resblocks_warp_network
        if n_resblocks_warp_network > 0 or warp_ablation_study:
            # Create latent.
            initial_value = tf.random.normal(
                shape=(n_real_train_frames + 1, warp_condition_latent_dim)
            )
            self.warp_condition_latentv2 = tf.Variable(
                initial_value,
                trainable=not freeze_warp_network,
                name="warp_condition_latent_%s_%s" % (self.dataset, self.subject),
            )

        if n_resblocks_warp_network > 0:
            # Build the warp network, which warps the position of sampled points.
            # See MonoAvatar supplementary material:
            # https://augmentedperception.github.io/monoavatar/5636_supp.pdf
            warp_network_dim_in = (
                3 * self.positional_encoder_point.dim_out + warp_condition_latent_dim
            )
            warp_network_dim_out = (
                9  # 3-d log-quaternio, 3-d rotation center, 3-d translation.
            )
            warp_network = [
                tf.keras.layers.Dense(
                    width, activation=activation, input_shape=(warp_network_dim_in,)
                ),
            ]
            for _ in range(n_resblocks_warp_network):
                warp_network.append(ResMLP(width=width, depth=2, activation=activation))
            warp_network.append(
                tf.keras.layers.Dense(
                    warp_network_dim_out, activation=activation, input_shape=(width,)
                )
            )
            self.warp_networkv2 = tf.keras.Sequential(warp_network)
            logging.info("==> Build warp network done.")

            if freeze_warp_network:
                self.warp_networkv2.trainable = False
                logging.info("==> Warp network frozen!")

        # Build expression adapter network.
        if n_resblocks_expression_adapter > 0:
            expression_adapter = [
                tf.keras.layers.Dense(
                    width, activation=activation, input_shape=(self.exp_dim,)
                ),
            ]
            for _ in range(n_resblocks_expression_adapter):
                expression_adapter.append(
                    ResMLP(width=width, depth=2, activation=activation)
                )
            expression_adapter.append(
                tf.keras.layers.Dense(
                    self.exp_dim, activation=None, input_shape=(width,)
                )  # FLAME expression is beyond -1/+1, tentatively no activation here.
            )
            self.expression_adapter = tf.keras.Sequential(expression_adapter)
            logging.info("==> Build expression adapter network done.")

        # Set data format.
        self._data_format = "RayFormat"  # Each example is a ray.

        # Build SR model．
        if n_resblocks_sr > 0:
            self.sr_network_head, self.sr_network_body, self.sr_network_tail = (
                self._build_sr_network(
                    sr_scale=sr_scale,
                    n_resblocks=n_resblocks_sr,
                    width=width_sr,
                )
            )
            logging.info("==> Build SR network done!")

    def _apply_rotation_limit(self, rotation: tf.Tensor, limit: tf.Tensor) -> tf.Tensor:
        """Apply limits to rotations. Refers to FLAME code.

        Args:
          rotation: Axis angle. Shape [num_rays, 3].
          limit: Limits for the axis angle. Shape [3, 2].

        Returns:
          rotation: Normalized axis angle.
        """
        r_min, r_max = limit[:, 0], limit[:, 1]
        r_min = tf.reshape(r_min, [1, 3])
        r_max = tf.reshape(r_max, [1, 3])
        diff = r_max - r_min  # [1, 3]
        return r_min + (tf.math.tanh(rotation) + 1) / 2 * diff

    def apply_rotation_limits_flame(
        self, jaw_pose: tf.Tensor, eyes_pose: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply limits to rotations.

        Args:
          jaw_pose: Jaw axis angle. Shape [num_rays, 3].
          eyes_pose: Eyes axis angle. Shape [num_rays, 6].

        Returns:
          jaw_pose: Normalized jaw pose.
          eyes_pose: Normalized eyes pose.
        """
        # These limits are from FLAME code.
        eye_limit = ((-50, 50), (-50, 50), (-0.1, 0.1))
        jaw_limit = ((-5, 60), (-0.1, 0.1), (-0.1, 0.1))
        eye_limit = tf.constant(eye_limit) / 180 * np.pi
        jaw_limit = tf.constant(jaw_limit) / 180 * np.pi
        jaw_pose = self._apply_rotation_limit(jaw_pose, jaw_limit)
        eye_r = self._apply_rotation_limit(eyes_pose[:, :3], eye_limit)
        eye_l = self._apply_rotation_limit(eyes_pose[:, 3:], eye_limit)
        eyes_pose = tf.concat([eye_r, eye_l], axis=-1)
        return jaw_pose, eyes_pose

    def _build_mlp(
        self,
        dim_in: int,
        dim_out: int,
        n_resblocks: int,
        width: int = 128,
        activation: str = "LeakyReLU",
        activation_out: Optional[str] = None,
        residual: bool = True,
    ) -> tf.keras.Model:
        """Build a residual MLP model.

        Args:
          dim_in: Input dimension.
          dim_out: Output dimension.
          n_resblocks: Num of resblocks. Each residual block has 2 layers.
          width: Num of neurons of internal layers.
          activation: Activation function.
          activation_out: Output activation function.
          residual: Whether to use residual.

        Returns:
          An MLP model.
        """
        res_mlp = [
            tf.keras.layers.Dense(width, activation=activation, input_shape=(dim_in,))
        ]
        for _ in range(n_resblocks):
            res_mlp.append(
                ResMLP(width=width, depth=2, activation=activation, residual=residual)
            )
        if activation_out is not None:
            activation_out = activation
        res_mlp.append(
            tf.keras.layers.Dense(
                dim_out, activation=activation_out, input_shape=(width,)
            )
        )
        return tf.keras.Sequential(res_mlp)

    def _build_sr_network(
        self,
        sr_scale: int,
        n_resblocks: int,
        width: int,
    ) -> List[tf.keras.Sequential]:
        """Build a SR model.

        Refer to EDSR:
        https://github.com/sanghyun-son/EDSR-PyTorch/blob/8dba5581a7502b92de9641eb431130d6c8ca5d7f/src/model/edsr.py#L17

        Args:
          sr_scale: Upsampling scale.
          n_resblocks: Num of resblocks. Each residual block has 2 layers.
          width: Num of neurons of internal layers.

        Returns:
          Head, body, tail of the SR network.
        """
        # Build SR head. One conv layer.
        sr_network_head = tf.keras.Sequential([conv(width)])

        # Build SR body. Several residual blocks.
        sr_network_body = []
        for _ in range(n_resblocks):
            sr_network_body.append(ResBlock(width=width, depth=2, norm=self.sr_norm))
        sr_network_body.append(conv(width))
        sr_network_body = tf.keras.Sequential(sr_network_body)

        # Build SR tail. Upsampling + one conv layer. Upsampling is (one conv +
        # PixelShuffle) * log2(sr_scale)
        tail = []
        assert self.sr_upsample in ["PixelShuffle", "TransposeConv"], (
            "Incorrect sr_upsample: %s" % self.sr_upsample
        )
        if (sr_scale & (sr_scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(sr_scale, 2))):
                if self.sr_upsample == "PixelShuffle":
                    tail.append(conv(width * 4))
                    tail.append(PixelShuffle(scale=2))  # Upsample by x2.
                    if self.sr_norm == "bn":
                        tail.append(bn(True))
                elif self.sr_upsample == "TransposeConv":
                    tail.append(transpose_conv(width))  # Upsample by x2.
                    if self.sr_norm == "bn":
                        tail.append(bn(True))
        else:
            raise NotImplementedError("Incorrect sr_scale: %s" % sr_scale)
        tail.append(conv(width=3))  # Map the feature to RGB.
        sr_network_tail = tf.keras.Sequential(tail)

        return [sr_network_head, sr_network_body, sr_network_tail]

    def _inner_forward(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the inner forward computation of the model.

        Args:
          x: The tensor that is immediately fed into the NeLF model.

        Returns:
          The output tensor of the NeLF model.
        """
        x = self.head(x)
        x = self.body(x) + x
        return self.tail(x)

    def _sr_forward(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward computation of the SR model.

        Args:
          x: The tensor that is immediately fed into the SR model. [B, H, W, C].

        Returns:
          The output tensor of the SR model. [B, H*sr_scale, W*sr_scale, C].
        """
        x = self.sr_network_head(x)
        x = self.sr_network_body(x) + x
        return self.sr_network_tail(x)

    def _add_normal_noise(
        self, x: tf.Tensor | None, std: float = 0.0
    ) -> tf.Tensor | None:
        """Add normal noise to input tensor.

        Args:
          x: Input tensor.
          std: Std of the normal noise.

        Returns:
          The noised tensor.
        """
        if x is None:
            return None
        return tf.random.normal(shape=x.shape, stddev=std) + x

    def call(
        self,
        input_tensor: tf.Tensor,
        training: bool = False,
        step: Optional[int] = None,
        image_batch_shape: Optional[Tuple[int, int, int, int]] = None,
    ) -> Any:
        """Performs the forward computation of the model.

        Args:
          input_tensor: Shape [num_rays, dim_input], where dim_input depends on
            different cases.
          training: training or testing.
          step: Training step.
          image_batch_shape: The shape of a batch data when using ImageFormat.

        Returns:
          A tensor with shape [num_rays, 3], which means the RGB color of the rays,
          and possibly, the immediate model input, which will be used for
          regularization loss calculation.
        """
        num_rays, feat_dim_per_ray = input_tensor.shape[0], input_tensor.shape[1]
        rays_e, rays_r = None, None
        rays_o_wrt_shoulder, rays_d_wrt_shoulder = None, None
        internal_results = {}  # Return intermediate features.

        # Get feature dim per ray based on the dataset. For GNOME data, rot_dim and
        # n_verts_per_ray are not really useful, but for back-compatibility, we keep
        # them here. No harm.
        rays_o_dim, rays_d_dim = 3, 3
        n_verts_per_ray = 64
        fid_dim = 1
        feat_dim_per_ray_flame = (rays_o_dim + rays_d_dim) + fid_dim + self.exp_dim
        feat_dim_per_ray_gnome = (
            (rays_o_dim + rays_d_dim) * 2 + fid_dim + self.exp_dim + self.rot_dim
        )  # GNOME also uses shoulder, samples two sets of points, so * 2.
        assert feat_dim_per_ray in [
            (rays_o_dim + rays_d_dim) + fid_dim,
            feat_dim_per_ray_flame,
            feat_dim_per_ray_gnome,
            feat_dim_per_ray_gnome + n_verts_per_ray,
        ], f"Incorrect feat_dim_per_ray ({feat_dim_per_ray}), please check!"

        # (Next, parse the data, rays_o, rays_d, rays_e, etc. from the input.)

        # This case is a naive scheme, which is not used as the final solution. Kept
        # here for potential use in code unit test.
        if feat_dim_per_ray == (rays_o_dim + rays_d_dim) + fid_dim:
            rays_o, rays_d, rays_o_wrt_shoulder, rays_d_wrt_shoulder, fid = tf.split(
                input_tensor, [3, 3, 3, 3, 1], axis=-1
            )
            pts1 = self.point_sampler(rays_o=rays_o, rays_d=rays_d, training=training)
            pts2 = self.point_sampler(
                rays_o=rays_o_wrt_shoulder,
                rays_d=rays_d_wrt_shoulder,
                training=training,
            )
            pts = tf.concat([pts1, pts2], axis=-1)
            model_input = self.positional_encoder_point(pts)  # [num_rays, dim_in]

        else:
            # GNOME data.
            if feat_dim_per_ray == feat_dim_per_ray_gnome:
                (
                    rays_o,
                    rays_d,
                    rays_e,
                    rays_r,
                    rays_o_wrt_shoulder,
                    rays_d_wrt_shoulder,
                    fid,
                ) = tf.split(
                    input_tensor,
                    [3, 3, self.exp_dim, self.rot_dim, 3, 3, 1],
                    axis=-1,
                )
                fid = (
                    tf.cast(fid[:, 0], tf.int32) + 1
                )  # [num_rays], frame_id starts from -1.
                if self.use_pseudo_frameid:
                    fid = fid * 0  # All pseudo frames share the same id 0.

            # GNOME data with knn index, not used anymore. kept for back-compatibility
            elif feat_dim_per_ray == feat_dim_per_ray_gnome + n_verts_per_ray:
                (
                    rays_o,
                    rays_d,
                    rays_e,
                    rays_r,
                    _,
                    rays_o_wrt_shoulder,
                    rays_d_wrt_shoulder,
                    fid,
                ) = tf.split(
                    input_tensor,
                    [3, 3, self.exp_dim, self.rot_dim, n_verts_per_ray, 3, 3, 1],
                    axis=-1,
                )
                fid = (
                    tf.cast(fid[:, 0], tf.int32) + 1
                )  # [num_rays], frame_id starts from -1.
                if self.use_pseudo_frameid:
                    fid = fid * 0  # All pseudo frames share the same id 0.

            # FLAME data.
            elif feat_dim_per_ray == feat_dim_per_ray_flame:
                (
                    rays_o,
                    rays_d,
                    rays_e,
                    fid,
                ) = tf.split(
                    input_tensor,
                    [3, 3, self.exp_dim, 1],
                    axis=-1,
                )
                fid = (
                    tf.cast(fid[:, 0], tf.int32) + 1
                )  # [num_rays], frame_id starts from -1.
                if self.use_pseudo_frameid:
                    fid = fid * 0  # All pseudo frames share the same id 0.

                # Normalize the jaw and eyes pose.
                if self.use_jaw_eyes_pose and self.apply_rotation_limits:
                    jaw_pose, eyes_pose = rays_e[:, -9:-6], rays_e[:, -6:]
                    jaw_pose, eyes_pose = self.apply_rotation_limits_flame(
                        jaw_pose, eyes_pose
                    )
                    rays_e = tf.concat([rays_e[:, :100], jaw_pose, eyes_pose], axis=-1)

            # Apply expression adapter network, if any. Not working so far.
            if self.n_resblocks_expression_adapter > 0:
                rays_e = self.expression_adapter(rays_e)

            # Sample points.
            pts = self.point_sampler(rays_o=rays_o, rays_d=rays_d, training=training)
            if rays_o_wrt_shoulder is not None:
                pts2 = self.point_sampler(
                    rays_o=rays_o_wrt_shoulder,
                    rays_d=rays_d_wrt_shoulder,
                    training=training,
                )
                pts = tf.concat([pts, pts2], axis=-1)  # [num_rays, num_points * 3]
            num_points = pts.shape[1] // 3

            # Warp the position of sampled points. Normally, its input is pts,
            # output is another pts (shape exactly same). In ablation study
            # (`warp_ablation_study`), the warp network is not used. Instead, directly
            # concat the latent to pts.
            now = "train" if training else "test"
            if now in self.use_warp:
                is_pseudo_batch = tf.math.reduce_sum(fid) == 0
                if is_pseudo_batch and self.no_warp_for_pseudo:
                    pass
                else:
                    latent = tf.gather(
                        self.warp_condition_latentv2, fid, axis=0
                    )  # [num_rays, warp_condition_latent_dim]
                    assert latent.shape == (num_rays, self.warp_condition_latent_dim)
                    if self.warp_ablation_study:
                        # At reviewer's request, remove warp network and only keep the
                        # warp_condition_latent (this increases the NeLF model input dim).
                        pts = tf.concat([pts, latent], axis=-1)
                    else:
                        latent = tf.broadcast_to(
                            latent[:, None, :],
                            [num_rays, num_points, self.warp_condition_latent_dim],
                        )
                        latent = tf.reshape(latent, [num_rays * num_points, -1])

                        pts_ = tf.reshape(pts, [num_rays * num_points, 3])
                        encoded_pts = self.positional_encoder_point(pts_)

                        warp_network_input = tf.concat(
                            [encoded_pts, latent],
                            axis=-1,
                        )  # [num_rays * num_points, dim_latent]

                        assert (
                            self.n_resblocks_warp_network > 0
                        ), "n_resblocks_warp_network must be > 0 to build the warp network."
                        warp_out = self.warp_networkv2(warp_network_input)
                        rot, rot_center, trans = tf.split(warp_out, [3, 3, 3], axis=-1)
                        rot = expq(rot)
                        pts_ = q.rotate(pts_ + rot_center, rot) - rot_center + trans
                        pts = tf.reshape(pts_, [num_rays, num_points * 3])

            # Add noise to raw input.
            if (
                training
                and self.input_noise_std is not None
                and self.input_noise_position == "raw_input"
            ):
                if "points" in self.input_noise_target:
                    pts = self._add_normal_noise(pts, self.input_noise_std)
                if "expression" in self.input_noise_target:
                    rays_e = self._add_normal_noise(rays_e, self.input_noise_std)
                    rays_r = self._add_normal_noise(rays_r, self.input_noise_std)

            # Assemble the model input based on different settings.
            model_input = tf.reshape(tf.convert_to_tensor(()), (num_rays, 0))
            if self.use_ray_position:
                model_input = tf.concat(
                    [model_input, self.positional_encoder_point(pts)], axis=-1
                )
            if self.use_expressions and rays_e is not None:
                model_input = tf.concat(
                    [model_input, self.positional_encoder_expression(rays_e)], axis=-1
                )
            if self.use_rotations and rays_r is not None:
                model_input = tf.concat(
                    [model_input, self.positional_encoder_expression(rays_r)], axis=-1
                )

            # Use local feature model and spatial attention model.
            if self.expression_scheme in ["local_feats_v1", "local_feats_v2"]:
                # Get spatial attention weights.
                sa_input = self.positional_encoder_point(pts)
                if self.spatial_attention_use_expressions:
                    sa_input = tf.concat(
                        [sa_input, self.positional_encoder_expression(rays_e)], axis=-1
                    )
                attn_weights = self.spatial_attention_model(sa_input)

                # Get local features.
                if self.expression_scheme == "local_feats_v1":
                    if training:
                        local_feats = self.local_feat_model(
                            self.positional_encoder_expression(rays_e)
                        )
                        local_feats = tf.reshape(
                            local_feats,
                            [num_rays, self.n_local_feats, self.local_feat_dim],
                        )
                        # Apply attention weights to local features.
                        # [num_rays,n_local_feats,local_feat_dim]*[num_rays,n_local_feats,1]
                        local_feat = local_feats * attn_weights[..., None]
                        local_feat = tf.math.reduce_sum(local_feat, axis=1)
                    else:
                        local_feats = self.local_feat_model(
                            self.positional_encoder_expression(rays_e[0:1])
                        )
                        local_feats = tf.reshape(
                            local_feats, [self.n_local_feats, self.local_feat_dim]
                        )
                        local_feat = attn_weights @ local_feats
                        if self.return_internal_results:
                            internal_results["attn_weights"] = attn_weights
                            internal_results["local_feats"] = local_feats
                elif self.expression_scheme == "local_feats_v2":
                    local_feat = attn_weights @ self.local_feats

                # Add noise to the input of NeLF model. Not working so far.
                if (
                    training
                    and self.input_noise_std is not None
                    and self.input_noise_position == "nelf_input"
                ):
                    if "points" in self.input_noise_target:
                        model_input = self._add_normal_noise(
                            model_input, self.input_noise_std
                        )
                    if "expression" in self.input_noise_target:
                        local_feat = self._add_normal_noise(
                            local_feat, self.input_noise_std
                        )

                # Concat local feature into model input.
                model_input = tf.concat([model_input, local_feat], axis=-1)

        # Forward.
        return self._inner_forward(model_input), internal_results

    def get_model_outputs(
        self,
        data_batch: dict[str, tf.Tensor],
        training: bool,
        step: Optional[int] = None,
    ) -> dict[str, Any]:
        """This function handles the inference for the model and provide the results for the loss calculation.

        Args:
          data_batch: A dictionary holding the data batch.
          training: Training flag that is passed into layers.
          step: The current training step.

        Returns:
          A dictionary holding model_outputs with string keys and tensor values.
        """
        # Read data and reshape. The keys in data indicate different data formats:
        # If C2W_KEY is in keys, the data format is ImageFormat, i.e., each tf
        # example is a single image (along with its c2w, expression, head rotation,
        # etc.). If RAY_DATA_KEY is in keys, the data format is RayFormat,
        # i.e., each tf example is a ray.

        assert (
            C2W_KEY in data_batch or RAY_DATA_KEY in data_batch
        ), "%s or %s MUST be in the data batch keys." % (C2W_KEY, RAY_DATA_KEY)
        cam_params = (
            self.cam_params
        )  # Loading cam_params from data_batch may be better.
        fx, fy, cx, cy, h, w = cam_params

        # ImageFormat data. Can be used in training and testing.
        if C2W_KEY in data_batch:
            c2w = data_batch[C2W_KEY]
            batch_size = data_batch[C2W_KEY].shape[0]  # Num of examples per batch.
            num_images_per_example = 1
            self._data_format = "ImageFormat"  # Each example is an image.
            if c2w.shape[1] > 12:
                assert c2w.shape[1] % 12 == 0, "c2w.shape[1] must be a multiple of 12."
                num_images_per_example = c2w.shape[1] // 12
                self._data_format = "%dImageFormat" % num_images_per_example
            num_images = batch_size * num_images_per_example  # Num images per batch.

            # Resize if necessary.
            resize_factor = data_batch.get("resize_factor", None)
            if resize_factor is not None:  # Resize image during rendering.
                assert self.dataset == "GNOME", "Only GNOME dataset can be resized now."
                fx = fx / resize_factor
                fy = fy / resize_factor
                cx = cx / resize_factor
                cy = cy / resize_factor
                h = int(h / resize_factor)
                w = int(w / resize_factor)
                cam_params = [fx, fy, cx, cy, h, w]

            # Set up rays origin and direction
            c2w = tf.reshape(data_batch[C2W_KEY], [num_images, 3, 4])

            # Set up input tensor based on the dataset.
            if self.dataset == "GNOME":
                # Set up origin and direction.
                rays_o, rays_d = self._get_rays_origin_direction(
                    c2w, cam_params=cam_params
                )
                input_tensor = tf.concat([rays_o, rays_d], axis=-1)

                # Set up expression and rotation.
                if EXP_KEY in data_batch:
                    exp = tf.reshape(data_batch[EXP_KEY], [num_images, 1, self.exp_dim])
                    rays_e = tf.broadcast_to(exp, [num_images, h * w, self.exp_dim])
                    input_tensor = tf.concat([input_tensor, rays_e], axis=-1)
                if ROT_KEY in data_batch:
                    rot = tf.reshape(data_batch[ROT_KEY], [num_images, 1, self.rot_dim])
                    rays_r = tf.broadcast_to(rot, [num_images, h * w, self.rot_dim])
                    input_tensor = tf.concat([input_tensor, rays_r], axis=-1)

                # Set up rays origin and direction w.r.t. shoulder.
                c2w_wrt_shoulder = tf.reshape(
                    data_batch[C2W_WRT_SHOULDER_KEY], [num_images, 3, 4]
                )
                rays_o_wrt_shoulder, rays_d_wrt_shoulder = (
                    self._get_rays_origin_direction(
                        c2w_wrt_shoulder, cam_params=cam_params
                    )
                )
                input_tensor = tf.concat(
                    [input_tensor, rays_o_wrt_shoulder, rays_d_wrt_shoulder], axis=-1
                )

            elif self.dataset == "FLAME":
                rays_o, rays_d = self._get_rays_origin_direction_FLAME(
                    c2w, cam_params=cam_params
                )
                exp = tf.reshape(data_batch[EXP_KEY], [num_images, 1, self.exp_dim])
                rays_e = tf.broadcast_to(exp, [num_images, h * w, self.exp_dim])
                input_tensor = tf.concat(
                    [
                        rays_o,
                        rays_d,
                        rays_e,
                    ],
                    axis=-1,
                )  # Shape: [B, h * w, ...]

            # Set up frame id.
            fid = tf.cast(data_batch["fid"], tf.float32)
            fid = tf.reshape(fid, [num_images, 1, 1])
            fid = tf.broadcast_to(fid, [num_images, h * w, 1])
            input_tensor = tf.concat([input_tensor, fid], axis=-1)

            # Reshape to [num_rays, feature_dim_per_ray].
            input_tensor = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])

        # RayFormat data. Can be used in training ONLY. If both RayFormat and
        # ImageFormat data found in the batch, RayFormat data is of higher priority.
        if RAY_DATA_KEY in data_batch and training:
            self._data_format = "RayFormat"
            input_tensor = tf.concat(
                [data_batch[RAY_DATA_KEY][:, :6], data_batch[RAY_DATA_KEY][:, 9:]],
                axis=-1,
            )

        # (Next is model forward. Note: the input tensor before being fed into the
        # NeLF model is always in the shape [num_rays, feature_dim_per_ray].)

        # Training model forward.
        if step is not None and step == 0:
            tf.print(
                f"is_training: {training}, input_tensor shape: {input_tensor.shape}"
            )
        if training:
            output, internal_results = self.call(input_tensor, training=True, step=step)
            # For ImageFormat, reshape back to [num_images, num_rays_per_image, 3].
            if self.n_resblocks_sr > 0:
                pattern = r"\d*ImageFormat"
                assert re.fullmatch(pattern, self._data_format), (
                    "Invalid data format: %s" % self._data_format
                )
                output = tf.reshape(output, [num_images, h, w, 3])
                output = self._sr_forward(output)
                h, w = output.shape[1:3]
                output = tf.reshape(output, [num_images, -1, 3])
            rets = {
                "output": output,
                "internal_results": internal_results,
                "height": h,  # Needed to calculate lpips loss in the loss fn.
                "width": w,
                "lpips_model": self._lpips_model,
            }
            if self.temporal_reg:
                input_tensor = input_tensor[:, :-1]  # Remove fid.
                rets["input"] = tf.reshape(
                    input_tensor, [num_images, -1, input_tensor.shape[-1]]
                )
                rets["num_images_per_example"] = num_images_per_example

            return rets

        # For testing, to avoid OOM, split to multiple runs.
        else:
            output, internal_results, chunk_size = [], {}, self.eval_chunk_size
            num_rays = input_tensor.shape[0]
            for i in range(0, num_rays, chunk_size):
                part_input_tensor = input_tensor[i : i + chunk_size]
                part_output_tensor, part_internal_results = self.call(
                    part_input_tensor, training=False
                )
                output.append(part_output_tensor)
                if self.return_internal_results:
                    for k, v in part_internal_results.items():
                        if k in internal_results:
                            internal_results[k] += [v]
                        else:
                            internal_results[k] = [v]
            output = tf.concat(output, axis=0)
            # Apply SR, if any.
            if self.n_resblocks_sr > 0:
                output = tf.reshape(output, [num_images, h, w, 3])
                output = self._sr_forward(output)
                h, w = output.shape[1:3]

            # Process internal results.
            if self.return_internal_results:
                for k, v in internal_results.items():
                    if k == "local_feats":
                        internal_results[k] = v[0]  # local_feats are the same.
                    else:
                        internal_results[k] = tf.concat(v, axis=0)

            fid = int(
                data_batch["fid"][0]  # fid is a one-element tensor.
            )  # Return frame ID for testing because we need to make a video later.

            # Maintain the shape [num_images, num_rays_per_image, 3].
            output = tf.reshape(output, [num_images, -1, 3])

            return {
                "output": output,
                "fid": fid,
                "internal_results": internal_results,
                "height": h,
                "width": w,
                "lpips_model": self._lpips_model,
            }

    def get_model_targets(
        self,
        data_batch: dict[str, tf.Tensor],
        training: bool,
    ) -> dict[str, tf.Tensor]:
        """This function prepares the supervision target for the model.

        Args:
          data_batch: A dictionary holding the data batch.
          training: Training flag that is passed into layers.

        Returns:
          A dictionary holding model targets.
        """
        # Note that keys added here will be used to add losses against matching keys
        # in model_outputs.

        if C2W_KEY in data_batch:
            c2w = data_batch[C2W_KEY]
            batch_size = data_batch[C2W_KEY].shape[0]  # Num of examples per batch.
            num_images_per_example = 1
            if c2w.shape[1] > 12:
                assert c2w.shape[1] % 12 == 0, "c2w.shape[1] must be a multiple of 12."
                num_images_per_example = c2w.shape[1] // 12
            num_images = batch_size * num_images_per_example  # Num images per batch.
            rays_c = data_batch[RGB_KEY]  # [B, h * w, 3]
            rays_c = tf.reshape(rays_c, [num_images, -1, 3])  # [B, h * w, 3]
        if RAY_DATA_KEY in data_batch and training:
            rays_c = data_batch[RAY_DATA_KEY][:, 6:9]
        return {"target": rays_c}

    def get_image_summaries(
        self,
        model_outputs: dict[str, tf.Tensor],
        model_targets: dict[str, tf.Tensor],
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Returns image summaries for the model that will be displayed on TensorBoard."""
        # Get image summaries.
        if training:
            return {}  # No image visualize during training.

        # Get image height/width. If eval_h/eval_w explicitly provided, use them.
        h, w = self.cam_params[-2:]
        if self.n_resblocks_sr > 0:
            h, w = h * self.sr_scale, w * self.sr_scale
        if hasattr(self, "eval_h") and hasattr(self, "eval_w"):
            h, w = self.eval_h, self.eval_w

        fid = model_outputs["fid"]
        pred_rays_c = model_outputs["output"]  # Predicted rays color.
        pred_rays_c = tf.reshape(pred_rays_c, [-1, h, w, 3])

        gt_rays_c = model_targets["target"]  # Ground-truth rays color.
        gt_rays_c = tf.reshape(gt_rays_c, [-1, h, w, 3])

        diff = tf.math.reduce_mean(tf.math.abs(gt_rays_c - pred_rays_c), axis=-1)
        diff = tf.broadcast_to(diff[..., None], gt_rays_c.shape)

        # Calculate metrics.
        psnr = tf.image.psnr(pred_rays_c, gt_rays_c, max_val=1)
        ssim = tf.image.ssim(pred_rays_c, gt_rays_c, max_val=1)
        lpips = tf.constant([0.0], dtype=tf.float32)
        if self.set_up_lpips_model:
            lpips = self._lpips_model(pred_rays_c, gt_rays_c)
        return {
            "image_pred": pred_rays_c,
            "image_gt": gt_rays_c,
            "image_diff": diff,
            "fid": fid,  # Return frame id for making a video later.
            "psnr_tf": psnr,
            "ssim_tf": ssim,
            "lpips_tf": lpips,
            "internal_results": model_outputs["internal_results"],
        }

    def _get_rays_origin_direction(
        self,
        c2w: tf.Tensor,
        cam_params: list[float],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gets rays origin and direction from c2w and camera params.

        Args:
          c2w: C2w matrix [num_frames, 3, 4].
          cam_params: Camera intrinsics. A list [fx, fy, cx, cy, h, w].

        Returns:
          rays_o: Rays origin, shape [num_frames, num_rays, 3].
          rays_d: Rays direction, shape [num_frames, num_rays, 3].
        """
        fx, fy, cx, cy, h, w = cam_params
        batch_size = c2w.shape[0]
        h = tf.cast(h, tf.int32)
        w = tf.cast(w, tf.int32)
        x, y = tf.meshgrid(
            tf.range(w, dtype=tf.float32) + 0.5, tf.range(h, dtype=tf.float32) + 0.5
        )
        x = (x - cx) / fx  # [h, w, 1]
        y = (y - cy) / fy  # [h, w, 1]

        # Get rays origin
        rays_o = c2w[:, :3, 3][:, None, None, :]  # [B, 1, 1, 3]
        rays_o = tf.broadcast_to(rays_o, [batch_size, h, w, 3])  # [B, h, w, 3]

        # Get rays direction
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        rays_d = tf.stack([x, y, tf.ones_like(x)], axis=-1)  # [h, w, 3]
        rays_d = tf.linalg.normalize(rays_d, axis=-1)[0]  # [h, w, 3]
        rays_d = tf.broadcast_to(rays_d, [batch_size, h, w, 3])  # [B, h, w, 3]
        rays_d = (
            tf.tile(
                rotation[:, None, None, :, :], [1, h, w, 1, 1]  # [B, 1, 1, 3, 3]
            )  # [B, h, w, 3, 3]
            @ rays_d[..., None]  # [B, h, w, 3, 1]
        )  # [B, h, w, 3, 1]
        rays_d = rays_d[..., 0]  # [B, h, w, 3]

        # Reshape to the uniform data layout [B, num_rays, 3]
        rays_o = tf.reshape(rays_o, [batch_size, h * w, 3])
        rays_d = tf.reshape(rays_d, [batch_size, h * w, 3])
        return rays_o, rays_d

    def _get_rays_origin_direction_FLAME(
        self,
        c2w: tf.Tensor,
        cam_params: list[float],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gets rays origin and direction from c2w and camera params for FLAME data.

        Args:
          c2w: C2w matrix [num_frames, 3, 4].
          cam_params: Camera intrinsics. A list [fx, fy, cx, cy, h, w].

        Returns:
          rays_o: Rays origin, shape [num_frames, num_rays, 3].
          rays_d: Rays direction, shape [num_frames, num_rays, 3].
        """
        batch_size = c2w.shape[0]
        h, w = cam_params[-2:]

        # Get rays origin, in batch.
        rays_o = c2w[:, :3, 3][:, None, None, :]  # [B, 1, 1, 3]
        rays_o = tf.broadcast_to(rays_o, [batch_size, h, w, 3])  # [B, h, w, 3]

        # Get rays direction, in batch.
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        rays_d = self.base_rays_d[None]  # [1, h, w, 3]
        rays_d = tf.broadcast_to(rays_d, [batch_size, h, w, 3])  # [B, h, w, 3]
        rays_d = (
            tf.tile(
                rotation[:, None, None, :, :], [1, h, w, 1, 1]  # [B, 1, 1, 3, 3]
            )  # [B, h, w, 3, 3]
            @ rays_d[..., None]  # [B, h, w, 3, 1]
        )  # [B, h, w, 3, 1]
        rays_d = rays_d[..., 0]  # [B, h, w, 3]

        # Reshape to the uniform data layout [B, num_rays, 3].
        rays_o = tf.reshape(rays_o, [batch_size, h * w, 3])
        rays_d = tf.reshape(rays_d, [batch_size, h * w, 3])
        return rays_o, rays_d


def expq(quat: tf.Tensor, eps: float = 1e-8, is_pure: bool = True) -> tf.Tensor:
    """Computes the quaternion exponential.

    References:
      https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

    Args:
      quat: the quaternion in (x,y,z,w) format or (x,y,z) if is_pure is True.
      eps: an epsilon value for numerical stability.
      is_pure: is True assumes q is in (x,y,z) format with only the pure part of
        the quaternion.

    Returns:
      The exponential of quat.
    """
    if is_pure:
        s = tf.zeros_like(quat[..., -1:])
        v = quat
    else:
        v, s = tf.split(quat, (3, 1), axis=-1)

    norm_v = tf.linalg.norm(v, axis=-1, keepdims=True)
    exp_s = tf.exp(s)
    w = tf.cos(norm_v)
    xyz = tf.sin(norm_v) * v / tf.maximum(norm_v, eps * tf.ones_like(norm_v))
    return exp_s * tf.concat((xyz, w), axis=-1)
