# Code from LightAvatar-TensorFlow: https://github.com/MingSun-Tse/lightavatar-tensorflow

"""Model library for efficient realvatar."""

import math
import sys
from typing import Any, List, Optional, Tuple

from absl import logging
import numpy as np
import torch
import torch.nn as nn
import time
from torchinfo import summary

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


class PointSampler(nn.Module):
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

    def forward(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        # Decide where to sample along each ray. Under the logic, all rays will be
        # sampled at the same times.
        t_vals = torch.linspace(0.0, 1.0, self.n_sampled_points).to(device="cuda")  # [n_sampled_points]

        # Linearly sample points between 'near' and 'far'.
        # Same linear sampling coefficients (t_vals) will be used for all rays.
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals

        # Perturb sampling points along each ray during training
        if self.perturb and training:
            # Get intervals between samples
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[None, -1]], -1)
            lower = torch.concat([z_vals[None, 0], mids], -1)  # [n_sampled_points]
            # Stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [n_sampled_points]

        # Get points
        z_vals = z_vals[None]
        # [num_rays, 1, 3] + [num_rays, 1, 3] * [1, n_sampled_points, 1]
        # -> [num_rays, n_sampled_points, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = torch.reshape(pts, [pts.shape[0], -1])  # Concat all points along a ray
        return pts


class PositionalEncoder(nn.Module):
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
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs).to(device="cuda")
        self.scale = scale

    @property
    def dim_out(self) -> int:
        # +1 for identity.
        return 2 * self.num_freqs + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A vectorized sinusoidal encoding.

        Args:
          x: the input features to encode with shape [..., C].

        Returns:
          A tensor containing the encoded features.
        """
        freqs = self.freq_bands
        batch_shape = x.shape[:-1]
        batch_ones = [1] * len(batch_shape)

        freqs = torch.reshape(freqs, (*batch_ones, self.num_freqs, 1))  # (*, F, 1).
        x_expanded = torch.unsqueeze(x, dim=-2)  # (*, 1, C).
        # Will be broadcasted to shape (*B, F, C).
        angles = self.scale * x_expanded * freqs

        # The shape of the features is (*B, F, 2, C) so that when we reshape it
        # it matches the ordering of the original NeRF code.
        features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-2)

        batch_shape_tf = x.shape[:-1]
        feats_shape = np.concatenate(
            [batch_shape_tf, [2 * self.num_freqs * x.shape[-1]]], axis=0
        )
        features = torch.reshape(features, tuple(feats_shape))

        # Prepend the original signal for the identity.
        features = torch.concat([x, features], dim=-1)
        return features


def fc(width: int):
    """Feedforward layer."""
    return nn.Linear(width, width)


def conv(width: int):
    """Standard 3x3 conv layer with SAME padding."""
    return nn.Conv2d(width, width, kernel_size=3, padding=1)


def leaky_relu(alpha: float):
    """Leaky ReLU layer."""
    return nn.LeakyReLU(negative_slope=0.2)


def relu():
    """ReLU layer."""
    return nn.ReLU()


def bn(synchronized: bool):
    """Batch Normalization layer."""
    return nn.BatchNorm2d(synchronized=synchronized)


class ResMLP(nn.Module):
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
        self._body = nn.Sequential(*modules)
        self._residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._residual:
            return self._body(x) + x
        else:
            return self._body(x)


class ResBlock(nn.Module):
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
        self._body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._body(x) + x


def transpose_conv(width: int):
    """Transpose 3x3 conv layer with SAME padding. Used for SR x2 upsampling."""
    return nn.ConvTranspose2d(width, width, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
    # return nn.ConvTranspose2d(width, width, kernel_size=(3, 3))



class Student(nn.Module):
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

        # Set up some attributes based on the data.
        assert self.dataset in ["GNOME", "FLAME"], "Data '%s' is not supported." % data

        # Preprocessing.
        self.point_sampler = PointSampler(near=2.0, far=4.0, n_sampled_points=8)
        self.positional_encoder_point = PositionalEncoder(num_freqs=num_freqs_point)
        self.positional_encoder_expression = PositionalEncoder(
            num_freqs=num_freqs_expression
        )

        # Get the input dimension of the student based on different settings.
        n_sampled_points = self.point_sampler.n_sampled_points
        pe_expansion_point = self.positional_encoder_point.dim_out
        pe_expansion_expression = self.positional_encoder_expression.dim_out
        self.dim_in = 0
        point_dim = 3  # Dimension of a point.
        self.exp_dim = 109
        self.rot_dim = 12

        if use_ray_position:
            self.dim_in += (n_sampled_points * point_dim) * pe_expansion_point
        if use_expressions:
            self.dim_in += self.exp_dim * pe_expansion_expression
        if use_rotations:
            self.dim_in += self.rot_dim * pe_expansion_expression
        if expression_scheme in ["local_feats_v1", "local_feats_v2"]:
            self.dim_in += self.local_feat_dim
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
            nn.Linear(self.dim_in, width),
        ]

        self.head = nn.Sequential(*head)

        # Build the body, multiple residual blocks.
        body = []
        for _ in range(n_resblocks):
            body.append(ResMLP(width=width, depth=2, activation=activation))
        self.body = nn.Sequential(*body)

        # Build the tail, a layer of FC.
        tail = nn.Linear(width, dim_out)
        self.tail = nn.Sequential(tail)

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

    def _build_mlp(
        self,
        dim_in: int,
        dim_out: int,
        n_resblocks: int,
        width: int = 128,
        activation: str = "LeakyReLU",
        activation_out: Optional[str] = None,
        residual: bool = True,
    ):
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
        res_mlp = [nn.Linear(dim_in, width)]
        for _ in range(n_resblocks):
            res_mlp.append(
                ResMLP(width=width, depth=2, activation=activation, residual=residual)
            )
        if activation_out is not None:
            activation_out = activation
        res_mlp.append(nn.Linear(width, dim_out))
        return nn.Sequential(*res_mlp)

    def _build_sr_network(
        self,
        sr_scale: int,
        n_resblocks: int,
        width: int,
    ):
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
        sr_network_head = nn.Conv2d(3, width, kernel_size=3, padding=1)

        # Build SR body. Several residual blocks.
        sr_network_body = []
        for _ in range(n_resblocks):
            sr_network_body.append(ResBlock(width=width, depth=2, norm=self.sr_norm))
        sr_network_body.append(conv(width))
        sr_network_body = nn.Sequential(*sr_network_body)

        # Build SR tail. Upsampling + one conv layer. Upsampling is (one conv +
        # PixelShuffle) * log2(sr_scale)
        tail = []
        assert self.sr_upsample in ["PixelShuffle", "TransposeConv"], (
            "Incorrect sr_upsample: %s" % self.sr_upsample
        )
        if (sr_scale & (sr_scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(sr_scale, 2))):
                if self.sr_upsample == "PixelShuffle":
                    pass
                elif self.sr_upsample == "TransposeConv":
                    tail.append(transpose_conv(width))  # Upsample by x2.
                    if self.sr_norm == "bn":
                        tail.append(bn(True))
        else:
            raise NotImplementedError("Incorrect sr_scale: %s" % sr_scale)
        tail.append(
            nn.Conv2d(width, 3, kernel_size=3, padding=1)
        )  # Map the feature to RGB.
        sr_network_tail = nn.Sequential(*tail)

        return [sr_network_head, sr_network_body, sr_network_tail]

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the inner forward computation of the model.

        Args:
          x: The tensor that is immediately fed into the NeLF model.

        Returns:
          The output tensor of the NeLF model.
        """
        x = self.head(x)
        x = self.body(x) + x
        return self.tail(x)

    def _sr_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward computation of the SR model.

        Args:
          x: The tensor that is immediately fed into the SR model. [B, H, W, C].

        Returns:
          The output tensor of the SR model. [B, H*sr_scale, W*sr_scale, C].
        """
        x = self.sr_network_head(x)
        x = self.sr_network_body(x) + x
        return self.sr_network_tail(x)

    def forward(
        self,
        input_tensor: torch.Tensor,
        training: bool = False,
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
        rays_o_dim, rays_d_dim = 3, 3
        fid_dim = 1
        # (Next, parse the data, rays_o, rays_d, rays_e, etc. from the input.)

        # FLAME data. rays_o, rays_d, rays_e, fid.
        rays_o, rays_d, rays_e, fid = torch.split(
            input_tensor, [3, 3, self.exp_dim, 1], dim=-1
        )

        # Sample points.
        pts = self.point_sampler(rays_o=rays_o, rays_d=rays_d, training=training)

        # Assemble the model input based on different settings.
        model_input = self.positional_encoder_point(pts)
        if self.use_expressions and rays_e is not None:
            model_input = torch.concat(
                [model_input, self.positional_encoder_expression(rays_e)], dim=-1
            )

        # Use local feature model and spatial attention model.
        if self.expression_scheme in ["local_feats_v1", "local_feats_v2"]:
                # Get spatial attention weights.
                sa_input = self.positional_encoder_point(pts)
                if self.spatial_attention_use_expressions:
                    sa_input = torch.concat(
                        [sa_input, self.positional_encoder_expression(rays_e)], dim=-1
                    )
                attn_weights = self.spatial_attention_model(sa_input)

                # Get local features.
                if self.expression_scheme == "local_feats_v1":
                    if training:
                        local_feats = self.local_feat_model(
                            self.positional_encoder_expression(rays_e)
                        )
                        local_feats = torch.reshape(
                            local_feats,
                            [num_rays, self.n_local_feats, self.local_feat_dim],
                        )
                        # Apply attention weights to local features.
                        # [num_rays,n_local_feats,local_feat_dim]*[num_rays,n_local_feats,1]
                        local_feat = local_feats * attn_weights[..., None]
                        local_feat = torch.math.reduce_sum(local_feat, dim=1)
                    else:
                        local_feats = self.local_feat_model(
                            self.positional_encoder_expression(rays_e[0:1])
                        )
                        local_feats = torch.reshape(
                            local_feats, [self.n_local_feats, self.local_feat_dim]
                        )
                        local_feat = attn_weights @ local_feats

                # Concat local feature into model input.
                model_input = torch.concat([model_input, local_feat], dim=-1)

        # NeLF model forward.
        output = self._inner_forward(model_input)
        
        # Apply SR, if any.
        h, w = self.cam_params[-2:]
        output = torch.reshape(output, [-1, 3, h, w])
        output = self._sr_forward(output)
        return output

    def set_up_model_input(self, data_batch: dict):
        # Read data and reshape. The keys in data indicate different data formats:
        # If C2W_KEY is in keys, the data format is ImageFormat, i.e., each tf
        # example is a single image (along with its c2w, expression, head rotation,
        # etc.). If RAY_DATA_KEY is in keys, the data format is RayFormat,
        # i.e., each tf example is a ray.
        *_, h, w = self.cam_params

        # ImageFormat data. Can be used in training and testing.
        if "c2w" in data_batch:
            c2w = data_batch["c2w"]
            batch_size = data_batch["c2w"].shape[0]  # Num of examples per batch.
            num_images_per_example = 1
            self._data_format = "ImageFormat"  # Each example is an image.
            num_images = batch_size * num_images_per_example  # Num images per batch.

            # Set up rays origin and direction
            c2w = torch.reshape(data_batch[C2W_KEY], [num_images, 3, 4])

            # Set up input tensor based on the dataset.
            if self.dataset == "GNOME":
                pass

            elif self.dataset == "FLAME":
                rays_o, rays_d = self._get_rays_origin_direction_FLAME(
                    c2w, cam_params=self.cam_params
                )
                exp = torch.reshape(data_batch[EXP_KEY], [num_images, 1, self.exp_dim])
                rays_e = torch.broadcast_to(exp, [num_images, h * w, self.exp_dim])
                input_tensor = torch.concat(
                    [
                        rays_o,
                        rays_d,
                        rays_e,
                    ],
                    dim=-1,
                )  # Shape: [B, h * w, ...]

            fid = torch.Tensor([0]).to(device="cuda")  # Dummy frame id.
            fid = torch.reshape(fid, [num_images, 1, 1])
            fid = torch.broadcast_to(fid, [num_images, h * w, 1])
            input_tensor = torch.concat([input_tensor, fid], dim=-1)

            # Reshape to [num_rays, feature_dim_per_ray].
            input_tensor = torch.reshape(input_tensor, [-1, input_tensor.shape[-1]])
            return input_tensor

    def get_model_outputs(
        self,
        data_batch: dict,
        training: bool,
        step: Optional[int] = None,
    ) -> dict:
        """This function handles the inference for the model and provide the results for the loss calculation.

        Args:
          data_batch: A dictionary holding the data batch.
          training: Training flag that is passed into layers.
          step: The current training step.

        Returns:
          A dictionary holding model_outputs with string keys and tensor values.
        """


        # (Next is model forward. Note: the input tensor before being fed into the
        # NeLF model is always in the shape [num_rays, feature_dim_per_ray].)

        # Set up input tensor.
        input_tensor = self.set_up_model_input(data_batch)

        output = self.forward(input_tensor, training=False)
        return output

    def _get_rays_origin_direction(
        self,
        c2w: torch.Tensor,
        cam_params: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        h = torch.cast(h, torch.int32)
        w = torch.cast(w, torch.int32)
        x, y = torch.meshgrid(
            torch.range(w, dtype=torch.float32) + 0.5,
            torch.range(h, dtype=torch.float32) + 0.5,
        )
        x = (x - cx) / fx  # [h, w, 1]
        y = (y - cy) / fy  # [h, w, 1]

        # Get rays origin
        rays_o = c2w[:, :3, 3][:, None, None, :]  # [B, 1, 1, 3]
        rays_o = torch.broadcast_to(rays_o, [batch_size, h, w, 3])  # [B, h, w, 3]

        # Get rays direction
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        rays_d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [h, w, 3]
        rays_d = torch.linalg.normalize(rays_d, dim=-1)[0]  # [h, w, 3]
        rays_d = torch.broadcast_to(rays_d, [batch_size, h, w, 3])  # [B, h, w, 3]
        rays_d = (
            torch.tile(
                rotation[:, None, None, :, :], [1, h, w, 1, 1]  # [B, 1, 1, 3, 3]
            )  # [B, h, w, 3, 3]
            @ rays_d[..., None]  # [B, h, w, 3, 1]
        )  # [B, h, w, 3, 1]
        rays_d = rays_d[..., 0]  # [B, h, w, 3]

        # Reshape to the uniform data layout [B, num_rays, 3]
        rays_o = torch.reshape(rays_o, [batch_size, h * w, 3])
        rays_d = torch.reshape(rays_d, [batch_size, h * w, 3])
        return rays_o, rays_d

    def _get_rays_origin_direction_FLAME(
        self,
        c2w: torch.Tensor,
        cam_params: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        rays_o = torch.broadcast_to(rays_o, [batch_size, h, w, 3])  # [B, h, w, 3]

        # Get rays direction, in batch.
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        rays_d = torch.randn(size=(1, h, w, 3)).to(device="cuda")  # [1, h, w, 3], Dummy rays direction for speed benchmark. The original code uses a base rays direction, which is not available here.
        rays_d = torch.broadcast_to(rays_d, [batch_size, h, w, 3])  # [B, h, w, 3]
        rays_d = (
            torch.tile(
                rotation[:, None, None, :, :], [1, h, w, 1, 1]  # [B, 1, 1, 3, 3]
            )  # [B, h, w, 3, 3]
            @ rays_d[..., None]  # [B, h, w, 3, 1]
        )  # [B, h, w, 3, 1]
        rays_d = rays_d[..., 0]  # [B, h, w, 3]

        # Reshape to the uniform data layout [B, num_rays, 3].
        rays_o = torch.reshape(rays_o, [batch_size, h * w, 3])
        rays_d = torch.reshape(rays_d, [batch_size, h * w, 3])
        return rays_o, rays_d


if __name__ == "__main__":
    # NeLF model configs.
    n_resblocks = 10
    width = 128
    num_freqs_point = 5
    num_freqs_expression = 1
    n_sampled_points = 8

    # SR model configs.
    n_resblocks_sr = 5
    sr_scale = 4
    width_sr = 56
    sr_upsample = "TransposeConv"

    # Local feature model configs.
    n_local_feats = 64
    local_feat_dim = 128

    # Spatial attention model configs.
    spatial_attention_width = 128
    expression_scheme = "local_feats_v1"

    # Other configs.
    use_jaw_eyes_pose = True
    data = "FLAME-0"

    # Test camera params.
    output_height = int(sys.argv[1])
    output_width = int(sys.argv[2])
    cam_params = [250, 250, 128, 128, output_height, output_width]
    cam_params = [x // sr_scale for x in cam_params]

    # Model.
    model = Student(
        # NeLF model configs.
        n_resblocks=n_resblocks,
        width=width,
        num_freqs_point=num_freqs_point,
        num_freqs_expression=num_freqs_expression,
        # SR model configs.
        sr_scale=sr_scale,
        n_resblocks_sr=n_resblocks_sr,
        width_sr=width_sr,
        sr_upsample=sr_upsample,
        # Local feature model configs.
        n_local_feats=n_local_feats,
        local_feat_dim=local_feat_dim,
        # Spatial attention model configs.
        spatial_attention_width=spatial_attention_width,
        expression_scheme=expression_scheme,
        # Other configs.
        use_jaw_eyes_pose=use_jaw_eyes_pose,
        data=data,
        cam_params=cam_params,
    )
    model.cuda()


    num_images = 1
    data_batch = {
        "c2w": torch.randn(
            size=(
                num_images,
                12,
            )
        ).to(device="cuda"),
        "exp": torch.randn(
            size=(
                num_images,
                model.exp_dim,
            )
        ).to(device="cuda"),
    }
    dummy_input = model.set_up_model_input(data_batch)


    # Use torch summary to get the model size.
    summary(model, input_data=(dummy_input, False))
    print('==> Dummy input shape: %s' % list(dummy_input.shape))
    print('==> Model device:', next(model.parameters()).device, 'Input device:', data_batch["c2w"].device)
    print('==> Input resolution: %s x %s (SR scale: %s)' % (output_height // sr_scale, output_width // sr_scale, sr_scale))

    # Speed benchmark, warm up.
    num_frames = 550
    warmup_frames = 50
    num_eval_runs = num_frames - warmup_frames
    for i in range(num_frames):
        if i == warmup_frames:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.no_grad():
            output = model(dummy_input, training=False)
    print('==> Output shape: %s' % list(output.shape))
    print('==> Output resolution: %s x %s' % (output.shape[-2], output.shape[-1]))
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed_time = (t1 - t0) / num_eval_runs
    print(f"==> Rendered {num_eval_runs} frames. Average time taken: {elapsed_time:.3f} seconds per frame. FPS: {1 / elapsed_time:.1f}")



############################
# Model params and FLOPs (Use torchinfo: https://github.com/tyleryep/torchinfo).

# Total params: 2,066,534
# Trainable params: 2,066,534
# Non-trainable params: 0
# Total mult-adds (G): 23.49
# FLOPs per ray: 8.960723876953124e-05 (G)



############################
# Speed benchmark results.
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 896 896
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 768 768
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 640 640
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 512 512
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 384 384
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 256 256
# CUDA_VISIBLE_DEVICES=0 python speed_benchmark_pytorch.py 128 128


# 896x896
# ==> Dummy input shape: [50176, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 224 x 224 (SR scale: 4)
# ==> Output shape: [1, 3, 896, 896]
# ==> Output resolution: 896 x 896
# ==> Rendered 500 frames. Average time taken: 0.016 seconds per frame. FPS: 62.3


# 768x768
# ==> Dummy input shape: [36864, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 192 x 192 (SR scale: 4)
# ==> Output shape: [1, 3, 768, 768]
# ==> Output resolution: 768 x 768
# ==> Rendered 500 frames. Average time taken: 0.012 seconds per frame. FPS: 82.3


# 640x640
# ==> Dummy input shape: [25600, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 160 x 160 (SR scale: 4)
# ==> Output shape: [1, 3, 640, 640]
# ==> Output resolution: 640 x 640
# ==> Rendered 500 frames. Average time taken: 0.008 seconds per frame. FPS: 119.0


# 512x512
# ==> Dummy input shape: [16384, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 128 x 128 (SR scale: 4)
# ==> Output shape: [1, 3, 512, 512]
# ==> Output resolution: 512 x 512
# ==> Rendered 500 frames. Average time taken: 0.005 seconds per frame. FPS: 186.3


# 384x384
# ==> Dummy input shape: [9216, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 96 x 96 (SR scale: 4)
# ==> Output shape: [1, 3, 384, 384]
# ==> Output resolution: 384 x 384
# ==> Rendered 500 frames. Average time taken: 0.003 seconds per frame. FPS: 295.1


# 256x256
# ==> Dummy input shape: [4096, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 64 x 64 (SR scale: 4)
# ==> Output shape: [1, 3, 256, 256]
# ==> Output resolution: 256 x 256
# ==> Rendered 500 frames. Average time taken: 0.003 seconds per frame. FPS: 331.4


# 128x128
# ==> Dummy input shape: [1024, 116]
# ==> Model device: cuda:0 Input device: cuda:0
# ==> Input resolution: 32 x 32 (SR scale: 4)
# ==> Output shape: [1, 3, 128, 128]
# ==> Output resolution: 128 x 128
# ==> Rendered 500 frames. Average time taken: 0.002 seconds per frame. FPS: 408.1


############################
# Speed benchmark results of GaussianAvatars: https://github.com/ShenhanQian/GaussianAvatars/tree/main (git id: faf0b13)
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 896 --width 896
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 768 --width 768
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 640 --width 640
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 512 --width 512
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 384 --width 384
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 256 --width 256
# CUDA_VISIBLE_DEVICES=0 python fps_benchmark_demo.py --height 128 --width 128


# 896x896
# Round 2 [26/10 21:58:23]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 210.86it/s]
# torch.Size([3, 896, 896]) [26/10 21:58:26]
# Rendering 500 images took 2.37 s [26/10 21:58:26]
# FPS: 210.80 [26/10 21:58:26]


# 768x768
# Round 2 [26/10 21:59:21]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 212.07it/s]
# torch.Size([3, 768, 768]) [26/10 21:59:23]
# Rendering 500 images took 2.36 s [26/10 21:59:23]
# FPS: 212.01 [26/10 21:59:23]


# 640x640
# Round 2 [26/10 21:59:53]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 215.87it/s]
# torch.Size([3, 640, 640]) [26/10 21:59:55]
# Rendering 500 images took 2.32 s [26/10 21:59:55]
# FPS: 215.81 [26/10 21:59:55]


# 512x512
# Round 2 [26/10 22:00:25]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 217.11it/s]
# torch.Size([3, 512, 512]) [26/10 22:00:28]
# Rendering 500 images took 2.30 s [26/10 22:00:28]
# FPS: 217.05 [26/10 22:00:28]


# 384x384
# Round 2 [26/10 22:00:58]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 211.34it/s]
# torch.Size([3, 384, 384]) [26/10 22:01:00]
# Rendering 500 images took 2.37 s [26/10 22:01:00]
# FPS: 211.28 [26/10 22:01:00]


# 256x256
# Round 2 [26/10 22:02:46]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 201.02it/s]
# torch.Size([3, 256, 256]) [26/10 22:02:48]
# Rendering 500 images took 2.49 s [26/10 22:02:48]
# FPS: 200.95 [26/10 22:02:48]


# 128x128
# Round 2 [26/10 22:02:07]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 189.15it/s]
# torch.Size([3, 128, 128]) [26/10 22:02:10]
# Rendering 500 images took 2.64 s [26/10 22:02:10]
# FPS: 189.06 [26/10 22:02:10]
