"""
POGS configuration.

Every default here is taken from the official POGS release
(``pogs/pogs_config.py``, ``pogs/pogs.py::POGSModelConfig``,
``pogs/tracking/rigid_group_optimizer.py::RigidGroupOptimizerConfig``)
so that the Kaggle re-implementation reproduces the paper's setup.

Reference: Yu, Hari, El-Refai, Dalil, Kerr, Kim, Cheng, Irshad, Goldberg,
"Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot
Manipulation of Irregularly Shaped Objects", ICRA 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# --------------------------------------------------------------------------- #
# Scene capture (Sec. IV-A)
# --------------------------------------------------------------------------- #


@dataclass
class CaptureConfig:
    """Multi-view hemisphere capture. The paper captures 35 views with a
    wrist-mounted ZED Mini as the UR5 sweeps a predefined trajectory."""

    num_views: int = 35
    """Paper: 'We capture images from 35 views'."""
    image_height: int = 360
    image_width: int = 640
    fov_x_deg: float = 62.0
    """Approximately the ZED Mini HD720 horizontal FoV."""
    radius_min: float = 0.55
    radius_max: float = 0.85
    elevation_min_deg: float = 22.0
    elevation_max_deg: float = 68.0
    azimuth_turns: float = 2.5
    """Number of times the hemisphere spiral wraps around the workspace."""
    depth_noise_std: float = 0.0035
    """Simulates RAFT-Stereo disparity error (metres, at 0.7 m range)."""
    depth_dropout: float = 0.01
    """Fraction of pixels with invalid stereo depth."""
    dbscan_eps: float = 0.012
    """Paper: 'We run DBSCAN on the fused pointcloud to filter noise and
    floater points.'"""
    dbscan_min_samples: int = 12
    max_seed_points: int = 60_000
    seed: int = 0

    @property
    def width(self) -> int:
        return self.image_width

    @width.setter
    def width(self, val: int):
        self.image_width = val

    @property
    def height(self) -> int:
        return self.image_height

    @height.setter
    def height(self, val: int):
        self.image_height = val


# --------------------------------------------------------------------------- #
# Model (POGSModelConfig)
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    gaussian_dim: int = 64
    """'we apply PCA to reduce their dimensionality from several hundred to
    d = 64 dimensions' (Sec. IV-B, Self-Supervised Features)."""
    dim: int = 64
    """Output dimension of the rendered DINO feature map."""
    instance_n_dims: int = 128
    """GaussianField.instance_net output width in the official code."""
    clip_n_dims: int = 512
    """OpenCLIP ViT-B-16 embedding width."""
    log2_hashmap_size: int = 19
    """``GaussianField``'s two tcnn HashGrid encodings both use 2^19 in the
    official config.  With tiny-cuda-nn's per-level cap that is ~140 MB of
    parameters, which Adam turns into ~560 MB of optimiser state and gradient.
    Lower it (16 -> ~9 MB) on a memory-constrained host; it is the first knob to
    reach for if training dies with an allocation failure."""
    dino_rescale_factor: int = 5
    min_mask_screensize: float = 0.003
    clip_loss_weight: float = 0.5
    depth_loss_mult: float = 0.05
    depth_loss_type: Literal["pearson", "mse", "none"] = "pearson"
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 3.0
    ssim_lambda: float = 0.2
    background_color: Tuple[float, float, float] = (0.1412, 0.1412, 0.1412)
    rasterize_mode: Literal["antialiased", "classic"] = "antialiased"

    # densification / pruning (splatfacto defaults used by POGS)
    warmup_length: int = 500
    refine_every: int = 100
    reset_alpha_every: int = 30
    stop_split_at: int = 2500
    stop_screen_size_at: int = 2000
    densify_grad_thresh: float = 0.0008
    densify_size_thresh: float = 0.01
    split_screen_size: float = 0.025
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    n_split_samples: int = 2
    max_gaussians: int = 300_000


# --------------------------------------------------------------------------- #
# Training (TrainerConfig in pogs_config.py)
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    max_num_iterations: int = 4000
    """Paper / README: 'Script used to train the POGS for 4000 steps'."""
    dino_step: int = 100
    """FullImageDatamanagerConfig.dino_step -- when DINO supervision starts."""
    lerf_step: int = 2500
    """FullImageDatamanagerConfig.lerf_step -- when CLIP + grouping start."""
    dino_nn_delay: int = 1000
    """dino_nn_loss starts at ``dino_step + 1000`` in the official code."""

    # Learning rates -- verbatim from pogs_config.py optimizers block.
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_features_dc: float = 0.0025
    lr_features_rest: float = 0.0025 / 20
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_scales_final: float = 1e-3
    lr_quats: float = 0.001
    lr_lerf: float = 2.5e-3
    lr_lerf_final: float = 1e-3
    lr_dino_feats: float = 1e-2
    lr_dino_feats_final: float = 1e-3
    lr_nn_projection: float = 1e-2

    grad_accum_color: int = 10
    grad_accum_shs: int = 10
    grad_accum_lerf: int = 3

    clip_downscale_factor: int = 2
    """FullImageDatamanagerConfig.clip_downscale_factor"""
    num_clip_scales: int = 10
    """patch_tile_size_res"""
    clip_scale_min: float = 0.05
    clip_scale_max: float = 0.5
    """patch_tile_size_range"""
    patch_stride_scaler: float = 0.5
    num_random_masks: int = 15
    """Number of random masks sampled per step for the contrastive group loss."""

    render_height: int = 360
    render_width: int = 640
    log_every: int = 100
    seed: int = 0

    @property
    def max_steps(self) -> int:
        return self.max_num_iterations

    @max_steps.setter
    def max_steps(self, val: int):
        self.max_num_iterations = val

    @property
    def refine_every(self) -> int:
        return getattr(self, "_refine_every", 100)

    @refine_every.setter
    def refine_every(self, val: int):
        self._refine_every = val

TrainingConfig = TrainConfig


# --------------------------------------------------------------------------- #
# Grouping / clustering (Sec. IV-B, "Before the tracking phase begins...")
# --------------------------------------------------------------------------- #


@dataclass
class GroupingConfig:
    cluster_eps: float = 0.012
    """ViewerSlider("Cluster Eps", 0.012, ...) default in the official code."""
    min_samples: int = 50
    min_cluster_size: int = 300
    voxel_size: float = 0.0001
    allow_single_cluster: bool = False
    knn_k: int = 3
    """Feature smoothing over 3 nearest neighbours + self (4 total)."""


# --------------------------------------------------------------------------- #
# Tracking (RigidGroupOptimizerConfig)
# --------------------------------------------------------------------------- #


@dataclass
class TrackingConfig:
    use_depth: bool = True
    use_dino: bool = True
    use_rgb: bool = False
    depth_loss_mult: float = 3.7
    depth_ignore_threshold: float = 0.26
    """metres; squared internally"""
    pose_lr: float = 0.004
    pose_lr_final: float = 0.0008
    rot_lr_scaler: float = 3.0
    blur_kernel_size: int = 5
    blur2_kernel_size: int = 45
    clip_grad: float = 0.8
    use_roi: bool = True
    use_mask_loss: bool = False
    mask_loss_mult: float = 0.6
    roi_inflate_proportion: float = 0.25
    roi_inflate: float = 75.0
    max_roi_resolution: int = 490
    match_resolution: int = 500
    """Optimizer.MATCH_RESOLUTION in the official code."""
    upright_constraint: bool = False
    """When True, projects pose updates onto the upright yaw axis (qx = qy = 0) for planar/tabletop manipulation."""

    init_n_seeds: int = 1
    """Yaw seeds tried by ``initialize_obj_pose``.  1 == the official code, which
    seeds from identity only -- correct, because Sec. IV-C canonicalises each
    object frame so its scan-time pose is identity.  Values > 1 run a greedy
    *per-object* sweep; never a single yaw shared by every object, which scores
    one object's gain against every other object's corruption."""
    init_niter_coarse: int = 100
    """First pass: DINO-only, full frame (official ``initialize_obj_pose``)."""
    init_niter_fine: int = 70
    """Second pass: + depth + ROI."""
    init_lr: float = 0.005
    track_niter: int = 25
    """Paper Sec. VI: multiple optimisation steps per frame at ~5 Hz."""
    accumulation_thresh: float = 0.85
    depth_outlier_ratio: float = 1.3
    erode_depth_kernel: int = 5
    erode_valid_kernel: int = 9
    depth_mask_relative_fallback: bool = True
    """The official mask is the absolute ``accumulation > 0.85``.  On a thin or
    under-trained splat no pixel ever reaches it, and the Sec. IV-D depth term
    then vanishes *silently* -- which makes the paper's "No Depth" ablation
    compare a run against itself.  With this on, an object whose absolute mask is
    empty falls back to ``accumulation > accumulation_thresh * accumulation.max()``
    for that render only.  Every engagement is counted in
    ``RigidGroupOptimizer.n_depth_fallback`` and reported, so it is never quiet.
    Set to ``False`` to reproduce the released behaviour exactly."""

    @property
    def n_iters(self) -> int:
        return self.track_niter

    @n_iters.setter
    def n_iters(self, val: int):
        self.track_niter = val

    @property
    def init_iters(self) -> int:
        return self.init_niter_fine

    @init_iters.setter
    def init_iters(self, val: int):
        self.init_niter_fine = val

@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    feature_device: str = "cuda:1"
    """Second T4. DINOv2 / CLIP run here so they overlap with the pose
    optimisation on cuda:0 (paper Sec. VI lists this as future work)."""
    use_second_gpu: bool = True
    raster_backend: Literal["auto", "gsplat", "torch"] = "auto"
    torch_raster_tile: int = 16
    torch_raster_max_per_tile: int = 1024
    torch_raster_tile_chunk: int = 128
    dino_max_size: int = 1260
    """Long side the image is resampled to before DINOv2. The official code
    uses MAX_DINO_SIZE = 1260. The kaggle preset halves this to 644."""
    amp: bool = False
    output_dir: str = "/kaggle/working/pogs_out"


@dataclass
class EncoderConfig:
    dino_model: str = "dinov2_vits14"
    clip_model: str = "ViT-B-16"


@dataclass
class POGSConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    encoders: EncoderConfig = field(default_factory=EncoderConfig)
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is not None:
            self.runtime.device = self.device

    @property
    def training(self) -> TrainConfig:
        return self.train

    @training.setter
    def training(self, val: TrainConfig):
        self.train = val

    def scale_for_smoke_test(self) -> "POGSConfig":
        """Tiny settings so the whole pipeline can be exercised in ~2 minutes."""
        self.capture.num_views = 8
        self.capture.image_height = 120
        self.capture.image_width = 160
        self.capture.max_seed_points = 4000
        self.train.max_num_iterations = 60
        self.train.lerf_step = 20
        self.train.dino_step = 2
        self.train.dino_nn_delay = 20
        self.train.render_height = 120
        self.train.render_width = 160
        self.train.num_clip_scales = 3
        self.train.num_random_masks = 4
        self.model.warmup_length = 20
        self.model.refine_every = 20
        self.model.stop_split_at = 40
        self.model.max_gaussians = 20_000
        self.model.log2_hashmap_size = 16
        self.grouping.min_samples = 5
        self.grouping.min_cluster_size = 20
        self.grouping.voxel_size = 0.004
        self.tracking.init_niter_coarse = 4
        self.tracking.init_niter_fine = 3
        self.tracking.track_niter = 3
        self.tracking.match_resolution = 120
        self.tracking.max_roi_resolution = 84
        self.runtime.use_second_gpu = False
        self.runtime.dino_max_size = 210
        return self