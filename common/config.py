"""
POGS configuration.

Every default is taken from the official POGS release so that ``v1`` reproduces
the paper's setup:

* ``pogs/pogs_config.py``                       (trainer + optimiser schedules)
* ``pogs/pogs.py::POGSModelConfig``             (model; inherits nerfstudio's
                                                 ``SplatfactoModelConfig``)
* ``pogs/data/full_images_datamanager.py``      (dino/lerf start steps, masks)
* ``pogs/tracking/rigid_group_optimizer.py``    (``RigidGroupOptimizerConfig``)
* ``pogs/tracking/optim.py``                    (``MATCH_RESOLUTION``)

Where we deliberately differ from the released code, the field's docstring says
so and why. Reference: Yu, Hari, El-Refai et al., "Persistent Object Gaussian
Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped
Objects", ICRA 2025.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, Tuple

# --------------------------------------------------------------------------- #
# Model (POGSModelConfig <- SplatfactoModelConfig)
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    gaussian_dim: int = 64
    """'we apply PCA to reduce their dimensionality from several hundred to
    d = 64 dimensions' (Sec. IV-B)."""
    dim: int = 64
    """Output dimension of the rendered DINO feature map (``POGSModel.nn``)."""
    dino_init_std: float = 1.0
    """Official: ``torch.randn((num_points, gaussian_dim))``."""
    instance_n_dims: int = 128
    clip_n_dims: int = 512
    log2_hashmap_size: int = 19
    """Both tcnn HashGrids use 2^19 in the official field. Lower it (e.g. 16)
    only on a memory-constrained host."""
    dino_rescale_factor: int = 5
    min_mask_screensize: float = 0.003
    clip_loss_weight: float = 0.5
    depth_loss_mult: float = 0.05
    depth_loss_type: Literal["pearson", "mse", "none"] = "pearson"
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 3.0
    ssim_lambda: float = 0.2
    dino_nn_k: int = 3
    """``cuml NearestNeighbors(n_neighbors=3)`` -- three neighbours *including*
    the point itself."""

    background_color: Tuple[float, float, float] = (0.1490, 0.1647, 0.2157)
    """Background used at eval / tracking time (splatfacto's 'random' preset
    falls back to this colour outside training)."""
    random_background: bool = True
    """Splatfacto ``background_color='random'``: a random colour per training
    step, which stops the splat from explaining the background with
    semi-transparent Gaussians. POGS does not override it."""
    rasterize_mode: Literal["antialiased", "classic"] = "classic"
    """Splatfacto default; POGS does not override it."""

    # densification / pruning -- splatfacto defaults, POGS overrides split_screen_size
    warmup_length: int = 500
    refine_every: int = 100
    reset_alpha_every: int = 30
    stop_split_at: int = 15000
    stop_screen_size_at: int = 4000
    densify_grad_thresh: float = 0.0008
    use_absgrad: bool = True
    densify_size_thresh: float = 0.01
    split_screen_size: float = 0.025
    """POGS override (splatfacto: 0.05)."""
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    cull_screen_size: float = 0.15
    n_split_samples: int = 2
    max_gaussians: int = 1_000_000
    """Only enforced by the pure-torch fallback densifier (gsplat's strategy has
    no cap, like the official code)."""


# --------------------------------------------------------------------------- #
# Training (TrainerConfig in pogs_config.py + FullImageDatamanagerConfig)
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    max_num_iterations: int = 4000
    """README: 'train the POGS for 4000 steps'."""
    dino_step: int = 100
    """DINO supervision starts after this step."""
    lerf_step: int = 2500
    """CLIP + grouping supervision start after this step."""
    dino_nn_delay: int = 1000
    """dino_nn_loss starts at ``dino_step + 1000``."""

    # Learning rates -- verbatim from pogs_config.py
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
    lr_nn_projection_final: float = 1e-3

    grad_accum_lerf: int = 3
    """``gradient_accumulation_steps = {'camera_opt': 100, 'color': 10,
    'shs': 10, 'lerf': 3}``. nerfstudio matches these keys against parameter
    *group names*; POGS's groups are called ``features_dc`` / ``features_rest``,
    so ``color`` and ``shs`` match nothing and are no-ops in the release. Only
    ``lerf`` is effective (camera_opt is unused: synthetic poses are exact)."""

    clip_downscale_factor: int = 2
    num_clip_scales: int = 10
    clip_scale_min: float = 0.05
    clip_scale_max: float = 0.5
    patch_stride_scaler: float = 0.5
    num_random_masks: int = 15
    """Masks sampled per step for the contrastive group loss."""
    mask_max_area_frac: float = 1.0 / 3.5
    """Detic stand-in: the official loader drops masks larger than H*W/3.5."""
    mask_background_ids: Tuple[int, ...] = (0,)
    """Instance ids a detector would not report as objects (floor, support
    table). They join the negative mask, as in the paper ('a negative mask
    wherever an object mask does not exist')."""

    log_every: int = 100
    seed: int = 0

    @property
    def max_steps(self) -> int:
        return self.max_num_iterations

    @max_steps.setter
    def max_steps(self, val: int):
        self.max_num_iterations = val


TrainingConfig = TrainConfig


# --------------------------------------------------------------------------- #
# Grouping / clustering (Sec. IV-B, "Before the tracking phase begins...")
# --------------------------------------------------------------------------- #


@dataclass
class GroupingConfig:
    cluster_eps: float = 0.012
    """``ViewerSlider("Cluster Eps", 0.012, ...)`` default."""
    min_samples: int = 50
    min_cluster_size: int = 300
    voxel_size: float = 0.0001
    allow_single_cluster: bool = False
    knn_k: int = 3
    """Feature smoothing over 3 nearest neighbours + self (4 total)."""
    max_cluster_points: int = 25000
    """CPU-HDBSCAN budget (the release uses cuML on the GPU); see
    ``grouping.cluster_scene``."""
    select_method: Literal["geometry", "language"] = "geometry"
    """How tracked clusters are chosen. The paper uses an interactive click /
    language query; ``geometry`` drops the floor and support surfaces (the
    official code removes everything below ``table_z + 0.015``)."""
    support_z: float = 0.015
    """Gaussians below this height (metres above the floor) are treated as the
    support surface and dropped from tracked clusters."""
    max_footprint_m2: float = 0.06
    """Robust (5-95th percentile) XY extent above which a cluster is a support
    surface. A min/max extent is not robust: a few hundred stray floor
    Gaussians in an object's cluster inflated it past 0.4 m^2 (scene v2)."""
    max_share: float = 0.40
    background_sim_thresh: float = 0.98
    """Cosine similarity to the negative-mask target (all-ones unit vector)
    above which a cluster is background. The negative-mask term drives true
    background to ~1.0 (0.999 measured); 0.8 also rejected the mug (0.94 on
    scene v2), whose features are distinct but happen to lie nearer."""
    min_object_share: float = 0.01
    """Clusters holding less than this share of the remaining Gaussians are
    fragments, not objects."""


# --------------------------------------------------------------------------- #
# Tracking (RigidGroupOptimizerConfig)
# --------------------------------------------------------------------------- #


@dataclass
class TrackingConfig:
    use_depth: bool = True
    """Paper Sec. IV-D: 'The total loss is a weighted sum of the feature loss
    and depth loss'. NOTE: the released ``step()`` defaults to
    ``use_depth=False`` (depth only during initialisation); run the
    "No Depth" variant to reproduce the release behaviour."""
    use_dino: bool = True
    use_rgb: bool = False
    depth_loss_mult: float = 3.7
    depth_ignore_threshold: float = 0.26
    """metres; squared internally"""
    pose_lr: float = 0.004
    pose_lr_final: float = 0.0008
    blur_kernel_size: int = 5
    clip_grad: float = 0.0
    """The release defines ``clip_grad = 0.8`` but never applies it. 0 = off."""
    use_roi: bool = True
    use_mask_loss: bool = False
    mask_loss_mult: float = 0.6
    roi_inflate_proportion: float = 0.25
    roi_inflate: float = 75.0
    max_roi_resolution: int = 490
    match_resolution: int = 500
    """``Optimizer.MATCH_RESOLUTION``: the tracking camera's short side."""
    dino_alpha_thresh: float = 0.8
    """Rendered DINO is zeroed where alpha < 0.8 (``pogs.py``, eval mode)."""
    render_mask_thresh: float = 0.9
    """``render_mask``: object mask = accumulation > 0.9."""
    render_untracked: bool = False
    """The release replaces the model's Gaussians with the selected objects
    only (``keep_inds``) before tracking, so floor/table are never rendered."""
    upright_constraint: bool = False
    """Not in POGS. Projects pose updates onto yaw-only; off by default."""

    init_n_seeds: int = 1
    """The release seeds ``initialize_obj_pose`` from identity only."""
    init_niter_coarse: int = 100
    """First pass: DINO-only, full frame."""
    init_niter_fine: int = 70
    """Second pass: + depth + ROI."""
    init_lr: float = 0.005
    init_use_depth: Optional[bool] = None
    """Depth in the fine init pass. ``None`` follows ``use_depth``. The release
    always uses depth here (``try_opt(..., use_depth=True)``), even though its
    per-frame ``step()`` does not."""
    track_niter: int = 25
    """``track_main_online_demo.py``: ``n_opt_iters = 25``."""
    accumulation_thresh: float = 0.85
    depth_outlier_ratio: float = 1.3
    erode_depth_kernel: int = 5
    erode_valid_kernel: int = 9
    depth_mask_relative_fallback: bool = False
    """Not in the release. When True, an object whose ``accumulation > 0.85``
    mask is empty falls back to a relative threshold (counted in
    ``RigidGroupOptimizer.n_depth_fallback``)."""

    @property
    def n_iters(self) -> int:
        return self.track_niter

    @n_iters.setter
    def n_iters(self, val: int):
        self.track_niter = val


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    raster_backend: Literal["auto", "gsplat", "torch"] = "auto"
    torch_raster_tile: int = 16
    torch_raster_max_per_tile: int = 1024
    torch_raster_tile_chunk: int = 128
    dino_max_size: int = 1260
    """``MAX_DINO_SIZE``: long side the image is resampled to before DINOv2."""
    output_dir: str = "outputs"


@dataclass
class EncoderConfig:
    dino_model: str = "dinov2_vits14"
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "laion2b_s34b_b88k"


@dataclass
class POGSConfig:
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

    def to_dict(self) -> dict:
        return asdict(self)

    def scale_for_smoke_test(self) -> "POGSConfig":
        """Tiny settings so the whole pipeline can be exercised quickly."""
        self.train.max_num_iterations = 60
        self.train.lerf_step = 20
        self.train.dino_step = 2
        self.train.dino_nn_delay = 20
        self.train.num_clip_scales = 3
        self.train.num_random_masks = 4
        self.model.warmup_length = 20
        self.model.refine_every = 20
        self.model.log2_hashmap_size = 16
        self.grouping.min_samples = 5
        self.grouping.min_cluster_size = 20
        self.tracking.init_niter_coarse = 4
        self.tracking.init_niter_fine = 3
        self.tracking.track_niter = 3
        self.runtime.dino_max_size = 210
        return self
