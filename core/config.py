"""flight2world.core.config

Single source of truth for the V10 reference configuration.

Every value here is reproduced EXACTLY from `fusion_v10.py` (the
immutable scientific baseline). These are the CURRENT V10 reference
camera parameters. The reconstruction happens in COLMAP's *arbitrary*
coordinate system: the metric scale is UNKNOWN, and these values must
NOT be interpreted as, or silently converted into, metric units.

The depth model and the fusion/cleanup thresholds are also recorded so
the full pipeline can be reproduced deterministically later. The
lowest-level modules (colmap_io, calibration) depend only on the camera
parameters and the calibration gates defined here.
"""

import os
from dataclasses import dataclass, field


# ============================================================
# CAMERA INTRINSICS (from COLMAP cameras.txt, current V10)
# ============================================================
#
# Reference line in test/sparse_txt/cameras.txt:
#   1 SIMPLE_RADIAL 1280 720 1167.4277386087481 640 360 -0.05865183452009938
#
# Hardcoded identically in fusion_v10.py lines 71-76.
# SIMPLE_RADIAL params are [f, cx, cy, k] -> fx == fy == f. There is a
# single radial distortion coefficient k that V10 does not apply in its
# own projection math (it uses the pinhole model with FX/FY/CX/CY), so we
# preserve EXACTLY the same pinhole convention V10 uses.
#
# IMPORTANT: metric scale is unknown. These focal lengths are in pixel
# units, and all world coordinates are in COLMAP scene units. Do not
# convert them to metres.
# ============================================================

FX = 1167.4277386087481
FY = 1167.4277386087481
CX = 640.0
CY = 360.0
IMAGE_W = 1280
IMAGE_H = 720

# The single radial-distortion coefficient recorded in cameras.txt. Kept
# for faithful reproduction of the source data; V10's own projection path
# does not use it.
CAMERA_DISTORTION_K = -0.05865183452009938  # SIMPLE_RADIAL k (unused by V10)

# Metric scale is not recoverable from the available artifacts.
METRIC_SCALE_UNKNOWN = True

# Default path the rest of the pipeline reads COLMAP output from.
# (mirrors fusion_v10.py lines 55-57)
SPARSE_TXT = os.path.expanduser("~/flight2world/test/sparse_txt")


# ============================================================
# CALIBRATION (robust per-frame depth calibration, V10)
# ============================================================
#
# fusion_v10.py lines 118-121, 234-283.
# ============================================================

# Minimum |correlation| for a per-frame calibration to be accepted.
CALIB_MIN_CORR = 0.55

# Minimum number of (depth, COLMAP-Z) pairs before a frame is calibrated.
CALIB_MIN_SAMPLES = 100

# Frames whose calibration RMSE exceeds this multiple of the median RMSE
# are dropped as outliers (applied across all frames, in the pipeline).
CALIB_RMSE_OUTLIER_FACTOR = 2.5

# robust_fit lower bounds (fusion_v10.py line 236): the internal fit
# refuses to operate on fewer than this many valid paired points.
ROBUST_FIT_MIN_SAMPLES = 50

# robust_fit default IRLS iteration count (fusion_v10.py line 234).
ROBUST_FIT_ITERATIONS = 5

# robust_fit MAD trimming (fusion_v10.py lines 265-268): 3 * 1.4826 * MAD.
ROBUST_FIT_MAD_SIGMAS = 3.0
ROBUST_FIT_MAD_SCALING = 1.4826

# Inverse-depth handling (fusion_v10.py lines 554, 658): predicted depth
# is treated as inverse depth, guarded with a small positive epsilon.
INVERSE_DEPTH_EPS = 1e-6


# ============================================================
# VIDEO / FRAME EXTRACTION (Phase 4)
# ============================================================
# Target extraction — defaults preserve the benchmark (1280×720 frames
# extracted at 1 fps from the original drone video; resize uses cv2
# INTER_AREA for downscale).
VIDEO_DEFAULT_FPS: float = 1.0
VIDEO_TARGET_WIDTH: int = IMAGE_W
VIDEO_TARGET_HEIGHT: int = IMAGE_H
# Deterministic frame naming: frame_0001.jpg, frame_0002.jpg, ...
VIDEO_FRAME_PATTERN: str = "frame_{idx:04d}.jpg"
VIDEO_JPEG_QUALITY: int = 95

# ============================================================
# COLMAP (Phase 4 — validated benchmark defaults)
# ============================================================
# The COLMAP executable is configurable; default "colmap" relies on PATH
# (Phase 3 reported colmap 4.1.1 on PATH).
COLMAP_EXE: str = "colmap"
COLMAP_CAMERA_MODEL: str = "SIMPLE_RADIAL"
COLMAP_SINGLE_CAMERA: int = 1
COLMAP_SEQUENTIAL_OVERLAP: int = 10
# Feature extraction / matching — validated defaults (CPU; GPU disabled
# on the M3/8 GB reference environment).
COLMAP_USE_GPU: int = 0
# Sparse mapper options (mirrors the project.ini used for the benchmark):
COLMAP_MAPPER_MIN_NUM_MATCHES: int = 15
COLMAP_MAPPER_BA_REFINE_FOCAL_LENGTH: int = 1
COLMAP_MAPPER_BA_REFINE_PRINCIPAL_POINT: int = 0
COLMAP_MAPPER_BA_REFINE_EXTRA_PARAMS: int = 1

# ============================================================
# PIPELINE / OUTPUT (Phase 4 + Phase 5A)
# ============================================================
# Output layout under a run root (filesystem caching):
#   {output_root}/frames/        extracted JPG frames
#   {output_root}/colmap/        COLMAP database + sparse model
#   {output_root}/reconstruction/ fused_raw/clean + confidence (Phase 5A)
#   {output_root}/diagnostics/   per-stage JSON
#   {output_root}/metadata.json  run metadata (coordinate system etc.)
PIPELINE_FRAMES_SUBDIR: str = "frames"
PIPELINE_COLMAP_SUBDIR: str = "colmap"
PIPELINE_DATABASE_NAME: str = "database.db"
PIPELINE_SPARSE_SUBDIR: str = "sparse"
PIPELINE_RECONSTRUCTION_SUBDIR: str = "reconstruction"
PIPELINE_FUSED_RAW_NAME: str = "fused_raw.ply"
PIPELINE_FUSED_CLEAN_NAME: str = "fused_clean.ply"
PIPELINE_CONFIDENCE_PLY: str = "confidence.ply"
PIPELINE_CONFIDENCE_JSON: str = "confidence.json"
PIPELINE_DIAGNOSTICS_SUBDIR: str = "diagnostics"
PIPELINE_METADATA_NAME: str = "metadata.json"
# Cache manifest file recording the extraction / COLMAP config that
# produced the cached outputs (for invalidation).
PIPELINE_CACHE_MANIFEST: str = ".cache_manifest.json"

# ============================================================
# METADATA / COORDINATE SYSTEM (Phase 4)
# ============================================================
# The benchmark reconstruction has no GPS/RTK/GCP source; all geometry
# is in COLMAP's arbitrary coordinate system.
COORDINATE_SYSTEM: str = "COLMAP_relative"
GEOREFERENCED: bool = False
# metric_scale is unknown (null in JSON); keep sentinel None in Python.
METRIC_SCALE = None

# ============================================================
# DEPTH MODEL (V10)
# ============================================================
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_DEPTH_DEVICE = "mps"


# ============================================================
# FUSION (track-anchored, V10). Recorded for downstream phases.
# ============================================================
PIXEL_STRIDE = 8
NEIGHBOR_WINDOW = 3
MIN_NET_VOTES = 2
RELATIVE_TOLERANCE = 0.18
ABSOLUTE_TOLERANCE = 0.05
TRACK_RADIUS_PX = 50

GATE_FACTOR = 2.0
GATE_RADIUS_MIN = 0.015
GATE_RADIUS_MAX = 0.12

MIN_BASELINE_FRAC = 0.08

# Final cleanup (V10).
VOXEL_SIZE = 0.04
STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 1.5


# ============================================================
# TRACK MAPS (build_track_maps, V10)
# ============================================================
# Cap on iterative dilation passes (fusion_v10.py line 355).
TRACK_MAP_MAX_STEPS = 100
# The 8-neighbour dilation kernel is 3x3 (fusion_v10.py line 351).
TRACK_MAP_KERNEL = 3


# ============================================================
# PASS-2 FUSION (track-anchored, V10)
# ============================================================
# Candidate calibrated-Z range (fusion_v10.py line 687).
FUSE_Z_MIN = 0.15
FUSE_Z_MAX = 50.0

# Voting: neighbour-camera Z must exceed this (fusion_v10.py line 757).
FUSE_VOTE_Z_MIN = 0.15
# Division guard in neighbour reprojection (fusion_v10.py lines 761-762).
FUSE_REPROJECT_EPS = 1e-8

# Adaptive-gate local density: 8th-nearest-neighbour of each sparse point
# (fusion_v10.py lines 476-484) and the fallback radius when a sparse
# point has fewer than 2 neighbours.
GATE_KNN_K = 8
GATE_LOCAL_R_DEFAULT = 0.02


# ============================================================
# Structured configuration objects
# ============================================================


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in the exact convention V10 uses.

    V10 assumes a simple pinhole with Fx == Fy == FX (single focal len)
    even though the COLMAP camera model is SIMPLE_RADIAL. The radial
    distortion term is deliberately NOT applied here, matching V10.
    """

    fx: float = FX
    fy: float = FY
    cx: float = CX
    cy: float = CY
    width: int = IMAGE_W
    height: int = IMAGE_H


@dataclass(frozen=True)
class VideoConfig:
    """Video / frame-extraction configuration (Phase 4)."""

    target_fps: float = VIDEO_DEFAULT_FPS
    target_width: int = VIDEO_TARGET_WIDTH
    target_height: int = VIDEO_TARGET_HEIGHT
    frame_pattern: str = VIDEO_FRAME_PATTERN
    jpeg_quality: int = VIDEO_JPEG_QUALITY


@dataclass(frozen=True)
class ColmapConfig:
    """COLMAP runner configuration (Phase 4 — validated benchmark defaults)."""

    exe: str = COLMAP_EXE
    camera_model: str = COLMAP_CAMERA_MODEL
    single_camera: int = COLMAP_SINGLE_CAMERA
    sequential_overlap: int = COLMAP_SEQUENTIAL_OVERLAP
    use_gpu: int = COLMAP_USE_GPU
    min_num_matches: int = COLMAP_MAPPER_MIN_NUM_MATCHES
    ba_refine_focal_length: int = COLMAP_MAPPER_BA_REFINE_FOCAL_LENGTH
    ba_refine_principal_point: int = COLMAP_MAPPER_BA_REFINE_PRINCIPAL_POINT
    ba_refine_extra_params: int = COLMAP_MAPPER_BA_REFINE_EXTRA_PARAMS
    database_name: str = PIPELINE_DATABASE_NAME
    sparse_subdir: str = PIPELINE_SPARSE_SUBDIR


@dataclass(frozen=True)
class PipelineConfig:
    """Complete V10 reference pipeline configuration.

    Defaults reproduce fusion_v10.py exactly. Metric scale is unknown.
    """

    # --- camera ---
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)

    # --- calibration gates ---
    calib_min_corr: float = CALIB_MIN_CORR
    calib_min_samples: int = CALIB_MIN_SAMPLES
    calib_rmse_outlier_factor: float = CALIB_RMSE_OUTLIER_FACTOR
    robust_fit_min_samples: int = ROBUST_FIT_MIN_SAMPLES
    robust_fit_iterations: int = ROBUST_FIT_ITERATIONS
    robust_fit_mad_sigmas: float = ROBUST_FIT_MAD_SIGMAS
    robust_fit_mad_scaling: float = ROBUST_FIT_MAD_SCALING
    inverse_depth_eps: float = INVERSE_DEPTH_EPS

    # --- depth model ---
    depth_model: str = DEFAULT_DEPTH_MODEL
    depth_device: str = DEFAULT_DEPTH_DEVICE

    # --- fusion (recorded; consumed in Phase 3+) ---
    pixel_stride: int = PIXEL_STRIDE
    neighbor_window: int = NEIGHBOR_WINDOW
    min_net_votes: int = MIN_NET_VOTES
    relative_tolerance: float = RELATIVE_TOLERANCE
    absolute_tolerance: float = ABSOLUTE_TOLERANCE
    track_radius_px: int = TRACK_RADIUS_PX
    gate_factor: float = GATE_FACTOR
    gate_radius_min: float = GATE_RADIUS_MIN
    gate_radius_max: float = GATE_RADIUS_MAX
    min_baseline_frac: float = MIN_BASELINE_FRAC

    # --- pass-2 fusion depth gates (V10) ---
    fuse_z_min: float = FUSE_Z_MIN
    fuse_z_max: float = FUSE_Z_MAX
    fuse_vote_z_min: float = FUSE_VOTE_Z_MIN
    fuse_reproject_eps: float = FUSE_REPROJECT_EPS

    # --- track maps (V10) ---
    track_map_max_steps: int = TRACK_MAP_MAX_STEPS
    track_map_kernel: int = TRACK_MAP_KERNEL

    # --- adaptive gate density (V10) ---
    gate_knn_k: int = GATE_KNN_K
    gate_local_r_default: float = GATE_LOCAL_R_DEFAULT

    # --- cleanup (recorded; consumed in Phase 3+) ---
    voxel_size: float = VOXEL_SIZE
    stat_nb_neighbors: int = STAT_NB_NEIGHBORS
    stat_std_ratio: float = STAT_STD_RATIO

    # --- global caveat ---
    metric_scale_unknown: bool = METRIC_SCALE_UNKNOWN