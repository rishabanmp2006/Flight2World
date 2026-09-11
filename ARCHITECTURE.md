# FLIGHT2WORLD — Architecture & Migration Document

> Phase 1 deliverable. This document maps every line of the current
> experimental codebase to a proposed modular architecture and defines
> the migration strategy. No code is modified in this phase.

---

## A. Current System

### A.1 Repository layout (flat, experimental)

```
flight2world/
├── fusion_v10.py          992 lines   ← CURRENT BASELINE (DO NOT MODIFY)
├── fusion_v9.py           991 lines   ← V9 experiment (DO NOT MODIFY)
├── fusion_55_v8.py       1187 lines   ← V8 experiment (DO NOT MODIFY)
├── fusion_15_v6.py       1631 lines   ← V6 experiment (DO NOT MODIFY)
├── fusion_15_v5.py       1433 lines   ← V5 experiment (DO NOT MODIFY)
├── fusion_15_v4.py       1835 lines   ← V4 experiment (DO NOT MODIFY)
├── fusion_15.py          1384 lines   ← V3 experiment (DO NOT MODIFY)
├── fusion_v3.py          1248 lines   ← V3 experiment (DO NOT MODIFY)
├── fusion_v2.py           701 lines   ← V2 experiment (DO NOT MODIFY)
├── confidence_v10.py      610 lines   ← V10 confidence layer (DO NOT MODIFY)
├── confidence_v9.py       610 lines   ← V9 confidence layer (DO NOT MODIFY)
├── compare_v9_v10.py      290 lines   ← V9 vs V10 comparison (DO NOT MODIFY)
├── analyze_v9.py          157 lines   ← multi-version analysis (DO NOT MODIFY)
├── clean_v7.py            313 lines   ← V7 cleanup (DO NOT MODIFY)
├── calibration_test.py    502 lines   ← calibration study (DO NOT MODIFY)
├── test/                             ← data + results (DO NOT MODIFY)
│   ├── frames/              86 JPG frames (1280×720, ~16MB)
│   ├── sparse_txt/          COLMAP text export (55 images, 20262 pts)
│   ├── sparse/              COLMAP binary output (2 models)
│   ├── database/            COLMAP SQLite database (108MB)
│   ├── analysis/            18 rendered comparison PNGs
│   ├── *.ply                16 point cloud files
│   ├── *.json               7 diagnostic/comparison JSONs
│   └── *.log                3 run logs
└── .venv/                            ← Python 3.14 virtualenv
```

### A.2 Installed packages (all already present, zero-cost)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.14.0 | Depth model backend (MPS) |
| transformers | 5.16.1 | HuggingFace depth pipeline |
| open3d | 0.19.0 | Point cloud I/O, cleanup, visualization |
| opencv-python | 5.0.0 | Track map dilation, image resize |
| numpy | 2.5.3 | All numerical computation |
| pillow | 12.3.0 | Image loading |
| dash | 4.4.1 | Web visualization framework |
| plotly | 7.0.0 | 3D scatter rendering |
| flask | 3.1.3 | Dash server backend |
| psutil | 7.2.2 | Memory monitoring |
| tqdm | 4.70.0 | Progress bars |
| pyyaml | 6.3 | Config file parsing (available) |
| typer | 0.27.2 | CLI framework (available) |
| rich | 15.0.0 | Terminal formatting (available) |

External tools on PATH: `colmap` 4.1.1, `ffmpeg`, `ffprobe`

### A.3 Key limitation

The supplied drone video has **no usable GPS/telemetry metadata**.
All reconstructions are in COLMAP's **arbitrary coordinate system**.
Distances, areas, and heights are in **scene units**, not metric.
No geospatial accuracy is claimed or implied.

---

## B. V10 Algorithm & Data Flow

### B.1 Complete line-by-line map of `fusion_v10.py`

```
Lines    Section                          Target Module
──────   ─────────────────────────────    ─────────────────────
  1-8    Imports                           (standard library + deps)
 11-48   Header comment (V9/V10 history)   (historical, stays in V10)
 50-64   PATHS + SMOKE_TEST               core/config.py
 66-77   CAMERA INTRINSICS                core/config.py
 79-127  SETTINGS (all thresholds)         core/config.py
─────────────────────────────────────────────────────────────────
130-221  COLMAP LOADERS                    core/colmap_io.py
 133     qvec2rotmat(q)                    core/colmap_io.py
 142     load_colmap_images(path)          core/colmap_io.py
 198     load_colmap_points(path)          core/colmap_io.py
 218     camera_center(R, t)               core/colmap_io.py
─────────────────────────────────────────────────────────────────
223-316  ROBUST CALIBRATION                core/calibration.py
 234     robust_fit(x, z, iterations)      core/calibration.py
 286     calibrate_frame(img, pts, depth)  core/calibration.py
─────────────────────────────────────────────────────────────────
319-382  TRACK DEPTH MAP                   core/track_maps.py
 330     build_track_maps(img, pts)        core/track_maps.py
─────────────────────────────────────────────────────────────────
385-489  MAIN: Load COLMAP + Curate        core/frame_selection.py
 401-402 Load images.txt + points3D.txt    core/colmap_io.py (called)
 414-419 Filter registered + sort          core/frame_selection.py
 423-428 Smoke test slice                  core/frame_selection.py
 435-461 Greedy min-baseline curation      core/frame_selection.py
 468-489 Sparse cloud + local_r density    core/geometry.py
─────────────────────────────────────────────────────────────────
492-505  Load Depth Anything               core/depth.py
 499-503 pipeline("depth-estimation",..)   core/depth.py
─────────────────────────────────────────────────────────────────
508-632  PASS 1: DEPTH + CALIB + TRACKS    app/pipeline.py
 521-534 Per-frame depth inference          core/depth.py (called)
 535-539 Resize predicted depth             core/depth.py (called)
 541     calibrate_frame()                 core/calibration.py (called)
 542-576 Reject: few_samples, low_corr     app/pipeline.py
 553-566 Fit linear + inverse, pick best   core/calibration.py (called)
 578     build_track_maps()                core/track_maps.py (called)
 579-583 Reject: few_tracks                app/pipeline.py
 585-608 Build frame_records dict          app/pipeline.py
 615-632 RMSE outlier gate                 app/pipeline.py
─────────────────────────────────────────────────────────────────
635-826  PASS 2: TRACK-ANCHORED FUSION     core/fusion.py
 645-653 Accumulator init                  core/fusion.py
 656-659 calibrated_depth(frame, d)        core/fusion.py (helper)
 662-825 Main fusion loop:
  676-688 Sample pixel grid + calibrated Z  core/fusion.py
  699-702 Backproject + world transform     core/fusion.py
  709-727 ADAPTIVE COLMAP GATE             core/geometry.py (called)
  740-807 TRACK-ANCHORED VOTING            core/fusion.py
  809-822 Net vote filter + accumulate     core/fusion.py
─────────────────────────────────────────────────────────────────
828-832  Fusion stage totals               core/fusion.py (return stats)
─────────────────────────────────────────────────────────────────
835-868  MERGE + CLEANUP                   core/filtering.py
 845-852 Vstack + o3d PointCloud           core/filtering.py
 854-865 Voxel downsample + stat outlier   core/filtering.py
 867-868 Write OUTPUT_RAW                  core/export.py
─────────────────────────────────────────────────────────────────
871-909  DIAGNOSTICS                       core/export.py
 875-908 Build diag dict + write JSON      core/export.py
─────────────────────────────────────────────────────────────────
912-984  DBSCAN CLEAN STAGE               core/filtering.py
 922-984 Adaptive eps + main cluster       core/filtering.py
 975-977 Write OUTPUT_CLEAN               core/export.py
 979-981 Update diag JSON                 core/export.py
─────────────────────────────────────────────────────────────────
987-991  V10 COMPLETE banner              app/pipeline.py
```

### B.2 Complete line-by-line map of `confidence_v10.py`

```
Lines    Section                          Target Module
──────   ─────────────────────────────    ─────────────────────
  1-9    Imports                           (standard library)
 10-36   Header comment                    (historical)
 37-51   PATHS                             core/config.py
 53-69   INTRINSICS + SETTINGS            core/config.py (shared)
 71-159  COLMAP loaders                    core/colmap_io.py (shared)
160-216  build_track_maps()               core/track_maps.py (shared)
218-286  CAMERA + PROJECTION helpers       core/geometry.py
─────────────────────────────────────────────────────────────────
288-610  MAIN: confidence computation      core/confidence.py
 290-345 Load COLMAP + frames              core/confidence.py
 347-400 Replay calibration + track maps   core/confidence.py
 402-475 Per-point voting (global)         core/confidence.py
 477-540 Confidence + residual stats       core/confidence.py
 542-570 dist_sparse computation           core/confidence.py
 572-610 PLY export + JSON summary         core/export.py
```

### B.3 Data flow (V10 single-pass pipeline)

```
video.mp4
    │
    ▼  [ffmpeg, external]
frames/frame_NNNN.jpg
    │
    ▼  [COLMAP, external]
sparse_txt/{cameras,images,points3D}.txt
    │
    ├───────────────────────────────────────┐
    │                                       │
    ▼                                       ▼
colmap_io.load_colmap_images()     colmap_io.load_colmap_points()
    │                                       │
    ├──── frame_selection.curate() ─────────┘
    │         │
    │         ▼
    │    curated_frames (52 of 55)
    │         │
    │         ├──► geometry.build_sparse_tree()
    │         │         │
    │         │         ▼
    │         │    sparse_tree + local_r (adaptive gate data)
    │         │
    │         ▼
    │    ┌─── PASS 1: per-frame depth + calibration ────┐
    │    │  depth.estimate(frame)                        │
    │    │  calibration.calibrate_frame(img, pts, depth) │
    │    │  calibration.robust_fit(d, z)                 │
    │    │  track_maps.build_track_maps(img, pts)        │
    │    │  ──────────────────────────────────────────   │
    │    │  Output: frame_records[] with:                │
    │    │    {name, R, t, depth, rgb, a, b, model,     │
    │    │     rmse, corr, samples, z_grid, dist_grid}   │
    │    └───────────────────────────────────────────────┘
    │         │
    │         ▼
    │    RMSE outlier gate → 49 frames
    │         │
    │         ▼
    │    ┌─── PASS 2: track-anchored fusion ────────────┐
    │    │  for each frame:                              │
    │    │    candidate_generation (pixel grid)           │
    │    │    backproject (camera → world)                │
    │    │    geometry.adaptive_gate()                    │
    │    │    reprojection + track voting (±3 neighbors)  │
    │    │    net_votes >= MIN_NET_VOTES → keep           │
    │    │    accumulate world_pts + colors               │
    │    └───────────────────────────────────────────────┘
    │         │
    │         ▼
    │    filtering.cleanup(points, colors)
    │      ├── voxel_downsample(0.04)
    │      ├── remove_statistical_outlier(30, 1.5)
    │      └── dbscan_main_cluster()
    │         │
    │         ▼
    │    export.save_ply(fused_v10_clean.ply)
    │    export.save_diag(fused_v10_diag.json)
    │
    ▼  [post-hoc, no depth model needed]
confidence_v10.py
    │  Replay voting on final clean cloud
    │  Compute per-point: confidence, residual, worst_residual,
    │                     evidence, dist_sparse
    ▼
export.save_ply(fused_v10_confidence.ply)
export.save_json(fused_v10_confidence.json)
```

### B.4 V10 record structure (`frame_records[]`)

Each calibrated frame is stored as a dict with these fields:

```python
{
    "name": str,           # e.g. "frame_0027.jpg"
    "R": np.ndarray,       # 3×3 rotation (world→camera)
    "t": np.ndarray,       # 3×1 translation (world→camera)
    "depth": np.ndarray,   # float32 (H, W) raw predicted_depth
    "rgb": np.ndarray,     # uint8 (H, W, 3) source image
    "a": float,            # calibration slope
    "b": float,            # calibration intercept
    "model": str,          # "linear" or "inverse"
    "rmse": float,         # calibration RMSE
    "corr": float,         # calibration correlation
    "samples": int,        # number of (depth, Z) pairs used
    "z_grid": np.ndarray,  # float32 (H, W) track depth map
    "dist_grid": np.ndarray,# int32 (H, W) dilation distance
}
```

This structure is the **interface contract** between the calibration
stage and the fusion stage. Any refactoring must preserve this exactly.

---

## C. Existing Experiment Lineage

### C.1 Version history

| Version | Script | Frames | Model | Key change |
|---------|--------|--------|-------|------------|
| V2 | fusion_v2.py | 5 | DAv2-Small | Initial prototype |
| V3 | fusion_v3.py | 5 | DAv2-Small | Per-frame calibration |
| V3 (15) | fusion_15.py | 15 | DAv2-Small | Scaled to 15 frames |
| V4 | fusion_15_v4.py | 15 | DAv2-Small | Anchor-guided correction |
| V5 | fusion_15_v5.py | 15 | DAv2-Small | Strict multi-view consistency |
| V6 | fusion_15_v6.py | 15 | DAv2-Small | Hybrid + COLMAP spatial gate |
| V7 | clean_v7.py | (V6 output) | — | DBSCAN cleanup stage |
| V8 | fusion_55_v8.py | 55 | DAv2-Small | Scaled to all 55 frames |
| V9 | fusion_v9.py | 55 | DAv2-Small | Track-anchored fusion |
| V10 | fusion_v10.py | 55 | DAv2-Base | Depth model upgrade |

### C.2 Artifacts to preserve

Every PLY, JSON, log, and rendered PNG in `test/` is a permanent
record of a specific experiment. The `experiments/` directory will
hold the Python scripts; the `test/` directory holds their outputs.
Neither should be modified.

---

## D. Proposed Production Architecture

### D.1 Target directory structure

```
flight2world/
├── ARCHITECTURE.md            ← THIS DOCUMENT
├── README.md                  ← project overview (Phase 2)
├── requirements.txt           ← pinned dependencies (Phase 2)
├── pyproject.toml             ← package metadata (Phase 2)
│
├── core/                      ← reusable pipeline modules
│   ├── __init__.py
│   ├── config.py              ← PipelineConfig dataclass + defaults
│   ├── colmap_io.py           ← COLMAP text format parsers
│   ├── calibration.py         ← robust_fit, calibrate_frame, model selection
│   ├── track_maps.py          ← build_track_maps (iterative dilation)
│   ├── frame_selection.py     ← greedy min-baseline curation
│   ├── depth.py               ← Depth Anything V2 wrapper
│   ├── geometry.py            ← backprojection, adaptive gate, sparse tree
│   ├── fusion.py              ← track-anchored fusion core
│   ├── filtering.py           ← voxel + statistical + DBSCAN cleanup
│   ├── confidence.py          ← per-point confidence/error layer
│   └── export.py              ← PLY/JSON output with metadata
│
├── app/                       ← orchestration layer
│   ├── __init__.py
│   ├── pipeline.py            ← full pipeline: video → cloud
│   └── cli.py                 ← typer CLI entry point
│
├── viewer/                    ← visualization (Phase 2+)
│   ├── __init__.py
│   ├── open3d_viewer.py       ← desktop interactive viewer
│   └── dash_viewer.py         ← web-based viewer
│
├── experiments/               ← IMMUTABLE experiment scripts
│   ├── README.md              ← explains these are historical
│   ├── fusion_v2.py
│   ├── fusion_v3.py
│   ├── fusion_15.py
│   ├── fusion_15_v4.py
│   ├── fusion_15_v5.py
│   ├── fusion_15_v6.py
│   ├── fusion_55_v8.py
│   ├── fusion_v9.py
│   ├── fusion_v10.py          ← THE REFERENCE BASELINE
│   ├── confidence_v9.py
│   ├── confidence_v10.py
│   ├── compare_v9_v10.py
│   ├── analyze_v9.py
│   ├── clean_v7.py
│   └── calibration_test.py
│
├── test/                      ← data + results (IMMUTABLE)
│   ├── frames/
│   ├── sparse_txt/
│   ├── sparse/
│   ├── database/
│   ├── analysis/
│   ├── *.ply
│   ├── *.json
│   └── *.log
│
└── tests/                     ← unit + integration tests (Phase 2)
    ├── __init__.py
    ├── test_colmap_io.py
    ├── test_calibration.py
    ├── test_track_maps.py
    └── test_regression.py     ← V10 reproducibility check
```

### D.2 Design decisions

**Why not `app/` and `core/` as separate packages?**
They share the same venv. A flat `core/` import path (`from core.fusion import ...`) is simpler than nested packages for a project of this size. `app/` is thin orchestration; `core/` is the library.

**Why not a monolithic refactor?**
V10 is 992 lines of interleaved functions and main-loop code.
Extracting modules incrementally lets us test each in isolation
and diff against the original to verify bit-identical behavior.

**Why `experiments/` instead of deleting old scripts?**
Each script is a frozen research artifact. Researchers may need to
re-run, compare, or cite specific versions. Moving them to
`experiments/` with a README preserves them without cluttering
the production import path.

**Why no `setup.py` / editable install yet?**
We can run `python -m app.cli` or `PYTHONPATH=. python app/cli.py`
during Phase 1. A proper package install is Phase 2 polish.

---

## E. Module Responsibilities

### E.1 `core/config.py`

**Purpose:** Single source of truth for all pipeline constants and
the `PipelineConfig` dataclass that replaces module-level globals.

**Extracted from:** fusion_v10.py lines 50-127 (PATHS, INTRINSICS, SETTINGS)

```python
@dataclass
class PipelineConfig:
    # Paths
    base_dir: str = "~/flight2world"
    frame_dir: str = ""          # default: {base}/test/frames
    sparse_dir: str = ""         # default: {base}/test/sparse_txt
    output_dir: str = ""         # default: {base}/test

    # Camera intrinsics (from COLMAP cameras.txt)
    fx: float = 1167.4277386087481
    fy: float = 1167.4277386087481
    cx: float = 640.0
    cy: float = 360.0
    image_w: int = 1280
    image_h: int = 720

    # Fusion parameters
    pixel_stride: int = 8
    neighbor_window: int = 3
    min_net_votes: int = 2
    relative_tolerance: float = 0.18
    absolute_tolerance: float = 0.05
    track_radius_px: int = 50

    # Adaptive COLMAP gate
    gate_factor: float = 2.0
    gate_radius_min: float = 0.015
    gate_radius_max: float = 0.12

    # Frame curation
    min_baseline_frac: float = 0.08

    # Calibration
    calib_min_corr: float = 0.55
    calib_min_samples: int = 100
    calib_rmse_outlier_factor: float = 2.5

    # Cleanup
    voxel_size: float = 0.04
    stat_nb_neighbors: int = 30
    stat_std_ratio: float = 1.5

    # Depth model
    depth_model: str = "depth-anything/Depth-Anything-V2-Base-hf"
    depth_device: str = "mps"
```

**Key point:** V10's exact behavior is reproduced by `PipelineConfig()`
(defaults match V10's hardcoded constants exactly).

### E.2 `core/colmap_io.py`

**Purpose:** Parse COLMAP text-format outputs. Zero dependencies beyond
numpy. Pure I/O, no computation.

**Extracted from:** fusion_v10.py lines 130-221

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `qvec2rotmat(q)` | 133-139 | `(4,) → (3,3)` |
| `load_colmap_images(path)` | 142-195 | `str → dict[int, dict]` |
| `load_colmap_points(path)` | 198-215 | `str → dict[int, ndarray]` |
| `camera_center(R, t)` | 218-220 | `(3,3), (3,) → (3,)` |

**Also used by:** confidence_v10.py (lines 77-159, identical copies),
confidence_v9.py, fusion_v9.py, fusion_55_v8.py.

**Shared across:** 5 scripts today. Deduplication eliminates ~190 lines
of copy-pasted code.

### E.3 `core/calibration.py`

**Purpose:** Robust per-frame depth-to-COLMAP-Z calibration.

**Extracted from:** fusion_v10.py lines 223-316

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `robust_fit(x, z, iterations=5)` | 234-283 | `ndarray, ndarray → dict or None` |
| `calibrate_frame(image_data, sparse_points, depth_map, img_w, img_h)` | 286-316 | `dict, dict, ndarray → (ndarray, ndarray)` |

**Note:** `calibrate_frame` currently references module-level
`IMAGE_W`, `IMAGE_H`. In the refactored version, these come from
`PipelineConfig`. The signature changes to accept `img_w, img_h`
as explicit parameters (or a config object).

**Also used by:** fusion_v9.py (identical).

### E.4 `core/track_maps.py`

**Purpose:** Build approximate nearest-track depth maps via iterative
dilation. Depends on cv2 and numpy only.

**Extracted from:** fusion_v10.py lines 319-382

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `build_track_maps(image_data, sparse_points, img_w, img_h)` | 330-382 | `dict, dict → (ndarray, ndarray, int)` |

Returns `(z_grid, dist_grid, n_tracks)`.

**Also used by:** confidence_v9.py, confidence_v10.py, fusion_v9.py
(all identical copies, ~50 lines each).

### E.5 `core/frame_selection.py`

**Purpose:** Filter COLMAP-registered frames to remove near-duplicate
views that add noise but no parallax.

**Extracted from:** fusion_v10.py lines 385-461

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `curate_frames(registered, centers, min_baseline_frac)` | 435-461 | `list, ndarray, float → list` |

Takes the registered frame list (after chronological sort) and
returns the curated subset. Pure function, no I/O.

**Also used by:** fusion_v9.py (identical).

### E.6 `core/depth.py`

**Purpose:** Load and run Depth Anything V2 for monocular depth
estimation. Thin wrapper around HuggingFace pipeline.

**Extracted from:** fusion_v10.py lines 492-539

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `load_depth_model(model_name, device)` | 499-503 | `str, str → pipeline` |
| `estimate_depth(depth_pipe, image, target_h, target_w)` | 525-539 | `pipeline, Image → ndarray` |

Returns `predicted_depth` as float32 numpy array at target resolution.

### E.7 `core/geometry.py`

**Purpose:** Sparse cloud spatial indexing, local density estimation,
backprojection, and adaptive COLMAP gating.

**Extracted from:** fusion_v10.py lines 468-489 + 699-727

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `build_sparse_tree(sparse_xyz)` | 468-472 | `ndarray → (KDTreeFlann, ndarray)` |
| `compute_local_density(sparse_tree, sparse_xyz, k=8)` | 476-484 | `KDTreeFlann, ndarray → ndarray` |
| `backproject_to_world(cam_pts, R, t)` | 699-702 | `ndarray, ndarray, ndarray → ndarray` |
| `adaptive_gate(world_pts, sparse_tree, local_r, config)` | 709-720 | `... → ndarray[bool]` |

**Also used by:** confidence_v10.py (lines 218-286, projection helpers).

### E.8 `core/fusion.py`

**Purpose:** The core track-anchored fusion algorithm. This is the
largest and most critical module (~160 lines).

**Extracted from:** fusion_v10.py lines 635-832

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `calibrated_depth(frame, d)` | 656-659 | `dict, ndarray → ndarray` |
| `fuse_frame(frame, frame_records, frame_idx, sparse_tree, local_r, config)` | 662-825 | `... → (ndarray, ndarray, dict)` |

The fusion loop is the heart of the algorithm. It:
1. Samples the pixel grid and applies calibrated depth (lines 676-688)
2. Backprojects to world coordinates (lines 699-702)
3. Applies adaptive COLMAP gate (lines 709-727)
4. Runs track-anchored reprojection voting over ±3 neighbors (lines 740-807)
5. Filters by net vote threshold (lines 809-822)

**Critical interface:** The `frame_records[]` list (section B.4 above)
is passed in its entirety. The function accesses neighbor frames
by index offset (NEIGHBOR_WINDOW), so the list ordering matters.

### E.9 `core/filtering.py`

**Purpose:** Post-fusion point cloud cleanup.

**Extracted from:** fusion_v10.py lines 835-984

| Function | Source lines | Signature |
|----------|-------------|-----------|
| `merge_points(all_points, all_colors)` | 845-852 | `list, list → o3d.PointCloud` |
| `cleanup_cloud(pcd, config)` | 854-865 | `PointCloud → PointCloud` |
| `dbscan_clean(pcd)` | 912-984 | `PointCloud → PointCloud` |

**Also extracted from:** clean_v7.py (the DBSCAN algorithm origin).

### E.10 `core/confidence.py`

**Purpose:** Post-hoc per-point confidence/error layer. Runs without
the depth model (replays voting geometry only).

**Extracted from:** confidence_v10.py lines 288-570

| Function | Signature |
|----------|-----------|
| `compute_confidence(clean_pcd, frame_records, sparse_points, config)` | `PointCloud, list, dict, config → dict` |

Returns per-point arrays: `{confidence, residual, worst_residual,
evidence, dist_sparse, net_votes}` plus summary statistics.

### E.11 `core/export.py`

**Purpose:** Write output files with proper metadata.

| Function | Purpose |
|----------|---------|
| `save_ply(path, pcd, extra_fields=None)` | ASCII PLY with XYZRGB + scalars |
| `save_config_json(path, config)` | Serialize pipeline config |
| `save_diagnostics(path, diag_dict)` | Full pipeline diagnostics |

### E.12 `app/pipeline.py`

**Purpose:** Orchestrate the full pipeline end-to-end. Calls into
`core/` modules in sequence. Handles progress reporting, error
recovery, and intermediate state.

```python
def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full V10-equivalent pipeline."""
    # 1. Load COLMAP
    # 2. Curate frames
    # 3. Build sparse tree + density
    # 4. Load depth model
    # 5. Pass 1: depth + calibration + track maps
    # 6. RMSE outlier gate
    # 7. Pass 2: fusion
    # 8. Cleanup
    # 9. Export
    # 10. Diagnostics
```

### E.13 `app/cli.py`

**Purpose:** CLI entry point using typer.

```bash
python -m app.cli run --video flight.mp4    # full pipeline
python -m app.cli run --frames test/frames  # skip extraction
python -m app.cli view test/fused_v10_clean.ply  # open viewer
```

---

## F. Dependency Graph

```
                    core/config.py
                         │
            ┌────────────┼────────────────┐
            │            │                │
            ▼            ▼                ▼
     core/colmap_io.py  core/depth.py  core/export.py
            │            │
     ┌──────┼──────┐     │
     │      │      │     │
     ▼      ▼      ▼     │
  frame_  calib-  track_  │
  select  ration  maps    │
     │      │      │      │
     │      └──┬───┘      │
     │         │          │
     ▼         ▼          │
  core/geometry.py        │
     │                    │
     ▼                    │
  core/fusion.py          │
     │                    │
     ▼                    │
  core/filtering.py ──────┘
     │
     ▼
  core/confidence.py (post-hoc, optional)
```

External dependencies (not imported between modules):
- `torch` / `transformers` → only in `core/depth.py`
- `cv2` → only in `core/track_maps.py`
- `open3d` → in `core/geometry.py`, `core/filtering.py`, `core/export.py`
- `numpy` → everywhere

---

## G. Migration Order

Each step produces a module that can be **tested independently**
against the existing V10 data before moving to the next.

### Step 1: `core/config.py` (new file)
- Create `PipelineConfig` dataclass with V10 defaults
- No dependencies beyond `dataclasses` + `os`
- Test: instantiate config, verify all values match V10 constants

### Step 2: `core/colmap_io.py` (extract from fusion_v10.py)
- Copy lines 130-221 verbatim
- Remove dependence on module-level IMAGE_W/IMAGE_H (none in these functions)
- Test: parse existing `test/sparse_txt/`, verify same output as V10

### Step 3: `core/calibration.py` (extract from fusion_v10.py)
- Copy lines 223-316
- Replace `IMAGE_W`, `IMAGE_H` globals with parameters
- Test: run on V10's existing frame data, compare calibration output

### Step 4: `core/track_maps.py` (extract from fusion_v10.py)
- Copy lines 319-382
- Replace `IMAGE_W`, `IMAGE_H` globals with parameters
- Test: run on V10's frame_records, compare z_grid/dist_grid

### Step 5: `core/geometry.py` (extract from fusion_v10.py)
- Extract sparse tree building (lines 468-489)
- Extract backprojection (lines 699-702)
- Extract adaptive gate (lines 709-727)
- Test: gate V10's candidates, verify same pass/fail

### Step 6: `core/depth.py` (extract from fusion_v10.py)
- Extract model loading (lines 492-505) and inference (lines 525-539)
- Test: run on one frame, compare depth output

### Step 7: `core/frame_selection.py` (extract from fusion_v10.py)
- Extract curation (lines 414-461)
- Test: curate V10's registered frames, verify 52 → same curated set

### Step 8: `core/fusion.py` (extract from fusion_v10.py)
- Extract calibrated_depth helper (lines 656-659)
- Extract fusion loop body (lines 676-825)
- This is the largest extraction (~160 lines)
- Test: run on V10's frame_records, compare point counts per frame

### Step 9: `core/filtering.py` (extract from fusion_v10.py)
- Extract merge + voxel + stat outlier (lines 835-868)
- Extract DBSCAN clean (lines 912-984)
- Test: run on V10's fused points, compare clean output

### Step 10: `core/confidence.py` (extract from confidence_v10.py)
- Extract confidence computation (lines 288-570)
- Test: run on V10's clean cloud, compare confidence JSON

### Step 11: `core/export.py` (new file)
- PLY writer with extra scalar fields
- JSON diagnostics writer
- Test: round-trip V10's existing PLY files

### Step 12: `app/pipeline.py` (new file)
- Wire all core modules together
- Test: full pipeline on existing frames, compare to V10 output

### Step 13: `app/cli.py` (new file)
- Typer CLI wrapping pipeline.py
- Test: CLI invocation produces same output

### Step 14: Move experiment scripts to `experiments/`
- Move all fusion_*.py, confidence_*.py, etc. to experiments/
- Update any cross-references
- Test: V10 still reproducible from experiments/fusion_v10.py

---

## H. V10 Reproducibility Strategy

### H.1 Regression test contract

The following must remain **bit-identical** across refactoring:

| Artifact | Source | Verification |
|----------|--------|-------------|
| `fused_v10_diag.json` | fusion_v10.py | SHA-256 hash comparison |
| `fused_v10_confidence.json` | confidence_v10.py | SHA-256 hash comparison |
| `fused_v10_clean.ply` | fusion_v10.py | Point count + bounding box |
| `fused_v10_confidence.ply` | confidence_v10.py | Point count + scalar stats |

### H.2 Bit-identical reproduction protocol

To verify V10 is unchanged after any refactoring step:

```bash
# 1. Run original V10 (from experiments/ after move)
cd experiments
python fusion_v10.py
cp ../test/fused_v10_diag.json /tmp/diag_original.json

# 2. Run refactored pipeline
python -m app.cli run --config v10_defaults
cp test/fused_v10_diag.json /tmp/diag_refactored.json

# 3. Compare
diff /tmp/diag_original.json /tmp/diag_refactored.json
# Must be empty (identical)
```

### H.3 Why bit-identical is achievable

V10 uses deterministic algorithms:
- Depth model inference: same input image → same predicted_depth
  (fixed model weights, no dropout at inference)
- COLMAP data: read from files, no randomness
- Calibration: IRWLS with fixed iteration count and trimming
- Fusion: purely deterministic (no stochastic sampling)
- Cleanup: voxel grid + DBSCAN are deterministic given same input

The only source of non-determinism would be floating-point order
of operations changes from code restructuring. By keeping the
exact same numpy/cv2/o3d call sequences, bit-identical output
is preserved.

### H.4 Golden reference files

After this migration, the following files serve as golden references:

```
test/fused_v10_diag.json          ← full pipeline config + stats
test/fused_v10_confidence.json    ← confidence summary
test/fused_v10_run.log            ← complete execution log
```

Any future pipeline change that alters these outputs is a
**behavioral change** that must be documented and justified.

---

## I. Testing Strategy

### I.1 Unit tests (per module)

| Test file | Tests |
|-----------|-------|
| `test_colmap_io.py` | Parse cameras.txt, images.txt, points3D.txt; verify quaternion→rotation; verify camera_center |
| `test_calibration.py` | robust_fit on known linear/inverse data; calibrate_frame on V10's frame_0027; verify rejection thresholds |
| `test_track_maps.py` | build_track_maps on V10's frame_0027; verify track count, z_grid shape, dist_grid range |
| `test_geometry.py` | backproject known camera→world; adaptive_gate with known sparse cloud |
| `test_frame_selection.py` | curate_frames on V10's registered list; verify 52→curated count |
| `test_fusion.py` | fuse_frame on V10's frame_records[0]; verify candidate/gate/vote counts |
| `test_filtering.py` | cleanup_cloud on V10's fused points; verify point count |
| `test_config.py` | PipelineConfig() defaults match V10 constants exactly |

### I.2 Regression tests

| Test | What it checks |
|------|---------------|
| `test_v10_repro.py` | Full pipeline with V10 config produces same diag JSON |
| `test_v10_confidence.py` | Confidence layer on V10 cloud produces same confidence JSON |

### I.3 Test data strategy

- Use **existing V10 artifacts** as test fixtures (not synthetic data)
- `test/frames/frame_0027.jpg` through `frame_0030.jpg` as minimal test set
- `test/sparse_txt/` as COLMAP fixture
- `test/fused_v10_clean.ply` as input fixture for confidence tests

### I.4 What is NOT tested in Phase 1

- End-to-end video→cloud (requires COLMAP run, ~5 minutes)
- Dash/Open3D viewer (interactive, not unit-testable)
- Performance/memory (measured manually, not automated)

---

## J. Future Extension Points

### J.1 Live drone video ingestion

**Current:** ffmpeg extracts frames from a pre-recorded video file.

**Extension:** `core/video.py` with a `VideoSource` abstraction:

```python
class VideoSource(Protocol):
    def frames(self) -> Iterator[tuple[int, np.ndarray]]: ...
    def metadata(self) -> VideoMetadata: ...

class FileVideoSource(VideoSource): ...      # ffmpeg from .mp4
class RTSPVideoSource(VideoSource): ...      # live drone RTSP stream
class DirectoryVideoSource(VideoSource): ... # pre-extracted frames
```

COLMAP would need incremental mapping for live operation, or a
"batch after N frames" approach.

### J.2 GPS/IMU metadata

**Current:** No GPS available. Reconstruction is arbitrary-scale.

**Extension:** When a drone video includes DJI metadata or NMEA sentences:

1. `core/metadata.py` — extract GPS from video container or sidecar CSV
2. `core/georef.py` — compute similarity transform (rotation + scale + translation)
   from COLMAP arbitrary coords → WGS84 localENU
3. Apply transform to point cloud, camera poses
4. Export with geo-referenced PLY (per-vertex lat/lon/alt)

**Key constraint:** Must never fabricate GPS when none exists.
The georef module should raise明确 error if no GPS data found.

### J.3 Ground Control Points (GCPs)

**Extension:** When survey-grade GCP coordinates are available:

1. `core/gcp.py` — load GCP file (CSV with name, lat, lon, alt, pixel coords)
2. `core/georef.py` — refine transform using GCP residuals
3. Report GCP residuals (cm-level accuracy metric)

### J.4 Geospatial anchoring

**Extension:** Combining GPS + GCP for full geo-referencing:

1. `core/georef.py` — seven-parameter Helmert transform
2. Per-point uncertainty propagation (GPS accuracy → point cloud confidence)
3. Export in georeferenced PLY/LAZ with coordinate system metadata

### J.5 Confidence/error visualization

**Current:** Static PNG renders via numpy splatting.

**Extension:**
1. `viewer/dash_viewer.py` — Dash web app with:
   - Plotly Scatter3d colored by confidence
   - Toggle: RGB / confidence / residual / evidence / dist_sparse
   - Camera path overlay (shows COLMAP poses)
   - Hover tooltips with per-point stats
2. `viewer/open3d_viewer.py` — Open3D desktop with:
   - Color-by-scalar dropdown
   - Point picking for measurement
   - View save/restore

### J.6 Progressive 3D visualization

**Extension:** Show reconstruction progress as frames are processed:

1. `app/pipeline.py` emits events: `FrameStart`, `DepthComplete`,
   `FusionComplete`, `PointCloudUpdated`
2. Viewer subscribes and updates incrementally
3. Uses Open3D's `Visualizer.add_geometry()` / `update_geometry()`
   for live point cloud growth

### J.7 Measurements

**Extension:** Distance, area, height measurements on the point cloud:

1. Open3D point picking (shift-click two points → distance)
2. Plane fitting (pick 3 points → plane equation, height above ground)
3. Polygon area (pick N points → enclosed area)
4. All in **scene units** (arbitrary scale) unless georeferenced

### J.8 Textured mesh

**Extension:** Surface reconstruction from point cloud:

1. `core/meshing.py` — Poisson surface reconstruction (open3d)
2. `core/texturing.py` — Project source images onto mesh (UV mapping)
3. Export as OBJ with MTL + texture PNG

**Caveat:** Poisson reconstruction on 30K points produces a smooth
surface but may lose fine detail. Ball pivoting or alpha shapes
may be better for this density.

### J.9 Web frontend

**Extension:** Full web application replacing CLI:

1. Upload video via browser
2. Server runs pipeline with progress WebSocket
3. Interactive 3D viewer in browser (Three.js or Plotly)
4. Download point cloud / mesh
5. Measurement tools in browser

**Stack:** Flask (already installed) + Dash (already installed) +
Plotly (already installed). No new dependencies needed.

---

## K. Files Inspected

| File | Lines | Action |
|------|-------|--------|
| `fusion_v10.py` | 992 | Read completely (line-by-line mapping) |
| `fusion_v9.py` | 991 | Read headers + key sections (differs from V10 in model name only) |
| `confidence_v10.py` | 610 | Read completely (reusable functions identified) |
| `confidence_v9.py` | 610 | Read headers (identical to V10 except paths) |
| `compare_v9_v10.py` | 290 | Read completely (evaluation functions identified) |
| `analyze_v9.py` | 157 | Read headers |
| `clean_v7.py` | 313 | Read headers (DBSCAN algorithm origin) |
| `calibration_test.py` | 502 | Read headers |
| `fusion_55_v8.py` | 1187 | Read headers |
| `fusion_15_v6.py` | 1631 | Read headers |
| `fusion_15_v5.py` | 1433 | Read headers |
| `fusion_15_v4.py` | 1835 | Read headers |
| `fusion_15.py` | 1384 | Read headers |
| `fusion_v3.py` | 1248 | Read headers |
| `fusion_v2.py` | 701 | Read headers |
| `test/sparse_txt/cameras.txt` | 2 | Read |
| `test/sparse_txt/images.txt` | 114 | Read header + sample |
| `test/sparse_txt/points3D.txt` | 20265 | Line count verified |
| `test/fused_v10_diag.json` | ~90 | Read (from earlier conversation) |
| `test/fused_v10_confidence.json` | ~50 | Read |
| `test/v9_v10_comparison.json` | 117 | Read |
| `test/database/database.db` | — | Schema inspected (SQLite) |
| `.venv/bin/pip list` | — | Full package inventory |

## L. Files Created

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | This document |

## M. Files Modified

**None.** No existing files were modified.

## N. Files NOT Modified

| Category | Files |
|----------|-------|
| Experiment scripts (15) | `fusion_v2.py`, `fusion_v3.py`, `fusion_15.py`, `fusion_15_v4.py`, `fusion_15_v5.py`, `fusion_15_v6.py`, `fusion_55_v8.py`, `fusion_v9.py`, `fusion_v10.py`, `confidence_v9.py`, `confidence_v10.py`, `compare_v9_v10.py`, `analyze_v9.py`, `clean_v7.py`, `calibration_test.py` |
| Test data (all) | `test/` directory contents (frames, PLY, JSON, logs, COLMAP outputs, analysis PNGs) |

## O. Packages NOT Installed

No packages were installed. The existing venv already contains all
required dependencies for the planned architecture.

---

## P. Phase History

| Phase | Scope | Status | Tests |
|-------|-------|--------|-------|
| 1 | Architecture survey + line-by-line V10 map | 2026-09-09 COMPLETE | — |
| 2 | `core/config`, `core/colmap_io`, `core/calibration` + 28 tests | 2026-09-09 COMPLETE | 28/28 |
| 3 | `core/track_maps`, `core/depth`, `core/gate`, `core/fusion` + V10 head-to-head | 2026-09-09 COMPLETE | +48 (76/76) |
| 4 | Pipeline / orchestration — see §Q | 2026-09-10 COMPLETE | +81 (157/157) |

## Q. Phase 4 — Pipeline / Orchestration (2026-09-10, COMPLETE)

### Q.1 Goal

Connect the validated reconstruction components (Phase 2/3) with a
resumable, testable orchestration layer without changing the V10
mathematics. Video → frames → curation → COLMAP → depth →
calibration → fusion → cleanup → output, with filesystem caching and
honest `metadata.json` (no fabricated metres/GPS).

Phase 4 explicitly does **not** build: web UI, measurements, mesh/DEM,
georeferencing, GPS/RTK/GCP fusion, segmentation, splatting.

### Q.2 Modules added

| Module | Source in V10 | Responsibility |
|--------|---------------|---------------|
| `core/config.py` (extended) | lines 50-127 + new Phase 4 sections | Video/COLMAP/pipeline/output constants, `VideoConfig`, `ColmapConfig`, `PipelineConfig` remains canonical `PipelineConfig` for V10 thresholds |
| `core/curate.py` | lines 414-461 (greedy min-baseline) | `order_frames`, `filter_registered_by_existing_frames`, `compute_baseline_stats`, `greedy_min_baseline`, `curate_frames`, `CurationResult` — deterministic, no model/Open3D/CLI dep |
| `core/calibration.py` (extended) | lines 623-632 (RMSE outlier gate) | `filter_rmse_outliers(frame_records, factor)` → `(kept, dropped, diagnostics)` — pure, testable; default `factor = CALIB_RMSE_OUTLIER_FACTOR = 2.5` from `core.config` |
| `core/video.py` | new (Phase 4) | `VideoInfo`, `ExtractionConfig`, `ExtractionResult`, `inspect_video` (OpenCV + ffprobe codec), `extract_frames` (FPS-controlled, deterministic `frame_{idx:04d}.jpg`, resize via `INTER_AREA`/`INTER_LINEAR`, JPEG quality), `frames_cache_valid` + `.cache_manifest.json` |
| `core/colmap_runner.py` | new (Phase 4) | `resolve_colmap_exe`, `check_colmap_available`, `feature_extraction` (`SIMPLE_RADIAL`, `single_camera=1`), `sequential_matching` (`overlap=10`), `sparse_mapping`, `model_converter`, `run_colmap` + `ColmapRunConfig`/`ColmapCommandResult`/`ColmapError` — all args as list, no `shell=True` |
| `core/diagnostics.py` | new (Phase 4) | `StageRecord`, `Diagnostics`, `StageTimer`, `write_stage_diagnostics`, `load_stage_diagnostics` — `{output_root}/diagnostics/*.json` sidecars |
| `core/pipeline.py` | new (Phase 4) | `PipelineConfig` (run root, `force`), `PipelineResult`, `build_metadata`/`write_metadata`/`load_metadata` (honest `coordinate_system="COLMAP_relative"`, `georeferenced=false`, `metric_scale=null`), `stage_cached`, `stage_extract_frames`, `stage_run_colmap`, `stage_curate_and_summarise`, `Pipeline` (resumable, per-stage cache via `diagnostics/*.json` + video `.cache_manifest.json`, independent stage callability) |

`core/__init__.py` bumped to `0.4.0` with Phase 4 scope docstring.

### Q.3 Output layout

```
{output_root}/
  frames/              frame_0001.jpg ... + .cache_manifest.json  (video → frames)
  colmap/              database.db, sparse/0/{cameras,images,points3D}.bin, sparse_txt/{...}.txt
  diagnostics/         extract_frames.json, colmap.json, curation.json ...
  metadata.json        {coordinate_system: "COLMAP_relative", georeferenced: false, metric_scale: null, stages: [...]}
```

A stage with `diagnostics/{stage}.json` (`status == "ok"`) and, for video, a matching `.cache_manifest.json` (same `video_path` + `ExtractionConfig.cache_key()` + all frames present) is **skipped** on re-run. `force=True` bypasses the cache. Metadata is written even on failure.

### Q.4 V10 fidelity

- All V10 constants in `core/config.py` remain byte-identical to `fusion_v10.py` (verified by SHA-256 and `ast.literal_eval` comparison).
- `core/curate.greedy_min_baseline` is line-for-line the V10 loop (`min_baseline = MIN_BASELINE_FRAC * median(diff(centres))`, `>=` comparison, first frame always kept).
- `core/calibration.filter_rmse_outliers` is the V10 `rmse_all <= CALIB_RMSE_OUTLIER_FACTOR * median` gate.
- `core/colmap_runner` validated defaults: `ImageReader.camera_model=SIMPLE_RADIAL`, `ImageReader.single_camera=1`, `SequentialMatching.overlap=10` (from `core/config`, not hardcoded in call sites).
- `core/pipeline` does not reimplement fusion maths — it calls the validated `core.*` modules.

### Q.5 Tests added (81 new, 157 total — all passing on 2026-09-10)

| Suite | Tests | What is covered |
|-------|-------|----------------|
| `tests/test_curate.py` | 22 | Ordering, existence filter, baseline stats (empty/single/median), greedy edge cases (well-spaced, near-duplicate, first-always-kept, identical centres, determinism), high-level `curate_frames`, real-artifact head-to-head greedy vs V10 inline, RMSE outlier filtering (basic/all-kept/empty/head-to-head against V10/default factor/identity preservation) |
| `tests/test_video.py` | 15 | `VideoInfo`/`ExtractionConfig` shape, `inspect_video` (missing/garbage/synthetic), `extract_frames` (basic/no-resize/deterministic/missing/cache-valid/cache-hit/overwrite/delete-invalidation/auto-mkdir), real benchmark JPEG still 1280×720 |
| `tests/test_colmap_runner.py` | 23 | `resolve_colmap_exe` (explicit/missing/bare-on-PATH/not-on-PATH), `check_colmap_available`, command construction for all four stages (defaults, custom args, `extra_args`, spaced paths), list-not-shell safety, failure → `ColmapError` per stage, `run_colmap` orchestration + failure propagation, config benchmark defaults |
| `tests/test_pipeline.py` | 21 | `StageRecord`/`Diagnostics`/`StageTimer`, metadata honesty (`COLMAP_relative`, no metres, `metric_scale=null`, extra merge, write/load), stage helpers (`stage_curate_and_summarise`/`stage_extract_frames`/`stage_run_colmap` with real artifacts + fake COLMAP/video), `Pipeline.run` (metadata produced, video extraction, resumability cached vs `force`, failure propagation for extract/COLMAP, diagnostics ordering, `stage_cached` helper) |

No test downloads Depth Anything; no test runs a real COLMAP reconstruction on the full video; `tests/smoke_depth_integration.py` remains opt-in.

### Q.6 Known limitations / decisions

- `Pipeline.run(run_colmap=False)` by default (COLMAP is expensive and opt-in) — `stage_run_colmap` remains independently callable.
- Video FPS handling uses `round(ts * src_fps)` timestamp sampling; containers reporting `frame_count == 0` fall back to `duration*fps` estimate or `fps / target_fps` throttling.
- COLMAP text export uses the largest `sparse/0` model by `images.bin` size — matches the benchmark's single-model case.
- `core/curate` does not yet implement quality-gated frame selection beyond the V10 baseline curation; the track-count / `few_tracks` gate remains in the per-frame calibration stage (as in V10 pass 1).

---

*Document updated: 2026-09-10 — Phase 4 complete*
*V10 reproducibility status: VERIFIED — fusion_v10.py SHA bbe8005111115bba81bc3aa26673f1c45371e26314f647d3209a57dd74dae604 unchanged*
