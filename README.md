# FLIGHT2WORLD — Drone Video → Point Cloud

Reconstructs a dense point cloud from drone video using COLMAP (SfM) + Depth Anything V2 (monocular depth) + track-anchored fusion. All geometry is in COLMAP's **arbitrary coordinate system** — no GPS, no metric scale claimed.

> **Status:** Phase 4 — pipeline/orchestration layer. The reconstruction maths is the validated V10 baseline (`fusion_v10.py`, immutable). Web UI, georeferencing, measurements and meshing are future phases.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or: pip install opencv-python open3d transformers torch pillow numpy
colmap --help   # must be on PATH (tested: colmap 4.1.1)
ffmpeg -version # for video probing (optional; OpenCV is the fallback)
```

Benchmark data (86 frames, COLMAP sparse model, reference PLYs) lives under `test/` and is not modified by the pipeline.

## Module map

```
core/config.py          single source of truth — V10 thresholds + Phase 4 (video/COLMAP/pipeline)
core/colmap_io.py       COLMAP text-format readers (qvec2rotmat, load_colmap_images/points, camera_center)
core/calibration.py     robust_fit, calibrate_frame, fit_depth_calibration, filter_rmse_outliers, calibrated_depth
core/curate.py          greedy minimum-baseline curation (V10 lines 414-461)
core/track_maps.py      build_track_maps — iterative dilation (V10 lines 330-382)
core/depth.py           DepthAnythingV2 wrapper (injectable pipeline_factory for tests)
core/gate.py            adaptive COLMAP gate — build_sparse_kdtree, compute_local_radius, gate_candidates
core/fusion.py          track-anchored fusion — candidate_grid, sample_candidates, backproject, cam_to_world,
                        track_anchored_votes, fuse_frame, fuse_all
core/video.py           inspect_video, extract_frames, frames_cache_valid (OpenCV + ffprobe)
core/colmap_runner.py   feature_extraction, sequential_matching, sparse_mapping, model_converter, run_colmap
core/diagnostics.py     Diagnostics / StageRecord / StageTimer, diagnostics/*.json sidecars
core/pipeline.py        Pipeline, PipelineConfig, stage_* helpers, build_metadata / write_metadata,
                        filesystem cache resumability, {output_root}/frames|colmap|diagnostics|metadata.json
tests/                  deterministic unit tests (no model download, no COLMAP invocation in the suite)
```

V10 baseline scripts (`fusion_v*.py`, `confidence_*.py`, etc.) are **immutable experiments** at the repo root.

## Running tests

```bash
.venv/bin/python -m tests.test_colmap_io
.venv/bin/python -m tests.test_calibration
.venv/bin/python -m tests.test_track_maps
.venv/bin/python -m tests.test_depth
.venv/bin/python -m tests.test_gate
.venv/bin/python -m tests.test_fusion
.venv/bin/python -m tests.test_curate
.venv/bin/python -m tests.test_video
.venv/bin/python -m tests.test_colmap_runner
.venv/bin/python -m tests.test_pipeline
# or loop them:
for m in test_colmap_io test_calibration test_track_maps test_depth test_gate test_fusion test_curate test_video test_colmap_runner test_pipeline; do
  .venv/bin/python -m tests.$m
done
```

The Depth Anything integration smoke test is **opt-in** (downloads the model):

```bash
.venv/bin/python tests/smoke_depth_integration.py
```

## Pipeline (Phase 4)

```python
from pathlib import Path
from core.pipeline import Pipeline, PipelineConfig
from core.video import ExtractionConfig
from core.colmap_runner import ColmapRunConfig

# Minimal: just produce metadata + enumerate existing frames
pipe = Pipeline(config=PipelineConfig(output_root=Path("outputs/run_001")))
result = pipe.run()  # success, metadata.json written

# With video → frames (cached; re-run is a no-op when manifest matches)
pipe = Pipeline(
    config=PipelineConfig(output_root=Path("outputs/run_001")),
    video_path=Path("drone.mp4"),
)
result = pipe.run()  # extracts frames/frame_*.jpg, writes diagnostics/extract_frames.json

# With COLMAP (expensive — opt-in)
result = pipe.run(run_colmap=True)

# Stage-by-stage (for scripts / notebooks)
from core.pipeline import stage_extract_frames, stage_run_colmap, stage_curate_and_summarise
paths, rec = stage_extract_frames(video_path, output_root, extraction_config=ExtractionConfig(target_fps=1.0))
```

Output layout:

```
{output_root}/
  frames/              frame_0001.jpg ...  (+ .cache_manifest.json)
  colmap/              database.db, sparse/0/{cameras,images,points3D}.bin, sparse_txt/{...}.txt
  diagnostics/         extract_frames.json, colmap.json, curation.json ...
  metadata.json        {coordinate_system: "COLMAP_relative", georeferenced: false, metric_scale: null, ...}
```

Re-running the pipeline with the same inputs is a **cache hit** — stages with a valid `diagnostics/*.json` (`status == "ok"`) and (for video) a matching `.cache_manifest.json` are skipped. Pass `PipelineConfig(force=True)` to force re-execution.

`metadata.json` **never** fabricates metres, GPS, or a metric scale when none exists. It records `coordinate_system: "COLMAP_relative"` honestly.

## Reproducing the V10 reference

```bash
# The reference experiment is a standalone script (not the modular pipeline)
.venv/bin/python fusion_v10.py
# or a smoke slice (first N registered frames) to validate geometry fast:
V9_SMOKE=3 .venv/bin/python fusion_v10.py
```

## Coordinate system

All reconstructions are in COLMAP's arbitrary scene units. `metric_scale` is `null`. Distances/areas/heights must not be presented as metres unless a genuine metric source (RTK/PPK/GCP) is supplied — future phase, not yet implemented.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the V10 line-by-line module map, data-flow diagram, and Phase history.

## Licence / data

Benchmark frames and COLMAP outputs under `test/` are the project dataset. Do not delete or overwrite them; the test suite reads them but never writes to `test/`.
