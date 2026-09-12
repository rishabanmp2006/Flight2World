<div align="center">

# FLIGHT2WORLD

**Drone video → dense 3D point cloud**

COLMAP structure-from-motion · Depth Anything V2 monocular depth · track-anchored fusion

[![CI](https://github.com/rishabanmp2006/Flight2World/actions/workflows/ci.yml/badge.svg)](https://github.com/rishabanmp2006/Flight2World/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Phase 5A](https://img.shields.io/badge/phase-5A-orange.svg)](docs/ARCHITECTURE.md)

</div>

---

> [!IMPORTANT]
> **All geometry is in COLMAP's arbitrary coordinate system.** The source video
> carries no usable GPS or telemetry, so there is **no metric scale** — distances,
> areas and heights are in scene units, never metres. `metadata.json` records
> `coordinate_system: "COLMAP_relative"` and `metric_scale: null`, and it will
> not fabricate otherwise.

## What it does

Reconstructs a dense point cloud from ordinary drone footage. COLMAP recovers
camera poses and a sparse model; Depth Anything V2 predicts per-frame monocular
depth; a per-frame robust calibration fits that relative depth onto COLMAP's
scale; and a track-anchored fusion stage votes candidate points into the final
cloud, gated adaptively against sparse-point density. A confidence layer scores
every surviving point.

## Quick start

```bash
git clone https://github.com/rishabanmp2006/Flight2World.git
cd Flight2World

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

COLMAP and ffmpeg are external binaries and must be on `PATH`:

```bash
colmap --help     # required  (validated against 4.1.1)
ffprobe -version  # optional  (OpenCV is the fallback for video probing)
```

Then run the pipeline:

```python
from pathlib import Path
from core.pipeline import Pipeline, PipelineConfig

pipe = Pipeline(
    config=PipelineConfig(output_root=Path("outputs/run_001")),
    video_path=Path("drone.mp4"),
)
result = pipe.run(run_colmap=True)
```

Re-running with the same inputs is a **cache hit**: any stage with a valid
`diagnostics/*.json` (and, for extraction, a matching frame manifest) is
skipped. Force re-execution with `PipelineConfig(force=True)`.

## Repository layout

```
Flight2World/
├── core/                 the library — reusable pipeline modules
├── tests/                unit suite (deterministic; no model, no COLMAP)
├── experiments/          frozen V2→V10 research scripts  ← immutable
├── data/benchmark/       reference 86-frame dataset + artifacts  ← immutable
├── docs/ARCHITECTURE.md  line-by-line V10 map, data flow, phase history
└── outputs/              pipeline runs  ← generated, git-ignored
```

### Module map

| Module | Responsibility |
|---|---|
| [`core/config.py`](core/config.py) | Single source of truth — V10 thresholds, camera params, paths |
| [`core/colmap_io.py`](core/colmap_io.py) | COLMAP text-format readers (`qvec2rotmat`, `load_colmap_images/points`) |
| [`core/calibration.py`](core/calibration.py) | `robust_fit`, `calibrate_frame`, RMSE outlier filtering, `calibrated_depth` |
| [`core/curate.py`](core/curate.py) | Greedy minimum-baseline frame curation |
| [`core/track_maps.py`](core/track_maps.py) | `build_track_maps` — iterative dilation of sparse track depths |
| [`core/depth.py`](core/depth.py) | Depth Anything V2 wrapper; injectable factory, loads nothing on import |
| [`core/gate.py`](core/gate.py) | Adaptive COLMAP gate — sparse KD-tree, local radius, candidate gating |
| [`core/fusion.py`](core/fusion.py) | Track-anchored fusion — candidate grid, backprojection, voting, `fuse_all` |
| [`core/cleanup.py`](core/cleanup.py) | Voxel + statistical + DBSCAN cleanup |
| [`core/confidence.py`](core/confidence.py) | Per-point global confidence / error layer |
| [`core/export.py`](core/export.py) | PLY + JSON output with metadata |
| [`core/video.py`](core/video.py) | `inspect_video`, `extract_frames`, cache validation |
| [`core/colmap_runner.py`](core/colmap_runner.py) | COLMAP subprocess wrappers (features → matching → mapping → convert) |
| [`core/diagnostics.py`](core/diagnostics.py) | `Diagnostics` / `StageRecord` / `StageTimer`, JSON sidecars |
| [`core/pipeline.py`](core/pipeline.py) | Orchestration, resumability, honest metadata emission |

## Output layout

```
outputs/<run>/
├── frames/        frame_0001.jpg …  (+ .cache_manifest.json)
├── colmap/        database.db, sparse/0/*.bin, sparse_txt/*.txt
├── diagnostics/   extract_frames.json, colmap.json, curation.json …
└── metadata.json  coordinate_system, georeferenced: false, metric_scale: null
```

## Tests

```bash
pip install -r requirements-dev.txt

pytest tests                             # full suite
pytest tests -m "not requires_colmap"    # skip tests needing the binary
python -m tests.test_fusion              # any file runs standalone
```

The suite is deterministic: it downloads no model and invokes no COLMAP.
`tests/v10_reference.py` AST-extracts functions straight out of
`experiments/fusion_v10.py` and runs them head-to-head against `core/` on
identical inputs, so the refactor is validated numerically rather than by
eyeball.

The Depth Anything integration smoke test is opt-in — it downloads weights:

```bash
python tests/smoke_depth_integration.py
```

## Reproducing the V10 baseline

`experiments/fusion_v10.py` is the validated reference (55 frames,
Depth Anything V2 Base). It predates this layout and hardcodes
`~/flight2world/test`; see [`experiments/README.md`](experiments/README.md)
for the two symlinks that let it run unmodified.

## Status

Phase 5A. The reconstruction maths is the validated V10 baseline. Web viewer,
georeferencing, measurement and meshing are future phases — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Two rules dominate: `experiments/` and
`data/benchmark/` are immutable, and generated artifacts never get committed —
CI enforces both.
