# Data

## `benchmark/` — the reference dataset (read-only)

The validated 86-frame drone sequence and every artifact produced from it.
Formerly the top-level `test/` directory; renamed to `data/benchmark/` so it
is no longer one letter away from the [`tests/`](../tests) unit suite.

```
benchmark/
├── frames/          86 extracted JPEG frames (frame_0001.jpg …)
├── sparse/          COLMAP binary sparse model (0/, 1/)
├── sparse_txt/      COLMAP text export — cameras, images, points3D
├── analysis/        rendered comparison PNGs (V9/V10 views)
├── *.ply            fused point clouds per experiment version
├── *.json           confidence + diagnostic sidecars
└── *.log            experiment run logs
```

**Treat this directory as immutable.** The test suite reads from it and never
writes to it; the pipeline writes to `outputs/` instead. Each file is the
permanent record of a specific experiment in
[`../experiments/`](../experiments).

### Pointing the code at a different dataset

Both `core.config` and the test suite honour an environment variable, so you
do not need to move or copy files:

```bash
export FLIGHT2WORLD_DATA=/path/to/other/dataset
```

It must contain `sparse_txt/` and `frames/` in the same layout.

## Coordinate system

Everything here is in COLMAP's **arbitrary** coordinate system. There is no
usable GPS or telemetry in the source video: `metric_scale` is `null` and
distances are in scene units, never metres.
