# Contributing to FLIGHT2WORLD

## Setup

```bash
git clone https://github.com/rishabanmp2006/Flight2World.git
cd Flight2World
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

External binaries must be on `PATH`: `colmap` (>= 4.1.1) and, optionally,
`ffmpeg`/`ffprobe`.

```bash
colmap --help && ffprobe -version
```

## The two rules that matter

**1. `experiments/` and `data/benchmark/` are immutable.**
`experiments/fusion_v10.py` is the scientific reference baseline. The test
suite AST-extracts functions from it and runs them head-to-head against
`core/` on identical inputs, so any edit there silently changes what the
tests are validating against. Same for the dataset: tests read it, nothing
writes to it.

**2. Never commit generated artifacts.**
`outputs/`, `__pycache__/`, `*.pyc`, `*.db`, `.DS_Store` and rendered reports
are all ignored. CI fails the build if any of them appear in the index, or if
any tracked file exceeds 20 MB. Reconstruction runs belong in `outputs/`,
which is local-only.

## Tests

```bash
pytest tests                                  # full suite
pytest tests -m "not requires_colmap"         # skip COLMAP-dependent tests
pytest tests/test_fusion.py -v                # one module
```

Every test file is also directly runnable without pytest:

```bash
python -m tests.test_fusion
```

The Depth Anything integration smoke test is opt-in — it downloads model
weights:

```bash
python tests/smoke_depth_integration.py
```

## Lint

```bash
ruff check core tests
ruff check --fix core tests
```

`experiments/`, `data/` and `outputs/` are excluded by configuration.

## Adding a pipeline stage

1. Put the logic in a new `core/<stage>.py` — pure functions where possible,
   with dependencies injected so tests need neither COLMAP nor a model
   download (see `core/depth.py`'s `pipeline_factory` for the pattern).
2. Add thresholds to `core/config.py`. It is the single source of truth; do
   not scatter magic numbers.
3. Add `tests/test_<stage>.py`. If the stage exists in V10, diff it
   numerically against the extracted reference via `tests/v10_reference.py`.
4. Wire it into `core/pipeline.py` with a `diagnostics/<stage>.json` sidecar
   so the stage participates in cache-based resumability.

## Honesty constraints on output

`metadata.json` must never fabricate metres, GPS, or a metric scale where
none exists. Reconstructions are in COLMAP's arbitrary coordinate system;
`coordinate_system` stays `"COLMAP_relative"` and `metric_scale` stays `null`
until a genuine metric source (RTK/PPK/GCP) is supplied.

## Commits

Keep the subject in the imperative mood and under ~72 characters, prefixed by
area where it helps:

```
core/fusion: widen adaptive gate for sparse regions
docs: record V10 line map for confidence layer
```
