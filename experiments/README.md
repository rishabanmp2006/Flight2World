# Experiments — frozen research lineage

**These scripts are immutable.** Each one is the permanent record of a specific
experiment. Do not refactor, reformat, or "fix" them: the production code in
[`core/`](../core) is validated by diffing its numerical output against
`fusion_v10.py`, so editing these files would invalidate the test suite.

Ruff is configured to skip this directory entirely (see `pyproject.toml`).

## Version history

| Version | Script | Frames | Depth model | Key change |
|---------|--------|-------:|-------------|------------|
| V2  | `fusion_v2.py`      |  5 | DAv2-Small | Initial prototype |
| V3  | `fusion_v3.py`      |  5 | DAv2-Small | Per-frame calibration |
| V3′ | `fusion_15.py`      | 15 | DAv2-Small | Scaled to 15 frames |
| V4  | `fusion_15_v4.py`   | 15 | DAv2-Small | Anchor-guided correction |
| V5  | `fusion_15_v5.py`   | 15 | DAv2-Small | Strict multi-view consistency |
| V6  | `fusion_15_v6.py`   | 15 | DAv2-Small | Hybrid + COLMAP spatial gate |
| V7  | `clean_v7.py`       | — (V6 output) | — | DBSCAN cleanup stage |
| V8  | `fusion_55_v8.py`   | 55 | DAv2-Small | Scaled to all 55 frames |
| V9  | `fusion_v9.py`      | 55 | DAv2-Small | Track-anchored fusion |
| **V10** | **`fusion_v10.py`** | **55** | **DAv2-Base** | **Depth model upgrade — the reference baseline** |

Supporting scripts: `confidence_v9.py`, `confidence_v10.py` (per-point
confidence layer), `compare_v9_v10.py`, `analyze_v9.py` (comparison and
analysis), `calibration_test.py` (calibration model selection study).

Their outputs — every PLY, JSON, log and PNG — live in
[`../data/benchmark/`](../data/benchmark) and are equally immutable.

## Re-running an experiment

These scripts predate the packaged layout and hardcode an absolute path,
`~/flight2world/test`. That path is part of the frozen artifact and has
deliberately **not** been rewritten. To run them, create the layout they
expect with symlinks:

```bash
ln -s "$(git rev-parse --show-toplevel)" ~/flight2world
ln -s "$(git rev-parse --show-toplevel)/data/benchmark" \
      "$(git rev-parse --show-toplevel)/test"
```

Then, from the repository root:

```bash
python experiments/fusion_v10.py

# Fast geometry check on the first 3 registered frames:
V9_SMOKE=3 python experiments/fusion_v10.py
```

Note that `test` is listed in `.gitignore`'s sibling patterns only as a
symlink convenience — do not commit it.

For anything new, use the modular pipeline in [`core/`](../core) instead.
