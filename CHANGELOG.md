# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
phase-based versioning (see `docs/ARCHITECTURE.md`).

## [Unreleased]

### Changed
- **Repository restructured** to a conventional Python layout. Frozen research
  scripts moved from the repository root to `experiments/`; the benchmark
  dataset moved from `test/` to `data/benchmark/` so it no longer collides
  visually with the `tests/` suite; `ARCHITECTURE.md` moved to `docs/`.
- `core/config.py` now derives paths from the package location and honours
  `FLIGHT2WORLD_DATA`, replacing a hardcoded `~/flight2world/test/sparse_txt`
  that only resolved on case-insensitive filesystems.

### Added
- `requirements.txt` / `requirements-dev.txt` — the README referenced a
  requirements file that did not exist.
- `pyproject.toml` with packaging metadata, pytest and ruff configuration.
- `.github/workflows/ci.yml` — tests on Python 3.12/3.13, lint, plus hygiene
  gates that fail the build on committed build artifacts or files over 20 MB.
- `.gitattributes`, `CONTRIBUTING.md`, `CHANGELOG.md`, and per-directory
  READMEs for `experiments/` and `data/`.

### Removed
- Untracked ~162 MB of generated pipeline output from git, including two
  51 MB COLMAP feature databases, 30 committed `.pyc` files, `.DS_Store`
  entries, and a 1.5 MB generated validation report. All are now ignored;
  the files remain on disk.

## [0.5.0] — Phase 5A

### Added
- `core/cleanup.py` — voxel + statistical + DBSCAN cleanup.
- `core/confidence.py` — per-point global confidence / error layer.
- `core/export.py` — PLY and JSON I/O helpers.

## [0.4.0] — Phase 4

### Added
- `core/pipeline.py` — orchestration, filesystem-cache resumability, and
  honest `metadata.json` emission.
- `core/video.py` — video inspection and frame extraction (OpenCV + ffprobe).
- `core/colmap_runner.py` — COLMAP subprocess wrappers.
- `core/diagnostics.py` — structured per-stage diagnostics sidecars.

## [0.1.0] — V10 baseline

### Added
- `experiments/fusion_v10.py` — the validated reference reconstruction:
  55 frames, Depth Anything V2 Base, track-anchored fusion.
- Extraction of the V10 maths into `core/` modules, validated numerically
  against the frozen script.
