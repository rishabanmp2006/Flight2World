"""flight2world core package.

The reusable, immutable core extracted from the V10 experimental baseline
(`fusion_v10.py`, the scientific reference — DO NOT MODIFY).

Phase 5A scope:

    core/config.py         V10 + Phase 4/5A reference configuration
    core/colmap_io.py      pure COLMAP text-format readers
    core/calibration.py    robust per-frame calibration + RMSE outlier filtering
    core/curate.py         V10 greedy minimum-baseline curation
    core/track_maps.py     track depth map via iterative dilation
    core/depth.py          Depth Anything V2 wrapper
    core/gate.py           adaptive COLMAP gate
    core/fusion.py         track-anchored dense fusion
    core/video.py          video metadata + frame extraction (Phase 4)
    core/colmap_runner.py  COLMAP subprocess wrappers (Phase 4)
    core/diagnostics.py    structured stage diagnostics (Phase 4)
    core/pipeline.py       orchestration + resumability + metadata (Phase 4/5A)
    core/cleanup.py        voxel + statistical + DBSCAN cleanup (Phase 5A)
    core/confidence.py     per-point global confidence / error layer (Phase 5A)
    core/export.py         PLY + JSON I/O helpers (Phase 5A)

Deliberately NOT yet included (future phases): viewer, georeferencing,
measurements, mesh.
"""

__version__ = "0.5.0"
