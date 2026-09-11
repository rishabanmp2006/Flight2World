"""flight2world.core.curate

Frame curation / ordering logic, extracted faithfully from `fusion_v10.py`
(the immutable V10 baseline). The algorithm is NOT redesigned.

V10 curation (fusion_v10.py lines 414-461, 623-632) consists of:

  1. Ordering: keep only COLMAP-registered frames whose JPEG exists on
     disk and sort deterministically by filename (chronological).
  2. Greedy minimum-baseline selection: keep a frame only if its camera
     centre is at least MIN_BASELINE_FRAC * median(consecutive baseline)
     from the last kept frame. This removes near-duplicate views while
     preserving a deterministic subsequence (V10's "curated" frames).
  3. RMSE outlier filtering: calibration quality gate handled in
     core.calibration.filter_rmse_outliers (see that module for the
     post-calibration stage). RMSE filtering is NOT duplicated here.

This module is deterministic, testable, and has no dependency on model
loading, Open3D visualisation, or CLI code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.colmap_io import camera_center
from core.config import MIN_BASELINE_FRAC


# ---------------------------------------------------------------------------
# Helpers preserved from V10
# ---------------------------------------------------------------------------

def order_frames(
    registered: Sequence[Tuple[int, dict]],
) -> List[Tuple[int, dict]]:
    """Sort ``(image_id, image_data)`` pairs by image name (chronological).

    Faithful to fusion_v10.py line 419:
        registered.sort(key=lambda x: x[1]["name"])
    """
    return sorted(registered, key=lambda kv: kv[1]["name"])


def filter_registered_by_existing_frames(
    images: Dict[int, dict],
    frame_dir: str | os.PathLike,
) -> List[Tuple[int, dict]]:
    """Keep only COLMAP images whose JPEG exists under *frame_dir*.

    Faithful to fusion_v10.py lines 414-418:
        registered = [(image_id, data) for image_id, data in images.items()
                      if os.path.exists(os.path.join(FRAME_DIR, data["name"]))]
    """
    frame_dir = Path(frame_dir)
    out: List[Tuple[int, dict]] = []
    for image_id, data in images.items():
        if (frame_dir / data["name"]).exists():
            out.append((image_id, data))
    return out


def compute_baseline_stats(
    centers: np.ndarray,
    min_baseline_frac: float = MIN_BASELINE_FRAC,
) -> dict:
    """Compute V10 baseline statistics from ordered camera centres.

    Mirrors fusion_v10.py lines 435-441. Returns a dict with
    ``median``, ``min``, ``max``, and ``min_baseline`` (the curation
    threshold). For fewer than 2 centres the baselines are empty and
    ``median``/``min_baseline`` are 0.0.
    """
    if len(centers) < 2:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "min_baseline": 0.0}
    seq_baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median = float(np.median(seq_baselines))
    return {
        "median": median,
        "min": float(seq_baselines.min()),
        "max": float(seq_baselines.max()),
        "min_baseline": float(min_baseline_frac * median),
    }


# ---------------------------------------------------------------------------
# Greedy minimum-baseline curation (V10 lines 435-461)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurationResult:
    """Result of greedy minimum-baseline curation."""

    curated: List[Tuple[int, dict]]
    dropped: List[Tuple[int, dict]]
    baseline_stats: dict
    min_baseline: float


def greedy_min_baseline(
    registered: Sequence[Tuple[int, dict]],
    min_baseline_frac: float = MIN_BASELINE_FRAC,
) -> CurationResult:
    """Greedy minimum-baseline curation, faithful to fusion_v10.py.

    *registered* must already be ordered deterministically (e.g. via
    :func:`order_frames`). The threshold is
    ``MIN_BASELINE_FRAC * median(consecutive baseline)`` computed from
    the ordered centres. The first frame is always kept; each subsequent
    frame is kept iff ``||centre - last_kept_centre|| >= threshold``.

    Returns a :class:`CurationResult` with the kept subsequence, the
    dropped frames, and the baseline diagnostics.
    """
    if len(registered) == 0:
        stats = compute_baseline_stats(np.empty((0, 3)), min_baseline_frac)
        return CurationResult(curated=[], dropped=[], baseline_stats=stats, min_baseline=0.0)
    if len(registered) == 1:
        stats = compute_baseline_stats(np.empty((0, 3)), min_baseline_frac)
        return CurationResult(curated=list(registered), dropped=[], baseline_stats=stats, min_baseline=0.0)

    centers = np.array(
        [camera_center(data["R"], data["t"]) for _, data in registered],
        dtype=np.float64,
    )
    stats = compute_baseline_stats(centers, min_baseline_frac)
    min_baseline = stats["min_baseline"]

    curated: List[Tuple[int, dict]] = []
    dropped: List[Tuple[int, dict]] = []
    last_center = None
    for (image_id, data), center in zip(registered, centers):
        if last_center is None:
            curated.append((image_id, data))
            last_center = center
            continue
        if float(np.linalg.norm(center - last_center)) >= min_baseline:
            curated.append((image_id, data))
            last_center = center
        else:
            dropped.append((image_id, data))

    return CurationResult(
        curated=curated,
        dropped=dropped,
        baseline_stats=stats,
        min_baseline=float(min_baseline),
    )


def curate_frames(
    images: Dict[int, dict],
    frame_dir: str | os.PathLike,
    min_baseline_frac: float = MIN_BASELINE_FRAC,
) -> CurationResult:
    """High-level curation: existence filter → ordering → min-baseline.

    Combines the V10 "order frames chronologically" (line 419) and
    "greedy minimum-baseline frame curation" (lines 435-461) stages into
    a single convenience entry point. No calibration or RMSE filtering is
    performed here (see ``core.calibration.filter_rmse_outliers``).
    """
    registered = filter_registered_by_existing_frames(images, frame_dir)
    ordered = order_frames(registered)
    return greedy_min_baseline(ordered, min_baseline_frac=min_baseline_frac)
