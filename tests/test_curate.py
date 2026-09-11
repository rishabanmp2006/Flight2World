"""tests.test_curate

Deterministic tests for core.curate (V10 frame curation) and
core.calibration.filter_rmse_outliers.

Curation is the V10 "greedy minimum-baseline" stage (fusion_v10.py lines
414-461) plus the RMSE outlier gate (lines 623-632). Tests cover:

  - deterministic frame ordering (sort by name)
  - existence filtering
  - baseline statistics / threshold
  - greedy selection edge cases
  - determinism
  - RMSE filtering (including head-to-head against V10)
"""

import math
import pathlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import run_module_tests, SPARSE_TXT, FRAMES

import core.calibration as cal
import core.colmap_io as cio
import core.config as cfg
import core.curate as curate

IMAGES_TXT = SPARSE_TXT / "images.txt"
POINTS_TXT = SPARSE_TXT / "points3D.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(name, R=None, t=None, obs=None):
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return {"id": 1, "name": name, "R": R, "t": t, "observations": obs or []}


def _make_registered(n, spacing=1.0):
    """Make n registered (image_id, image_data) with centres at x = i*spacing."""
    out = []
    for i in range(n):
        t = np.array([-spacing * i, 0.0, 0.0])  # centre = -R^T t = (spacing*i, 0, 0)
        out.append((i, _img(f"frame_{i:04d}.jpg", t=t)))
    return out


# ---------------------------------------------------------------------------
# Ordering / existence filter
# ---------------------------------------------------------------------------

def test_order_frames_sorts_by_name():
    regs = [(2, _img("frame_0003.jpg")), (1, _img("frame_0001.jpg")), (3, _img("frame_0002.jpg"))]
    ordered = curate.order_frames(regs)
    assert [d["name"] for _, d in ordered] == ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]


def test_order_frames_is_stable_and_deterministic():
    regs = [(i, _img(f"frame_{i:04d}.jpg")) for i in [5, 3, 1, 4, 2]]
    a = curate.order_frames(regs)
    b = curate.order_frames(list(reversed(regs)))
    assert [d["name"] for _, d in a] == [d["name"] for _, d in b]


def test_filter_registered_by_existing_frames():
    images = {
        1: _img("frame_0001.jpg"),
        2: _img("frame_0002.jpg"),
        3: _img("missing.jpg"),
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "frame_0001.jpg").write_text("x")
        (td / "frame_0002.jpg").write_text("x")
        filtered = curate.filter_registered_by_existing_frames(images, td)
        names = sorted(d["name"] for _, d in filtered)
        assert names == ["frame_0001.jpg", "frame_0002.jpg"]


def test_filter_registered_empty_when_no_files():
    images = {1: _img("frame_0001.jpg")}
    with tempfile.TemporaryDirectory() as td:
        filtered = curate.filter_registered_by_existing_frames(images, td)
        assert filtered == []


# ---------------------------------------------------------------------------
# Baseline stats
# ---------------------------------------------------------------------------

def test_compute_baseline_stats_two_points():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    stats = curate.compute_baseline_stats(centers, min_baseline_frac=0.1)
    assert stats["median"] == 1.0
    assert stats["min_baseline"] == 0.1
    assert stats["min"] == 1.0
    assert stats["max"] == 1.0


def test_compute_baseline_stats_single_and_empty():
    s0 = curate.compute_baseline_stats(np.empty((0, 3)))
    assert s0["median"] == 0.0 and s0["min_baseline"] == 0.0
    s1 = curate.compute_baseline_stats(np.array([[1.0, 2.0, 3.0]]))
    assert s1["median"] == 0.0


def test_compute_baseline_stats_median_of_many():
    # 5 points along x: baselines = [1,2,1,1] -> median 1.0
    centers = np.array([[0,0,0],[1,0,0],[3,0,0],[4,0,0],[5,0,0]], dtype=float)
    stats = curate.compute_baseline_stats(centers, min_baseline_frac=0.08)
    assert stats["median"] == 1.0
    assert math.isclose(stats["min_baseline"], 0.08)


# ---------------------------------------------------------------------------
# Greedy minimum-baseline
# ---------------------------------------------------------------------------

def test_greedy_keeps_all_when_well_spaced():
    regs = _make_registered(5, spacing=1.0)
    # median baseline = 1.0, threshold = 0.08
    result = curate.greedy_min_baseline(regs, min_baseline_frac=0.08)
    assert len(result.curated) == 5
    assert len(result.dropped) == 0


def test_greedy_drops_near_duplicates():
    # Alternating close/far: centres at 0, 0.01, 1, 1.01, 2
    # baselines = [0.01, 0.99, 0.01, 0.99] -> median ~0.5, threshold ~0.04
    # So 0 -> keep, 0.01 dropped (0.01 < 0.04), 1 kept (1.0 >= 0.04), 1.01 dropped, 2 kept
    xs = [0.0, 0.01, 1.0, 1.01, 2.0]
    regs = []
    for i, x in enumerate(xs):
        t = np.array([-x, 0.0, 0.0])
        regs.append((i, _img(f"frame_{i:04d}.jpg", t=t)))
    result = curate.greedy_min_baseline(regs, min_baseline_frac=0.08)
    kept_names = [d["name"] for _, d in result.curated]
    dropped_names = [d["name"] for _, d in result.dropped]
    assert len(result.curated) == 3
    assert kept_names == ["frame_0000.jpg", "frame_0002.jpg", "frame_0004.jpg"]
    assert dropped_names == ["frame_0001.jpg", "frame_0003.jpg"]


def test_greedy_first_frame_always_kept():
    regs = _make_registered(3, spacing=0.001)
    result = curate.greedy_min_baseline(regs)
    assert result.curated[0][1]["name"] == "frame_0000.jpg"


def test_greedy_empty_and_single():
    r0 = curate.greedy_min_baseline([])
    assert r0.curated == [] and r0.dropped == []
    r1 = curate.greedy_min_baseline(_make_registered(1))
    assert len(r1.curated) == 1 and len(r1.dropped) == 0


def test_greedy_identical_centres_keeps_only_first():
    regs = _make_registered(4, spacing=0.0)
    result = curate.greedy_min_baseline(regs)
    # All centres at origin -> baselines all 0, median 0, threshold 0
    # With threshold 0, every subsequent frame has distance 0 >= 0 -> kept
    # (norm == 0 satisfies >= 0). This matches V10's >= comparison.
    assert len(result.curated) == 4


def test_greedy_is_deterministic():
    regs = _make_registered(10, spacing=0.5)
    a = curate.greedy_min_baseline(regs)
    b = curate.greedy_min_baseline(regs)
    assert [d["name"] for _, d in a.curated] == [d["name"] for _, d in b.curated]
    assert a.baseline_stats == b.baseline_stats


# ---------------------------------------------------------------------------
# High-level curate_frames
# ---------------------------------------------------------------------------

def test_curate_frames_integration():
    images = {i: _img(f"frame_{i:04d}.jpg", t=np.array([-float(i), 0, 0])) for i in range(5)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(5):
            (td / f"frame_{i:04d}.jpg").write_text("x")
        result = curate.curate_frames(images, td)
        # Existence filter keeps all 5, baseline curation keeps all (well spaced)
        assert len(result.curated) == 5


def test_curate_frames_filters_missing():
    images = {i: _img(f"frame_{i:04d}.jpg") for i in range(3)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "frame_0001.jpg").write_text("x")
        result = curate.curate_frames(images, td)
        assert len(result.curated) == 1
        assert result.curated[0][1]["name"] == "frame_0001.jpg"


# ---------------------------------------------------------------------------
# Real COLMAP artifact smoke
# ---------------------------------------------------------------------------

def test_real_images_greedy_matches_v10_inline():
    """Head-to-head: extracted greedy must match V10 inline logic on real data."""
    images = cio.load_colmap_images(IMAGES_TXT)
    # Filter registered exactly as V10 does
    registered = [(iid, d) for iid, d in images.items() if (FRAMES / d["name"]).exists()]
    registered.sort(key=lambda x: x[1]["name"])

    # V10 inline computation
    centers = np.array([cio.camera_center(d["R"], d["t"]) for _, d in registered])
    seq_baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median_baseline = float(np.median(seq_baselines))
    min_baseline = cfg.MIN_BASELINE_FRAC * median_baseline
    v10_curated = []
    last_center = None
    for (iid, data), center in zip(registered, centers):
        if last_center is None:
            v10_curated.append((iid, data))
            last_center = center
            continue
        if np.linalg.norm(center - last_center) >= min_baseline:
            v10_curated.append((iid, data))
            last_center = center

    # Extracted
    result = curate.greedy_min_baseline(registered)

    assert len(result.curated) == len(v10_curated)
    assert [iid for iid, _ in result.curated] == [iid for iid, _ in v10_curated]
    assert math.isclose(result.min_baseline, min_baseline, rel_tol=1e-12)
    assert math.isclose(result.baseline_stats["median"], median_baseline, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# RMSE outlier filtering (Phase 4 — belongs in calibration)
# ---------------------------------------------------------------------------

def test_filter_rmse_outliers_basic():
    records = [{"rmse": 0.01}, {"rmse": 0.012}, {"rmse": 0.011}, {"rmse": 0.10}]
    kept, dropped, diag = cal.filter_rmse_outliers(records, factor=2.5)
    # median ~0.0115, threshold ~0.02875 -> 0.10 dropped
    assert len(kept) == 3
    assert len(dropped) == 1
    assert dropped[0]["rmse"] == 0.10
    assert diag["kept"] == 3 and diag["dropped"] == 1


def test_filter_rmse_outliers_all_kept_when_no_outlier():
    records = [{"rmse": 0.01}, {"rmse": 0.011}, {"rmse": 0.012}]
    kept, dropped, diag = cal.filter_rmse_outliers(records, factor=2.5)
    assert len(kept) == 3 and len(dropped) == 0


def test_filter_rmse_outliers_empty():
    kept, dropped, diag = cal.filter_rmse_outliers([])
    assert kept == [] and dropped == []
    assert diag["median"] == 0.0 and diag["threshold"] == 0.0


def test_filter_rmse_outliers_matches_v10_inline():
    """Head-to-head: extracted RMSE filter must match V10 inline on real-ish data."""
    rng = np.random.default_rng(42)
    rmses = rng.uniform(0.005, 0.02, 20).tolist()
    rmses.append(0.10)  # outlier
    records = [{"rmse": r, "name": f"f{i}"} for i, r in enumerate(rmses)]

    # V10 inline
    rmse_all = np.array([f["rmse"] for f in records])
    rmse_median = float(np.median(rmse_all))
    keep_mask = rmse_all <= cfg.CALIB_RMSE_OUTLIER_FACTOR * rmse_median
    v10_kept = [f for f, ok in zip(records, keep_mask) if ok]
    v10_dropped = [f for f, ok in zip(records, keep_mask) if not ok]

    kept, dropped, diag = cal.filter_rmse_outliers(records)
    assert len(kept) == len(v10_kept)
    assert len(dropped) == len(v10_dropped)
    assert math.isclose(diag["median"], rmse_median, rel_tol=1e-12)
    assert math.isclose(diag["threshold"], cfg.CALIB_RMSE_OUTLIER_FACTOR * rmse_median, rel_tol=1e-12)


def test_filter_rmse_outliers_default_factor_from_config():
    records = [{"rmse": 0.01}, {"rmse": 0.10}]
    kept, dropped, diag = cal.filter_rmse_outliers(records)
    assert diag["factor"] == cfg.CALIB_RMSE_OUTLIER_FACTOR


def test_filter_rmse_outliers_preserves_records():
    # Need enough inliers that median*2.5 < outlier. With 3 records
    # [0.01, 0.012, 0.10], median=0.012, threshold=0.03 -> 0.10 dropped.
    records = [{"rmse": 0.01, "name": "a"}, {"rmse": 0.012, "name": "b"}, {"rmse": 0.10, "name": "c"}]
    kept, dropped, _ = cal.filter_rmse_outliers(records, factor=2.5)
    assert len(dropped) == 1 and dropped[0] is records[2]
    assert kept[0] is records[0] and kept[1] is records[1]


if __name__ == "__main__":
    run_module_tests(globals())
