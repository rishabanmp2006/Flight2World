"""tests.test_calibration

Deterministic tests for core.calibration. Covers requirement E (synthetic
linear / inverse recovery, sparse observation collection) and F (degenerate
cases fail cleanly instead of yielding NaN / nonsense), plus a structural
check against the real V10 artifacts for image 27.
"""

import math
import pathlib
import sys

# Make the project root importable regardless of how this file is invoked
# (`python tests/test_x.py` puts tests/ on sys.path, not the project root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import SPARSE_TXT, run_module_tests

import core.calibration as cal
import core.colmap_io as cio
import core.config as cfg

IMAGES = SPARSE_TXT / "images.txt"
POINTS = SPARSE_TXT / "points3D.txt"
FRAMES = SPARSE_TXT.parent / "frames"          # ../benchmark/frames (source images)
FRAME_IMG = FRAMES / "frame_0027.jpg"

N = 400  # synthetic sample count (> robust_fit min 50 and default min 100)


# ------------------------------------------------------------
# E1. Linear calibration recovery
# ------------------------------------------------------------
def test_robust_fit_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 10.0, N)
    scale, offset = 3.0, -1.5
    cam_z = scale * depth + offset + rng.normal(0, 0.01, N)  # small noise
    fit = cal.robust_fit(depth, cam_z)
    assert fit is not None
    assert math.isclose(fit["a"], scale, rel_tol=1e-2)
    assert math.isclose(fit["b"], offset, abs_tol=0.05)
    assert fit["corr"] > 0.99
    assert fit["rmse"] < 0.03
    assert fit["samples"] >= 50


def test_fit_depth_calibration_selects_linear_when_linear_is_exact():
    depth = np.linspace(0.5, 8.0, N)
    cam_z = 2.0 * depth + 1.0                       # exact linear, no noise
    res = cal.fit_depth_calibration(depth, cam_z, min_samples=50)
    assert res is not None
    assert res["model"] == "linear"
    assert math.isclose(res["a"], 2.0, rel_tol=1e-6)
    assert math.isclose(res["b"], 1.0, abs_tol=1e-6)


def test_fit_depth_calibration_selects_inverse_when_inverse_is_exact():
    # V10 treats predicted depth as INVERSE depth; when Z = a/D + b is
    # exact, the inverse model must win (lower RMSE).
    depth = np.linspace(1.5, 10.0, N)
    magnitude, offset = 12.0, 0.7
    cam_z = magnitude / depth + offset
    res = cal.fit_depth_calibration(depth, cam_z, min_samples=50)
    assert res is not None
    assert res["model"] == "inverse", "inverse model should be selected"
    assert math.isclose(res["a"], magnitude, rel_tol=1e-4)
    assert math.isclose(res["b"], offset, abs_tol=1e-4)


# ------------------------------------------------------------
# E2. Sparse observation collection (calibrate_frame)
# ------------------------------------------------------------
def _synthetic_frame(observations, R, t):
    return {
        "id": 1, "name": "synth.jpg", "R": np.array(R, dtype=np.float64),
        "t": np.array(t, dtype=np.float64),
        "observations": observations,
    }


def test_calibrate_frame_collects_expected_pairs():
    # Identity pose, one world point observed at pixel (100, 80).
    R = np.eye(3)
    t = np.zeros(3)
    X_world = np.array([2.0, 1.0, 5.0])           # Z_cam = 5.0
    obs = [(100.0, 80.0, 1)]
    frame = _synthetic_frame(obs, R, t)
    sparse = {1: X_world}

    depth = np.full((cfg.IMAGE_H, cfg.IMAGE_W), 7.25)  # d = 7.25 at every px
    d_vals, z_vals = cal.calibrate_frame(frame, sparse, depth)
    assert len(d_vals) == 1 and len(z_vals) == 1
    assert math.isclose(float(d_vals[0]), 7.25, abs_tol=1e-12)
    assert math.isclose(float(z_vals[0]), 5.0, abs_tol=1e-12)   # Z = Rw+t = z


def test_calibrate_frame_rounds_observation_to_pixel():
    R = np.eye(3)
    t = np.zeros(3)
    X_world = np.array([0.0, 0.0, 4.0])           # Z_cam = 4.0
    frame = _synthetic_frame([(100.4, 80.6, 1)], R, t)
    sparse = {1: X_world}
    depth = np.full((cfg.IMAGE_H, cfg.IMAGE_W), 0.0)
    depth[81, 100] = 9.9                          # int(round(80.6))=81, col 100
    d_vals, z_vals = cal.calibrate_frame(frame, sparse, depth)
    # (x=100.4 rounds to 100; y=80.6 rounds to 81) -> depth[81,100] = 9.9
    assert len(d_vals) == 1
    assert math.isclose(float(d_vals[0]), 9.9, abs_tol=1e-9)
    assert math.isclose(float(z_vals[0]), 4.0, abs_tol=1e-12)


def test_calibrate_frame_skips_nonpositive_z():
    R = np.eye(3)
    t = np.zeros(3)
    behind = np.array([0.0, 0.0, -1.0])           # behind camera, Z<0
    frame = _synthetic_frame([(100.0, 80.0, 1)], R, t)
    sparse = {1: behind}
    depth = np.full((cfg.IMAGE_H, cfg.IMAGE_W), 5.0)
    d_vals, z_vals = cal.calibrate_frame(frame, sparse, depth)
    assert len(d_vals) == 0 and len(z_vals) == 0   # cleanly excluded


# ------------------------------------------------------------
# E3. Real-artifact collection structure (frame_0027)
# ------------------------------------------------------------
def test_real_frame_has_calibration_observations():
    # Structural check on real V10 artifacts (frame_0027): with a positive,
    # finite constant depth, every observing track in front of the camera
    # contributes one (depth, COLMAP-Z) pair. Deterministic -- no depth
    # model is needed, because projected depth at each observation pixel is
    # the same constant.
    depth_map = np.full((cfg.IMAGE_H, cfg.IMAGE_W), 5.0, np.float32)
    imgs = cio.load_colmap_images(IMAGES)
    pts = cio.load_colmap_points(POINTS)
    d_vals, z_vals = cal.calibrate_frame(
        imgs[27], pts, depth_map, cfg.IMAGE_W, cfg.IMAGE_H)
    # frame_0027's obs with a positive depth should number in the hundreds.
    assert len(d_vals) > 500, "expected many pairs from frame_0027, got %d" % len(d_vals)
    assert len(d_vals) == len(z_vals)
    assert np.all(np.isfinite(z_vals))
    assert np.all(np.isclose(d_vals, 5.0, atol=1e-6))   # constant depth used


def test_real_frame_zero_depth_yields_no_pairs():
    # A zero depth map is correctly filtered (V10 requires d > 0), so a
    # frame gives cleanly-empty pairs -- no NaN, no nonsense.
    depth_map = np.zeros((cfg.IMAGE_H, cfg.IMAGE_W), np.float32)
    imgs = cio.load_colmap_images(IMAGES)
    pts = cio.load_colmap_points(POINTS)
    d_vals, z_vals = cal.calibrate_frame(
        imgs[27], pts, depth_map, cfg.IMAGE_W, cfg.IMAGE_H)
    assert len(d_vals) == 0 and len(z_vals) == 0


# ------------------------------------------------------------
# F. Degenerate / invalid cases fail cleanly (no silent NaN)
# ------------------------------------------------------------
def test_robust_fit_rejects_too_few_samples():
    assert cal.robust_fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None


def test_robust_fit_rejects_all_nonfinite():
    bad = np.array([np.nan, np.inf, np.nan, -np.inf, np.nan, np.nan] * 12)
    good = np.linspace(0, 1, 72)
    assert cal.robust_fit(bad, good) is None     # non-finite -> filtered out


def test_fit_depth_calibration_rejects_too_few_pairs():
    assert cal.fit_depth_calibration(np.array([1.0]), np.array([2.0])) is None


def test_fit_depth_calibration_rejects_constant_depth():
    # Constant predicted depth across a frame is degenerate: corrcoef is
    # NaN for the linear fit. The gating helper must fail cleanly (return
    # None) rather than emit a NaN-corr "accepted" calibration.
    depth = np.full(N, 5.0)                    # zero variance
    cam_z = 2.0 * depth + 1.0
    res = cal.fit_depth_calibration(depth, cam_z, min_samples=50)
    assert res is None, "degenerate constant depth must be rejected, got %r" % res


def test_fit_depth_calibration_rejects_low_correlation():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, N)
    y = rng.normal(0, 1, N)                    # unrelated -> |corr| ~ 0
    res = cal.fit_depth_calibration(x, y, min_samples=50, min_corr=0.55)
    assert res is None                          # low-corr gate rejects


def test_calibrated_depth_inverse_guards_zero():
    frame = {"model": "inverse", "a": 10.0, "b": 1.0}
    out = cal.calibrated_depth(frame, np.array([0.0, 1e-9]))
    # zero/epsilon-guarded, not division-by-zero inf/nan
    assert np.all(np.isfinite(out))
    assert out[1] == 10.0 / 1e-6 + 1.0          # uses INVERSE_DEPTH_EPS


if __name__ == "__main__":
    run_module_tests(globals())