"""tests.test_track_maps

Deterministic tests for core.track_maps. Covers the V10 track-map
construction: track rasterization, iterative dilation, nearest-track
depth propagation, the distance map, track count, exact 1280x720 output
dimensions, and no coordinate inversion -- plus a head-to-head numerical
comparison against the exact V10 ``build_track_maps`` compiled verbatim
from fusion_v10.py (requirement D).
"""

import math
import pathlib
import sys

# Make the project root importable regardless of how this file is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import SPARSE_TXT, run_module_tests

import core.colmap_io as cio
import core.config as cfg
import core.track_maps as tm
from tests import v10_reference

IMAGES = SPARSE_TXT / "images.txt"
POINTS = SPARSE_TXT / "points3D.txt"

V10_BUILD = v10_reference.v10_function("build_track_maps")


# ------------------------------------------------------------
# Rasterization: known observations appear in the correct pixels
# ------------------------------------------------------------
def _frame(observations, R=None, t=None):
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return {"id": 1, "name": "synth.jpg", "R": R, "t": t,
            "observations": observations}


def test_known_observation_lands_on_correct_pixel():
    # One track at world (2, 1, 5), identity pose -> Z_cam = 5.
    obs = [(100.0, 80.0, 1)]
    sparse = {1: np.array([2.0, 1.0, 5.0])}
    z_grid, dist_grid, n = tm.build_track_maps(_frame(obs), sparse)
    assert z_grid.shape == (cfg.IMAGE_H, cfg.IMAGE_W)
    # row = round(80) = 80, col = round(100) = 100
    assert z_grid[80, 100] == 5.0
    assert dist_grid[80, 100] == 0        # real track pixel
    assert n == 1


def test_observation_rounding_uses_row_major_indexing():
    # No coordinate inversion: (x=100.4, y=80.6) must index [81, 100]
    # (row=y, col=x), NOT [100, 81]. The exact track pixel has dist 0;
    # the swapped index is only ever REACHED by dilation (dist > 0).
    # Z_cam = 4.
    obs = [(100.4, 80.6, 1)]
    sparse = {1: np.array([0.0, 0.0, 4.0])}
    z_grid, dist_grid, n = tm.build_track_maps(_frame(obs), sparse)
    assert z_grid[81, 100] == 4.0
    assert dist_grid[81, 100] == 0        # the real track pixel (row=y first)
    assert dist_grid[100, 81] > 0         # only filled by dilation, not the track
    assert n == 1


def test_behind_camera_track_skipped():
    obs = [(100.0, 80.0, 1)]
    sparse = {1: np.array([0.0, 0.0, -1.0])}   # Z_cam < 0
    z_grid, _dist, n = tm.build_track_maps(_frame(obs), sparse)
    assert n == 0
    assert np.isnan(z_grid[80, 100])


def test_out_of_image_track_skipped():
    obs = [(2000.0, 5000.0, 1)]                # beyond 1280x720
    sparse = {1: np.array([0.0, 0.0, 5.0])}
    z_grid, _dist, n = tm.build_track_maps(_frame(obs), sparse)
    assert n == 0


# ------------------------------------------------------------
# Dilation: deterministic, mean-of-valid-neighbours propagation
# ------------------------------------------------------------
def test_dilation_is_deterministic():
    rng = np.random.default_rng(7)
    obs = []
    sparse = {}
    for pid in range(1, 121):
        x = float(rng.uniform(20, 1200))
        y = float(rng.uniform(20, 680))
        obs.append((x, y, pid))
        sparse[pid] = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3),
                                5.0 + rng.uniform(-0.5, 0.5)])
    frame = _frame(obs)
    a = tm.build_track_maps(frame, sparse)
    b = tm.build_track_maps(frame, sparse)
    assert np.array_equal(a[0], b[0], equal_nan=True)   # bit-exact float32 (NaN holes equal)
    assert np.array_equal(a[1], b[1], equal_nan=True)   # bit-exact int32
    assert a[2] == b[2]


def test_isolated_track_fills_neighbourhood_with_own_depth():
    # A single track at (50, 50) with Z=3: step-1 dilation fills its
    # 8-neighbours with the mean of the one valid neighbour -> exactly 3.
    obs = [(50.0, 50.0, 1)]
    sparse = {1: np.array([0.0, 0.0, 3.0])}
    z_grid, dist_grid, _n = tm.build_track_maps(_frame(obs), sparse)
    assert z_grid[50, 50] == 3.0
    assert z_grid[49, 50] == 3.0               # 8-neighbour
    assert z_grid[49, 49] == 3.0               # 8-neighbour
    assert dist_grid[49, 50] == 1              # filled at step 1
    assert dist_grid[50, 50] == 0              # real track pixel


def test_pixel_between_two_tracks_gets_mean_depth():
    # Tracks at (50, 40) Z=3 and (50, 44) Z=7. Pixel (50, 42) is adjacent
    # to both, so step-1 fill = mean(3, 7) = 5 (V10's sum/count rule).
    obs = [(40.0, 50.0, 1), (44.0, 50.0, 2)]
    sparse = {1: np.array([0.0, 0.0, 3.0]),
              2: np.array([0.0, 0.0, 7.0])}
    z_grid, _dist, _n = tm.build_track_maps(_frame(obs), sparse)
    assert z_grid[50, 42] == 5.0


def test_unreached_pixels_stay_nan():
    # A track at the far corner leaves most of the image unreached within
    # max_steps; those pixels must remain NaN (never garbage / inverted).
    obs = [(1279.0, 719.0, 1)]
    sparse = {1: np.array([0.0, 0.0, 6.0])}
    z_grid, _dist, _n = tm.build_track_maps(_frame(obs), sparse)
    assert np.isnan(z_grid[0, 0])               # unreached corner stays NaN
    assert not np.any(np.isinf(z_grid))         # never inf / garbage
    assert np.isnan(z_grid).any() and np.isfinite(z_grid).any()
    assert z_grid[719, 1279] == 6.0


def test_output_dimensions_exact():
    obs = [(100.0, 80.0, 1)]
    sparse = {1: np.array([2.0, 1.0, 5.0])}
    z_grid, dist_grid, _n = tm.build_track_maps(_frame(obs), sparse)
    assert z_grid.shape == (720, 1280)
    assert dist_grid.shape == (720, 1280)
    assert z_grid.dtype == np.float32
    assert dist_grid.dtype == np.int32
    assert np.all(np.isfinite(dist_grid))


# ------------------------------------------------------------
# D. V10 reference: identical inputs, numerical equality
# ------------------------------------------------------------
def test_v10_head_to_head_synthetic():
    rng = np.random.default_rng(11)
    obs, sparse = [], {}
    for pid in range(1, 91):
        x, y = float(rng.uniform(10, 1270)), float(rng.uniform(10, 710))
        obs.append((x, y, pid))
        sparse[pid] = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3),
                                rng.uniform(1, 12)])
    frame = _frame(obs)
    z10, d10, n10 = V10_BUILD(frame, sparse)
    z, d, n = tm.build_track_maps(frame, sparse)
    # equal_nan: unreached pixels are NaN in both (NaN != NaN otherwise).
    assert np.array_equal(z, z10, equal_nan=True), "z_grid differs from V10"
    assert np.array_equal(d, d10, equal_nan=True), "dist_grid differs from V10"
    assert n == n10


def test_v10_head_to_head_real_frame_0027():
    # Real COLMAP frame_0027 + the full sparse cloud: the strongest
    # deterministic check available (no depth model involved).
    imgs = cio.load_colmap_images(IMAGES)
    pts = cio.load_colmap_points(POINTS)
    z10, d10, n10 = V10_BUILD(imgs[27], pts)
    z, d, n = tm.build_track_maps(imgs[27], pts)
    assert np.array_equal(z, z10, equal_nan=True), "z_grid differs from V10 on frame_0027"
    assert np.array_equal(d, d10, equal_nan=True), "dist_grid differs from V10 on frame_0027"
    assert n == n10
    # Sanity: a real frame rasterizes thousands of tracks.
    assert n > 500


if __name__ == "__main__":
    run_module_tests(globals())
