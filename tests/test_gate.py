"""tests.test_gate

Deterministic tests for core.gate (V10 adaptive COLMAP spatial gate).
Covers sparse KD-tree construction, deterministic local sparse density,
per-candidate local radius, the GATE_FACTOR / radius clamps, candidate
acceptance, and safe behavior on empty/degenerate inputs -- plus
head-to-head numerical comparison against the exact V10 gate and
local-radius code compiled verbatim from fusion_v10.py (requirement D).
"""

import math
import pathlib
import sys

# Make the project root importable regardless of how this file is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import run_module_tests

import core.config as cfg
import core.gate as gate
from tests import v10_reference

# V10 reference code compiled verbatim from fusion_v10.py:
#   lines 476-484  -> local_r computation (needs sparse_tree built first)
#   lines 709-720  -> per-candidate gate acceptance loop
V10_LOCAL_R = v10_reference.v10_block(
    "v10_local_r", 476, 484, ["sparse_xyz", "sparse_tree"],
    returns="local_r")
V10_GATE = v10_reference.v10_block(
    "v10_gate", 709, 720, ["world_pts", "sparse_tree", "local_r"],
    returns="gate")


def _tree(xyz):
    return gate.build_sparse_kdtree(np.asarray(xyz, dtype=np.float64))


# ------------------------------------------------------------
# Local radius (sparse density)
# ------------------------------------------------------------
def test_single_sparse_point_uses_fallback():
    local_r = gate.compute_local_radius(np.array([[0.0, 0.0, 0.0]]))
    assert local_r.shape == (1,)
    assert local_r[0] == 0.02                      # V10 fallback (fewer than 2)


def test_two_points_radius_is_pairwise_distance():
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    local_r = gate.compute_local_radius(xyz)
    assert local_r[0] == 10.0 and local_r[1] == 10.0   # sqrt(d[-1]) branch


def test_eight_plus_points_uses_8th_nearest():
    # 3x3 grid at spacing 0.1 on the z=0 plane. KDTreeFlann's k=8 query
    # counts the query point ITSELF (distance 0) among the 8, so the
    # 8th value is the 7th distinct neighbour: for the corner (0,0,0)
    # that is a distance-sqrt(0.05) diagonal-2 step (0.1, 0.1) -- not the
    # far corner sqrt(0.08). For the center (0.1,0.1) it is a corner
    # -> sqrt(0.02). (V10 behavior; verified equal in
    # test_v10_local_radius_matches_exactly.)
    xs = [0.0, 0.1, 0.2]
    xyz = np.array([[x, y, 0.0] for x in xs for y in xs])
    local_r = gate.compute_local_radius(xyz)
    assert len(local_r) == 9
    assert math.isclose(local_r[0], math.sqrt(0.05), rel_tol=1e-12)   # corner
    assert math.isclose(local_r[4], math.sqrt(0.02), rel_tol=1e-12)   # center


def test_local_radius_is_deterministic():
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-2, 2, (300, 3))
    a = gate.compute_local_radius(xyz)
    b = gate.compute_local_radius(xyz)
    assert np.array_equal(a, b)


# ------------------------------------------------------------
# Candidate acceptance with clamps
# ------------------------------------------------------------
def test_candidate_within_radius_passes():
    # Single sparse point: local_r = 0.02, radius = clip(0.04) = 0.04.
    xyz = np.array([[0.0, 0.0, 0.0]])
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    inside = np.array([[0.01, 0.0, 0.0]])       # dist 0.01 <= 0.04
    outside = np.array([[0.1, 0.0, 0.0]])       # dist 0.1  >  0.04
    g = gate.gate_candidates(np.vstack([inside, outside]), tree, local_r)
    assert g.tolist() == [True, False]


def test_radius_is_clamped_by_minimum():
    # A very dense sparse line (local_r ~ 0.004-0.007) gives radius
    # clip(~0.012, 0.015, 0.12) = 0.015 -- a candidate at 0.01 (dist 0.003
    # from the last sparse point) passes, one at 0.03 (dist 0.023) fails.
    pts = []
    for i in range(8):
        pts.append([i * 0.001, 0.0, 0.0])
    xyz = np.array(pts)
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    assert local_r.min() <= 0.01
    g = gate.gate_candidates(
        np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]]),
        tree, local_r)
    assert g.tolist() == [True, False]


def test_radius_is_clamped_by_maximum():
    # Two sparse points 10 apart: local_r = 10, radius = clip(20) = 0.12.
    # A candidate 0.1 from a sparse point passes, one 0.2 away fails.
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    assert local_r[0] == 10.0
    g = gate.gate_candidates(
        np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        tree, local_r)
    assert g.tolist() == [True, False]


def test_gate_respects_custom_clamps():
    # When both sparse points are far apart the un-clamped radius would
    # be 20; passing radius_max=0.5 must bound it and change acceptance.
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    g = gate.gate_candidates(
        np.array([[0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        tree, local_r, radius_max=0.5)
    assert g.tolist() == [True, False]


# ------------------------------------------------------------
# Empty / degenerate inputs fail safely
# ------------------------------------------------------------
def test_empty_candidates_give_empty_gate():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    g = gate.gate_candidates(np.empty((0, 3)), tree, local_r)
    assert g.shape == (0,)
    assert g.dtype == np.bool_


def test_empty_sparse_cloud_gives_empty_local_r():
    local_r = gate.compute_local_radius(np.empty((0, 3)))
    assert local_r.shape == (0,)


def test_candidates_with_no_sparse_neighbour_never_accepted():
    # A tree with a single far-away point: a candidate far beyond any
    # radius is still queried (cnt >= 1) and correctly rejected.
    xyz = np.array([[0.0, 0.0, 0.0]])
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    g = gate.gate_candidates(np.array([[50.0, 0.0, 0.0]]), tree, local_r)
    assert g.tolist() == [False]


# ------------------------------------------------------------
# D. V10 reference: identical inputs, numerical equality
# ------------------------------------------------------------
def test_v10_local_radius_matches_exactly():
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-3, 3, (400, 3))
    tree = _tree(xyz)
    ours = gate.compute_local_radius(xyz)
    ref = V10_LOCAL_R(xyz, tree)
    assert np.array_equal(ours, ref), "local_r differs from V10"


def test_v10_gate_matches_exactly():
    rng = np.random.default_rng(13)
    xyz = rng.uniform(-3, 3, (500, 3))
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    world = rng.uniform(-4, 4, (2000, 3))
    ours = gate.gate_candidates(world, tree, local_r)
    ref = V10_GATE(world, tree, local_r)
    assert np.array_equal(ours, ref), "gate differs from V10"


def test_v10_gate_real_sparse_cloud():
    import core.colmap_io as cio
    from tests import SPARSE_TXT
    pts = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
    xyz = np.array(list(pts.values()), dtype=np.float64)
    tree = _tree(xyz)
    local_r = gate.compute_local_radius(xyz)
    # A sample of the cloud itself should mostly pass; far-away probes fail.
    probes = np.vstack([xyz[:300], xyz[:300] + 5.0])
    ours = gate.gate_candidates(probes, tree, local_r)
    ref = V10_GATE(probes, tree, local_r)
    assert np.array_equal(ours, ref)
    assert bool(ours[:300].any())              # cloud points pass sometimes
    assert not bool(ours[300:].any())          # +5.0 probes fail


if __name__ == "__main__":
    run_module_tests(globals())
