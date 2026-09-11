"""flight2world.core.gate

V10 adaptive COLMAP spatial gate, extracted faithfully from
`fusion_v10.py` (sparse KD-tree + local density, lines 468-489;
per-candidate acceptance, lines 709-721). The gate is NOT widened or
narrowed, and the algorithm is NOT redesigned.

The gate is a pure spatial ball whose radius adapts to local sparse
density:

    local_r[i] = distance from sparse point i to its 8th nearest
                 sparse neighbour         (fusion_v10.py lines 476-484)
    radius(candidate) = clip(local_r[nearest] * GATE_FACTOR,
                             GATE_RADIUS_MIN, GATE_RADIUS_MAX)
    accept if dist(candidate, nearest sparse point) <= radius

COLMAP coordinates are arbitrary scene units; metric scale is unknown
and nothing is reinterpreted as metres.
"""

import math

import numpy as np
import open3d as o3d

from core.config import (
    GATE_FACTOR,
    GATE_KNN_K,
    GATE_LOCAL_R_DEFAULT,
    GATE_RADIUS_MAX,
    GATE_RADIUS_MIN,
)


def build_sparse_kdtree(sparse_xyz):
    """Build the V10 sparse KD-tree over COLMAP points.

    Faithful to fusion_v10.py lines 470-474.
    """
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_xyz)
    return o3d.geometry.KDTreeFlann(sparse_pcd)


def compute_local_radius(sparse_xyz, k=GATE_KNN_K,
                         fallback=GATE_LOCAL_R_DEFAULT):
    """Per-sparse-point local radius: 8th-nearest-neighbour distance.

    Faithful to fusion_v10.py lines 476-484. ``k`` defaults to V10's
    hardcoded 8 (behavior identical at defaults); a sparse point with
    fewer than 2 neighbours gets ``fallback`` (V10's 0.02).

    Returns a float64 array of length len(sparse_xyz).
    """
    sparse_tree = build_sparse_kdtree(sparse_xyz)
    local_r = np.zeros(len(sparse_xyz), dtype=np.float64)
    for i in range(len(sparse_xyz)):
        cnt, _, d = sparse_tree.search_knn_vector_3d(sparse_xyz[i], k)
        if cnt >= k:
            local_r[i] = math.sqrt(d[k - 1])
        elif cnt >= 2:
            local_r[i] = math.sqrt(d[-1])
        else:
            local_r[i] = fallback
    return local_r


def gate_candidates(world_pts, sparse_tree, local_r,
                    gate_factor=GATE_FACTOR,
                    radius_min=GATE_RADIUS_MIN,
                    radius_max=GATE_RADIUS_MAX):
    """Per-candidate adaptive-gate acceptance (fusion_v10.py lines 709-721).

    Returns a boolean array, True where the candidate's nearest sparse
    point lies within the local radius. The V10 per-point Python loop is
    preserved exactly -- the gate is not widened, narrowed, or otherwise
    "improved".

    Degenerate inputs fail safely: an empty candidate set yields an empty
    boolean array, and a candidate with no sparse neighbour (cnt < 1) is
    simply not accepted.
    """
    gate = np.zeros(len(world_pts), dtype=bool)
    for i in range(len(world_pts)):
        cnt, idx, d = sparse_tree.search_knn_vector_3d(world_pts[i], 1)
        if cnt < 1:
            continue
        nn_dist = math.sqrt(d[0])
        radius = float(np.clip(
            local_r[idx[0]] * gate_factor,
            radius_min, radius_max,
        ))
        if np.isfinite(nn_dist) and nn_dist <= radius:
            gate[i] = True
    return gate
