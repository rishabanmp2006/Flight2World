"""flight2world.core.cleanup

V10 cleanup stages extracted verbatim from fusion_v10.py (the immutable
baseline). The algorithm is NOT redesigned or retuned.

Two stages, in order:

  1. Raw cleanup (fusion_v10.py lines 835-868):
       vstack → PointCloud → voxel_down_sample(0.04) →
       remove_statistical_outlier(30, 1.5) guarded by len>100.

  2. DBSCAN clean stage (fusion_v10.py lines 912-984, mirrors clean_v7.py):
       adaptive vox = clip(largest_dim*0.003, 0.005, 0.10)
       down = voxel_down_sample(vox)
       DBSCAN eps=vox*3.5, min_points=8
       keep main cluster + secondaries where
         count >= max(30, 0.001*len(main)) and rel_dist < 0.35
       then statistical outlier 20/2.5 guarded by len>100.

All thresholds come from core.config; defaults reproduce V10 exactly.
Metric scale is unknown (COLMAP arbitrary units).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from core.config import (
    STAT_NB_NEIGHBORS,
    STAT_STD_RATIO,
    VOXEL_SIZE,
)


# ---------------------------------------------------------------------------
# Stage 1: voxel + statistical outlier (V10 lines 854-865)
# ---------------------------------------------------------------------------

def voxel_and_statistical(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = VOXEL_SIZE,
    nb_neighbors: int = STAT_NB_NEIGHBORS,
    std_ratio: float = STAT_STD_RATIO,
) -> o3d.geometry.PointCloud:
    """Voxel downsample + statistical outlier removal, as V10.

    V10 does:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO)

    No other filtering is performed.
    """
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    return pcd


def make_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Build an Open3D PointCloud from world points + optional RGB.

    ``colors`` are expected in [0, 1] float64 (as V10's sample_rgb produces).
    If None or empty, the cloud is created without colors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    if colors is not None and len(colors) and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64)
        )
    return pcd


# ---------------------------------------------------------------------------
# Stage 2: DBSCAN main-cluster retention (V10 lines 922-984)
# ---------------------------------------------------------------------------

def dbscan_clean(
    pcd: o3d.geometry.PointCloud,
    voxel_factor: float = 0.003,
    voxel_min: float = 0.005,
    voxel_max: float = 0.10,
    eps_factor: float = 3.5,
    min_points: int = 8,
    secondary_min_count: int = 30,
    secondary_frac: float = 0.001,
    rel_dist_thresh: float = 0.35,
    stat_nb2: int = 20,
    stat_ratio2: float = 2.5,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """DBSCAN main-cluster retention, faithful to fusion_v10.py 922-984.

    Returns (clean_pcd, diagnostics dict). On failure (no valid clusters or
    exception) the diagnostics record the reason and the original/downsampled
    cloud is returned — matching V10's "keeping raw cloud" / exception-swallow
    behavior.
    """
    if len(pcd.points) == 0:
        return pcd, {
            "vox": 0.0,
            "eps": 0.0,
            "n_clusters": 0,
            "status": "empty_input",
        }

    try:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        largest_dim = float(np.max(extent)) if extent.size else 0.0

        # V10 line 927: vox = max(0.005, min(0.10, largest_dim * 0.003))
        vox = float(max(voxel_min, min(voxel_max, largest_dim * voxel_factor)))

        down = pcd.voxel_down_sample(vox)

        if len(down.points) == 0:
            return pcd, {
                "vox": vox,
                "eps": vox * eps_factor,
                "n_clusters": 0,
                "status": "downsample_empty",
                "largest_dim": largest_dim,
            }

        eps = vox * eps_factor
        labels = np.array(
            down.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
        )

        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            # V10: "No valid clusters; keeping raw cloud."
            return down, {
                "vox": vox,
                "eps": eps,
                "n_clusters": 0,
                "status": "no_valid_clusters",
                "largest_dim": largest_dim,
            }

        unique, counts = np.unique(valid_labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        main_label = int(unique[order[0]])
        main_mask = labels == main_label
        main_pts = np.asarray(down.points)[main_mask]
        main_center = np.mean(main_pts, axis=0)

        keep_labels = [main_label]
        for label, count in zip(unique, counts):
            if int(label) == main_label:
                continue
            if int(count) < max(secondary_min_count, int(len(main_pts) * secondary_frac)):
                continue
            center = np.mean(np.asarray(down.points)[labels == label], axis=0)
            rel_dist = float(np.linalg.norm(center - main_center) / largest_dim) if largest_dim > 1e-12 else 0.0
            if rel_dist < rel_dist_thresh:
                keep_labels.append(int(label))

        keep_mask = np.isin(labels, keep_labels)
        clean_pts = np.asarray(down.points)[keep_mask]
        clean_cols = None
        if down.has_colors():
            clean_cols = np.asarray(down.colors)[keep_mask]

        clean = o3d.geometry.PointCloud()
        clean.points = o3d.utility.Vector3dVector(clean_pts)
        if clean_cols is not None and len(clean_cols):
            clean.colors = o3d.utility.Vector3dVector(clean_cols)

        # V10 lines 970-972: second statistical pass
        if len(clean.points) > 100:
            clean, _ = clean.remove_statistical_outlier(
                nb_neighbors=stat_nb2,
                std_ratio=stat_ratio2,
            )

        return clean, {
            "vox": vox,
            "eps": eps,
            "n_clusters": int(len(unique)),
            "main_label": main_label,
            "main_points": int(len(main_pts)),
            "kept_labels": keep_labels,
            "largest_dim": largest_dim,
            "status": "ok",
        }

    except Exception as exc:  # V10 swallows and keeps raw
        return pcd, {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def run_cleanup(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, dict]:
    """Convenience: raw (voxel+stat) then DBSCAN clean.

    Raises RuntimeError if no points survived fusion (as V10 lines 842-843).

    Returns (pcd_raw, pcd_clean, diagnostics) where diagnostics has
    ``raw_points``, ``clean_points``, and the DBSCAN detail dict.
    """
    if points is None or len(points) == 0:
        raise RuntimeError("No points survived track-anchored fusion.")

    points = np.asarray(points, dtype=np.float64)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)

    pcd_raw = make_point_cloud(points, colors)
    pcd_raw = voxel_and_statistical(pcd_raw)

    # DBSCAN stage: if it fails or finds no clusters, pcd_clean is the
    # downsampled cloud (V10 keeps raw); diagnostics record the status.
    pcd_clean, dbscan_diag = dbscan_clean(pcd_raw)

    diagnostics = {
        "raw_points": int(len(pcd_raw.points)),
        "clean_points": int(len(pcd_clean.points)),
        "voxel_size": float(VOXEL_SIZE),
        "stat_nb": int(STAT_NB_NEIGHBORS),
        "stat_ratio": float(STAT_STD_RATIO),
        "dbscan": dbscan_diag,
    }
    return pcd_raw, pcd_clean, diagnostics
