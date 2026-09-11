"""flight2world.core.export

Thin PLY and diagnostics I/O helpers.

No reconstruction mathematics lives here — only filesystem writes that the
pipeline stages call after the mathematics have completed. V10 contracts are
preserved:

  * Colors are [0, 1] float64 (as V10's sample_rgb produces); conversion to
    Open3D's Vector3dVector is handled here.
  * Binary PLY via Open3D is used for dense clouds (fused_raw/clean).
  * ASCII PLY with 6 scalar fields is used only for the confidence layer
    (see core.confidence.write_confidence_ply for that header).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> Path:
    """Write an Open3D point cloud to *path* (PLY via o3d.io).

    ``colors`` when given are expected in [0, 1] float64 (V10 convention).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

    ok = o3d.io.write_point_cloud(str(path), pcd)
    if not ok:
        raise RuntimeError(f"failed to write PLY: {path}")
    return path


def write_point_cloud_ply(
    path: str | Path,
    pcd: o3d.geometry.PointCloud,
) -> Path:
    """Write an existing Open3D PointCloud to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd)
    if not ok:
        raise RuntimeError(f"failed to write PLY: {path}")
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    """Write *payload* as pretty-printed JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
