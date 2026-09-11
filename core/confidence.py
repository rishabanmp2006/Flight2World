"""flight2world.core.confidence

Per-point confidence / error layer extracted verbatim from confidence_v10.py
(the immutable V10 confidence baseline). The algorithm is NOT redesigned.

V10 confidence replays track-anchored voting on the FINAL clean cloud as a
GLOBAL (all fused frames, not ±3 window) prior-vs-truth test, without
re-running Depth Anything or candidate generation. Votes depend only on 3D
position, camera poses, and the triangulated track depth maps.

Outputs per point (scene units, ARBITRARY COLMAP scale -- NOT metric):
  confidence   Laplace-smoothed (agree+1)/(constrained+2)
  evidence     n_constrained = n_agree + n_contradict
  net_votes    n_agree - n_contradict
  residual     mean |zn - track_z| on agreeing views (-1 if none)
  worst_residual  max |zn - track_z| over contradicting views (-1 if none)
  dist_sparse  NN distance to sparse surface (scene units)

The PLY writer reproduces the 12-property ASCII header from
confidence_v10.py exactly (CloudCompare / MeshLab compatible).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from core.config import (
    ABSOLUTE_TOLERANCE,
    CX,
    CY,
    FX,
    FY,
    IMAGE_H,
    IMAGE_W,
    RELATIVE_TOLERANCE,
    TRACK_RADIUS_PX,
)


# ---------------------------------------------------------------------------
# Track maps helper (uses core.track_maps when available; falls back to
# local v10-faithful build to avoid circular import at module load)
# ---------------------------------------------------------------------------

def _build_track_maps_for_confidence(image_data, sparse_points,
                                      image_w=IMAGE_W, image_h=IMAGE_H):
    """Build z_grid / dist_grid for one frame, as confidence_v10 does.

    Faithful to confidence_v10.build_track_maps (which mirrors
    fusion_v10.build_track_maps). Delegates to core.track_maps to keep a
    single implementation.
    """
    from core.track_maps import build_track_maps as _btm
    z_grid, dist_grid, _n = _btm(
        image_data, sparse_points, image_w=image_w, image_h=image_h,
    )
    return z_grid, dist_grid


# ---------------------------------------------------------------------------
# Global voting + summary (confidence_v10.py lines 320-543)
# ---------------------------------------------------------------------------

def compute_confidence(
    P: np.ndarray,
    frames: list[dict],
    sparse_points: dict,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
) -> dict:
    """Compute per-point confidence and diagnostics for a clean cloud.

    Args:
        P: (N,3) float64 world points (e.g. from pcd_clean).
        frames: fused frame_records list, each with R, t, z_grid, dist_grid.
            If z_grid/dist_grid are missing they are built from COLMAP
            geometry via sparse_points.
        sparse_points: {point3d_id: XYZ} from COLMAP points3D.txt.

    Returns a dict with:
        N, confidence (N,), evidence (n_constrained), net_votes,
        residual (med_residual, NaN where none), worst_residual,
        dist_sparse, agree_ratio, coverage, plus scalar summary fields.
    """
    P = np.asarray(P, dtype=np.float64)
    N = int(len(P))
    if N == 0:
        return {
            "N": 0,
            "confidence": np.array([], dtype=np.float64),
            "evidence": np.array([], dtype=np.int32),
            "net_votes": np.array([], dtype=np.int32),
            "residual": np.array([], dtype=np.float64),
            "worst_residual": np.array([], dtype=np.float64),
            "dist_sparse": np.array([], dtype=np.float64),
            "agree_ratio": np.array([], dtype=np.float64),
            "summary": {
                "points": 0,
                "units": "ARBITRARY COLMAP scale -- NOT metric (no GPS source)",
            },
        }

    # Ensure track maps exist on every frame (confidence_v10 rebuilds them).
    for data in frames:
        if "z_grid" not in data or "dist_grid" not in data:
            z_grid, dist_grid = _build_track_maps_for_confidence(
                data, sparse_points, image_w=image_w, image_h=image_h,
            )
            data["z_grid"] = z_grid
            data["dist_grid"] = dist_grid

    n_agree = np.zeros(N, dtype=np.int16)
    n_contradict = np.zeros(N, dtype=np.int16)
    residual_sum = np.zeros(N, dtype=np.float64)
    residual_n = np.zeros(N, dtype=np.int16)
    worst_resid = np.full(N, -1.0, dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int16)

    # Global vote: every frame scores every point (strict, unlike fusion's ±3).
    for data in frames:
        R, t = data["R"], data["t"]
        z_grid = data["z_grid"]
        dist_grid = data["dist_grid"]

        cam = (R @ P.T).T + t  # N x 3
        zn = cam[:, 2]
        inside = zn > 0.15
        if not np.any(inside):
            continue

        # Reproject inside points
        un = FX * cam[inside, 0] / np.maximum(zn[inside], 1e-8) + CX
        vn = FY * cam[inside, 1] / np.maximum(zn[inside], 1e-8) + CY
        ui = np.rint(un).astype(np.int32)
        vi = np.rint(vn).astype(np.int32)
        in_img = (ui >= 0) & (ui < image_w) & (vi >= 0) & (vi < image_h)
        ids_all = np.where(inside)[0]
        ids = ids_all[in_img]
        if len(ids) == 0:
            continue

        coverage[ids] += 1

        track_z = z_grid[vi[in_img], ui[in_img]]
        track_d = dist_grid[vi[in_img], ui[in_img]]
        zn_i = zn[ids]

        constrained = np.isfinite(track_z) & (track_d <= TRACK_RADIUS_PX)
        c_ids = ids[constrained]
        if len(c_ids) == 0:
            continue

        tz = track_z[constrained]
        tol = np.maximum(np.abs(tz) * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
        resid = np.abs(zn_i[constrained] - tz)
        agree = resid <= tol

        # Use np.add.at for global accumulation where indices may repeat across frames?
        # Here c_ids are unique per frame, but across frames duplicates are handled by sequential adds.
        # The per-frame agree/contradict counts are distinct, so direct indexing add is fine.
        n_agree[c_ids[agree]] += 1
        n_contradict[c_ids[~agree]] += 1

        good = c_ids[agree]
        if len(good):
            residual_sum[good] += resid[agree]
            residual_n[good] += 1

        w = c_ids[~agree]
        if len(w):
            np.maximum.at(worst_resid, w, resid[~agree])

    n_constrained = n_agree.astype(np.int32) + n_contradict.astype(np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        agree_ratio = np.where(
            n_constrained > 0,
            n_agree.astype(np.float64) / np.maximum(n_constrained, 1).astype(np.float64),
            0.0,
        )

    confidence = (n_agree.astype(np.float64) + 1.0) / (n_constrained.astype(np.float64) + 2.0)

    med_residual = np.where(
        residual_n > 0,
        residual_sum / np.maximum(residual_n, 1).astype(np.float64),
        np.nan,
    )

    # dist_sparse via Open3D KDTree
    sparse_xyz = np.array(list(sparse_points.values()), dtype=np.float64) if sparse_points else np.empty((0, 3))
    if len(sparse_xyz) > 0:
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(sparse_xyz)
        sp_tree = o3d.geometry.KDTreeFlann(sp_pcd)
        dist_sparse = np.zeros(N, dtype=np.float64)
        for i in range(N):
            cnt, _, d = sp_tree.search_knn_vector_3d(P[i], 1)
            if cnt >= 1 and np.isfinite(d[0]):
                dist_sparse[i] = math.sqrt(d[0])
            else:
                dist_sparse[i] = np.nan
    else:
        dist_sparse = np.full(N, np.nan, dtype=np.float64)

    net = (n_agree.astype(np.int32) - n_contradict.astype(np.int32))

    # Summary helpers
    def _q(a, p):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if len(a) else float("nan")

    # Correlations (guard against constant arrays)
    def _corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 2:
            return float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return float("nan")

    summary = {
        "points": int(N),
        "frames_replayed": int(len(frames)),
        "units": "ARBITRARY COLMAP scale -- NOT metric (no GPS source)",
        "confidence": {
            "definition": "Laplace-smoothed agreement ratio "
                          "(agree+1)/(agree+contradict+2) against "
                          "triangulated COLMAP track depth over all "
                          "fused frames",
            "mean": float(np.nanmean(confidence)) if N else float("nan"),
            "median": _q(confidence, 50),
            "p10": _q(confidence, 10),
            "p90": _q(confidence, 90),
        },
        "evidence": {
            "definition": "number of fused frames whose track map "
                          "constrained this pixel",
            "p50": _q(n_constrained, 50),
            "p90": _q(n_constrained, 90),
            "zero_evidence_points": int(np.sum(n_constrained == 0)),
        },
        "agree_ratio_raw": {
            "median": _q(agree_ratio, 50),
            "p10": _q(agree_ratio, 10),
        },
        "net_votes": {
            "p10": _q(net, 10),
            "median": _q(net, 50),
        },
        "residual_error": {
            "definition": "mean |candidate Z - track Z| on agreeing "
                          "views, scene units (arbitrary)",
            "median": _q(med_residual, 50),
            "p90": _q(med_residual, 90),
        },
        "worst_residual": {
            "definition": "largest |Z - track Z| over all constraining views "
                          "(scene units, arbitrary); -1 if no view contradicts",
            "median": _q(worst_resid, 50),
            "p90": _q(worst_resid, 90),
            "fraction_with_contradiction": float(np.mean(worst_resid >= 0)) if N else 0.0,
        },
        "dist_sparse": {
            "definition": "NN distance to triangulated sparse "
                          "surface, scene units (arbitrary)",
            "median": _q(dist_sparse, 50),
            "p90": _q(dist_sparse, 90),
        },
        "corr_confidence_dist_sparse": _corr(
            confidence[np.isfinite(dist_sparse)] if N else np.array([]),
            dist_sparse[np.isfinite(dist_sparse)] if N else np.array([]),
        ),
        "corr_confidence_evidence": _corr(
            confidence[n_constrained > 0] if N and np.any(n_constrained > 0) else np.array([]),
            n_constrained[n_constrained > 0].astype(np.float64) if N and np.any(n_constrained > 0) else np.array([]),
        ),
        "low_conf_count": int(np.sum(confidence < 0.5)) if N else 0,
        "low_conf_frac": float(np.mean(confidence < 0.5)) if N else 0.0,
    }

    return {
        "N": N,
        "confidence": confidence,
        "evidence": n_constrained,
        "net_votes": net,
        "residual": med_residual,
        "worst_residual": worst_resid,
        "dist_sparse": dist_sparse,
        "agree_ratio": agree_ratio,
        "coverage": coverage,
        "n_agree": n_agree,
        "n_contradict": n_contradict,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Confidence PLY writer (confidence_v10.py lines 427-461)
# ---------------------------------------------------------------------------

def write_confidence_ply(
    path: str | Path,
    P: np.ndarray,
    C: Optional[np.ndarray],
    confidence: np.ndarray,
    evidence: np.ndarray,
    net_votes: np.ndarray,
    residual: np.ndarray,
    worst_residual: np.ndarray,
    dist_sparse: np.ndarray,
) -> Path:
    """Write an ASCII PLY with 6 scalar confidence fields, as V10.

    Header order and -1 fill for NaN residuals match confidence_v10.py
    exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    P = np.asarray(P, dtype=np.float64)
    N = int(len(P))
    if C is None:
        C = np.zeros((N, 3), dtype=np.uint8)
    else:
        C = (np.asarray(C) * 1.0)
        # If colors were float [0,1], convert to 0-255 uint8 as V10 expects
        if C.dtype != np.uint8:
            # Heuristic: if max <= 1.0+eps, scale to 255
            if np.nanmax(C) <= 1.01:
                C = (np.clip(C, 0, 1) * 255.0).astype(np.uint8)
            else:
                C = np.clip(C, 0, 255).astype(np.uint8)

    resid_fill = np.where(np.isfinite(residual), residual, -1.0)
    worst_fill = np.where(np.asarray(worst_residual) >= 0, worst_residual, -1.0)
    dist_fill = np.where(np.isfinite(dist_sparse), dist_sparse, -1.0)

    lines = [
        "ply",
        "format ascii 1.0",
        "comment FLIGHT2WORLD V10 confidence layer",
        "element vertex %d" % N,
        "property double x",
        "property double y",
        "property double z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
        "property int evidence",
        "property int net_votes",
        "property float residual",
        "property float worst_residual",
        "property float dist_sparse",
        "end_header",
    ]

    rows = []
    for i in range(N):
        rows.append("%.6f %.6f %.6f %d %d %d %.4f %d %d %.5f %.5f %.5f" % (
            float(P[i, 0]), float(P[i, 1]), float(P[i, 2]),
            int(C[i, 0]), int(C[i, 1]), int(C[i, 2]),
            float(confidence[i]),
            int(evidence[i]),
            int(net_votes[i]),
            float(resid_fill[i]),
            float(worst_fill[i]),
            float(dist_fill[i]),
        ))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if rows:
            f.write("\n".join(rows) + "\n")

    return path


def write_confidence_json(path: str | Path, summary: dict) -> Path:
    """Write the confidence summary JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path
