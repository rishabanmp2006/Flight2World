"""flight2world.core.fusion

V10 pass-2 track-anchored dense fusion, extracted faithfully from
`fusion_v10.py` (lines 656-825). The algorithm, thresholds, tolerances,
frame ordering, coordinate conventions, depth convention and voting
semantics are preserved exactly; the inline main-loop body is factored
into pure helpers and a ``fuse_frame`` entry point, with ``fuse_all``
reproducing the V10 aggregation loop behavior-for-behavior.

Everything that was inline in the V10 pass-2 loop is here:

    * candidate pixel generation   (grid at PIXEL_STRIDE, finite-depth filter)
    * calibrated depth             (Z = a*d + b or a/d + b; core.calibration)
    * pinhole backprojection       (u,v,z -> camera coordinates)
    * camera -> world transform    (COLMAP convention, as V10)
    * adaptive COLMAP spatial gate (core.gate)
    * track-anchored reprojection consistency voting against each
      neighbour's TRIANGULATED track map (agree / contradict / neutral)
    * net-vote thresholding and RGB sampling

COLMAP units are arbitrary scene units (metric scale unknown); nothing
is converted to metres.

Frame ordering and selection (V10 greedy minimum-baseline curation and
calibration/RMSE gating, lines 431-461 and 623-632) are pipeline-level
stages that happen BEFORE the pass-2 loop in V10. This module consumes
the curated, calibrated ``frame_records`` sequence exactly as V10's loop
does -- it neither reorders nor re-selects frames. The curation stage
itself is extracted in a later phase.
"""

import numpy as np

from core.calibration import calibrated_depth
from core.config import (
    ABSOLUTE_TOLERANCE,
    FUSE_REPROJECT_EPS,
    FUSE_VOTE_Z_MIN,
    FUSE_Z_MAX,
    FUSE_Z_MIN,
    IMAGE_H,
    IMAGE_W,
    MIN_NET_VOTES,
    NEIGHBOR_WINDOW,
    PIXEL_STRIDE,
    RELATIVE_TOLERANCE,
    TRACK_RADIUS_PX,
    CameraIntrinsics,
    PipelineConfig,
)
from core.gate import gate_candidates


def candidate_grid(pixel_stride, image_w=IMAGE_W, image_h=IMAGE_H):
    """V10 lines 676-680: meshgrid of sampled pixel coordinates."""
    ys = np.arange(0, image_h, pixel_stride)
    xs = np.arange(0, image_w, pixel_stride)
    u, v = np.meshgrid(xs, ys)
    u = u.reshape(-1)
    v = v.reshape(-1)
    return u, v


def sample_candidates(frame, pixel_stride, image_w=IMAGE_W, image_h=IMAGE_H,
                      z_min=FUSE_Z_MIN, z_max=FUSE_Z_MAX):
    """V10 lines 682-688: valid (u, v, calibrated-Z) candidate triples.

    Applies the V10 finite-depth filter (ai > 0, finite) and the
    calibrated-Z range gate (z_min < z < z_max).
    """
    depth = frame["depth"]
    u, v = candidate_grid(pixel_stride, image_w=image_w, image_h=image_h)

    ai = depth[v, u].astype(np.float64)
    valid = np.isfinite(ai) & (ai > 0)
    u, v, ai = u[valid], v[valid], ai[valid]

    z = calibrated_depth(frame, ai)
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    u, v, z = u[valid], v[valid], z[valid]
    return u, v, z


def backproject(u, v, z, intrinsics):
    """V10 lines 699-700: pinhole backprojection to camera coordinates.

        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
    """
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.column_stack([x, y, z])


def cam_to_world(cam_pts, R, t):
    """V10 line 702: camera -> world, COLMAP convention (X_cam = R X_world + t)."""
    return (R.T @ (cam_pts - t).T).T


def neighbor_range(frame_idx, n_frames, neighbor_window=NEIGHBOR_WINDOW):
    """V10 lines 742-743: ordered neighbour window for a frame."""
    start = max(0, frame_idx - neighbor_window)
    end = min(n_frames, frame_idx + neighbor_window + 1)
    return start, end


def track_anchored_votes(world_pts, frame_records, frame_idx,
                         neighbor_window=NEIGHBOR_WINDOW,
                         track_radius_px=TRACK_RADIUS_PX,
                         relative_tolerance=RELATIVE_TOLERANCE,
                         absolute_tolerance=ABSOLUTE_TOLERANCE,
                         vote_z_min=FUSE_VOTE_Z_MIN,
                         reproject_eps=FUSE_REPROJECT_EPS,
                         image_w=IMAGE_W, image_h=IMAGE_H,
                         intrinsics=None):
    """V10 lines 740-807: cross-view track-anchored reprojection voting.

    Each neighbour in the ordered window reprojects every candidate into
    itself and compares the candidate's Z against the neighbour's
    TRIANGULATED track depth (never its monocular depth): +1 when within
    tolerance of a nearby track, -1 when a track is nearby but disagrees,
    0 when no track constrains the pixel.

    Returns (net_votes int8 array, vote_delta dict) where vote_delta is
    {agree, contradict, neutral} counts for this frame (V10's
    ``vote_hist`` accumulation, per frame).
    """
    k = intrinsics if intrinsics is not None else CameraIntrinsics()
    fx, fy = k.fx, k.fy
    cx, cy = k.cx, k.cy

    net_votes = np.zeros(len(world_pts), dtype=np.int8)
    vote_delta = {"agree": 0, "contradict": 0, "neutral": 0}

    start, end = neighbor_range(frame_idx, len(frame_records),
                                neighbor_window=neighbor_window)
    for j in range(start, end):
        if j == frame_idx:
            continue

        nb = frame_records[j]
        Rn, tn = nb["R"], nb["t"]
        z_grid_n = nb["z_grid"]
        dist_grid_n = nb["dist_grid"]

        cam_n = (Rn @ world_pts.T).T + tn
        zn = cam_n[:, 2]

        ok_z = np.isfinite(zn) & (zn > vote_z_min)
        if not np.any(ok_z):
            continue

        un = fx * cam_n[:, 0] / np.maximum(zn, reproject_eps) + cx
        vn = fy * cam_n[:, 1] / np.maximum(zn, reproject_eps) + cy

        ui = np.rint(un).astype(np.int32)
        vi = np.rint(vn).astype(np.int32)

        inside = (
            ok_z &
            (ui >= 0) & (ui < image_w) &
            (vi >= 0) & (vi < image_h)
        )
        if not np.any(inside):
            continue

        ids = np.where(inside)[0]
        ui_i, vi_i = ui[ids], vi[ids]
        zn_i = zn[ids]

        track_z = z_grid_n[vi_i, ui_i]
        track_d = dist_grid_n[vi_i, ui_i]

        constrained = np.isfinite(track_z) & (track_d <= track_radius_px)

        agree = np.zeros(len(ids), dtype=bool)
        if np.any(constrained):
            tol = np.maximum(
                np.abs(track_z[constrained]) * relative_tolerance,
                absolute_tolerance,
            )
            agree[constrained] = (
                np.abs(zn_i[constrained] - track_z[constrained]) <= tol
            )

        # Vectorized vote accumulation over this neighbour.
        for sel_ids, delta in (
            (ids[agree], +1),
            (ids[constrained & ~agree], -1),
            (ids[~constrained], 0),
        ):
            if len(sel_ids):
                net_votes[sel_ids] += delta
                if delta > 0:
                    vote_delta["agree"] += len(sel_ids)
                elif delta < 0:
                    vote_delta["contradict"] += len(sel_ids)
                else:
                    vote_delta["neutral"] += len(sel_ids)

    return net_votes, vote_delta


def sample_rgb(rgb, v, u):
    """V10 line 818: colors = rgb[v, u] / 255.0 (float64, [0, 1])."""
    return rgb[v, u].astype(np.float64) / 255.0


def _empty_result(stats, u, v):
    return {
        "points": np.empty((0, 3), dtype=np.float64),
        "colors": np.empty((0, 3), dtype=np.float64),
        "stats": stats,
        "votes": {"agree": 0, "contradict": 0, "neutral": 0},
        "u": u,
        "v": v,
    }


def fuse_frame(frame, frame_records, frame_idx, sparse_tree, local_r,
               config=None):
    """Run the V10 pass-2 fusion for one frame (fusion_v10.py lines 662-825).

    ``frame_records`` is the curated, calibrated frame sequence exactly as
    V10 consumes it; ``frame_idx`` is the frame's index in that sequence
    (it drives the ordered neighbour window). ``sparse_tree`` and
    ``local_r`` come from core.gate for the adaptive spatial gate.

    Returns a dict:
        points  (N,3) float64 world points that passed every gate,
        colors  (N,3) float64 RGB in [0, 1],
        stats   per-frame {candidates, gate_pass, votes_pass, frames_fused},
        votes   {agree, contradict, neutral} counts for this frame,
        u, v    pixel coordinates of the kept candidates.
    """
    cfg = config if config is not None else PipelineConfig()
    k = cfg.camera

    u, v, z = sample_candidates(
        frame, cfg.pixel_stride, image_w=k.width, image_h=k.height,
        z_min=cfg.fuse_z_min, z_max=cfg.fuse_z_max,
    )
    stats = {
        "candidates": int(len(z)),
        "gate_pass": 0,
        "votes_pass": 0,
        "frames_fused": 0,
    }
    if len(z) == 0:
        return _empty_result(stats, u, v)

    cam_pts = backproject(u, v, z, k)
    world_pts = cam_to_world(cam_pts, frame["R"], frame["t"])

    gate = gate_candidates(
        world_pts, sparse_tree, local_r,
        gate_factor=cfg.gate_factor,
        radius_min=cfg.gate_radius_min,
        radius_max=cfg.gate_radius_max,
    )
    stats["gate_pass"] = int(np.sum(gate))
    if not np.any(gate):
        return _empty_result(stats, u, v)

    world_pts = world_pts[gate]
    u_g, v_g = u[gate], v[gate]

    net_votes, vote_delta = track_anchored_votes(
        world_pts, frame_records, frame_idx,
        neighbor_window=cfg.neighbor_window,
        track_radius_px=cfg.track_radius_px,
        relative_tolerance=cfg.relative_tolerance,
        absolute_tolerance=cfg.absolute_tolerance,
        vote_z_min=cfg.fuse_vote_z_min,
        reproject_eps=cfg.fuse_reproject_eps,
        image_w=k.width, image_h=k.height,
        intrinsics=k,
    )

    keep = net_votes >= cfg.min_net_votes
    stats["votes_pass"] = int(np.sum(keep))
    if not np.any(keep):
        return _empty_result(stats, u, v)

    world_keep = world_pts[keep]
    u_keep = u_g[keep]
    v_keep = v_g[keep]
    colors = sample_rgb(frame["rgb"], v_keep, u_keep)

    stats["frames_fused"] = 1
    return {
        "points": world_keep,
        "colors": colors,
        "stats": stats,
        "votes": vote_delta,
        "u": u_keep,
        "v": v_keep,
    }


def fuse_all(frame_records, sparse_tree, local_r, config=None):
    """Reproduce the V10 pass-2 aggregation loop (lines 662-825).

    Returns (points, colors, stats, vote_hist) where ``stats`` and
    ``vote_hist`` match V10's accumulators exactly and points/colors are
    the merged (vstacked) survivor cloud (empty (0,3) arrays when nothing
    survives; V10 raises RuntimeError at the merge step, which belongs to
    the pipeline stage, not this pure fusion layer).
    """
    cfg = config if config is not None else PipelineConfig()

    all_points = []
    all_colors = []
    vote_hist = {"agree": 0, "contradict": 0, "neutral": 0}
    stats = {"candidates": 0, "gate_pass": 0, "votes_pass": 0,
             "frames_fused": 0}

    for frame_idx, frame in enumerate(frame_records):
        res = fuse_frame(frame, frame_records, frame_idx,
                         sparse_tree, local_r, config=cfg)
        for key in stats:
            stats[key] += res["stats"][key]
        for key in vote_hist:
            vote_hist[key] += res["votes"][key]
        if len(res["points"]):
            all_points.append(res["points"])
            all_colors.append(res["colors"])

    if all_points:
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.float64)

    return points, colors, stats, vote_hist
