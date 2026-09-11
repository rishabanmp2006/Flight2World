import os
import math
import json
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from transformers import pipeline


# ============================================================
# FLIGHT2WORLD V9
# TRACK-ANCHORED FUSION
#
# HYPOTHESIS
# ----------
# V8's two discriminators are both non-discriminating:
#
#   1. The multi-view consistency check compares monocular
#      depth against monocular depth. Every frame's depth comes
#      from the SAME network, calibrated with the SAME global
#      linear fit against the SAME sparse cloud. Hallucinated
#      surfaces (flattened roofs, filled vegetation, stretched
#      facades) are therefore SELF-CONSISTENT across views.
#      V8 evidence: 373540/377702 = 98.9% of gated candidates
#      passed "multi-view consistency".
#
#   2. The COLMAP gate is a pure spatial ball. V8 used a global
#      radius clip(median*8, .08, .20) = 0.08 units against a
#      scene with p90 sparse spacing of only 0.042 units.
#      It asks "is ANY sparse point within a ball?", not
#      "does this surface agree with the triangulated tracks
#      visible at this pixel?".
#
# V9 FIX
# ------
# Replace both with a TRACK-ANCHORED reprojection consistency
# check: a candidate point is compared against the
# TRIANGULATED COLMAP track geometry (ground truth) visible at
# its reprojected pixel in neighbouring views -- prior-vs-truth,
# not prior-vs-prior. Surfaces that contradict the triangulated
# surface get rejected wherever tracks constrain them.
#
# Additional structural fixes (not threshold tweaks):
#   * quality-gated frame selection (greedy min-baseline + calib)
#   * robust per-frame calibration, linear vs inverse-depth
#   * density-adaptive COLMAP gate (local radius per candidate)
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = os.path.expanduser("~/flight2world")
FRAME_DIR = os.path.join(BASE, "test", "frames")
SPARSE_TXT = os.path.join(BASE, "test", "sparse_txt")

OUTPUT_RAW = os.path.join(BASE, "test", "fused_v9.ply")
OUTPUT_CLEAN = os.path.join(BASE, "test", "fused_v9_clean.ply")
OUTPUT_DIAG = os.path.join(BASE, "test", "fused_v9_diag.json")

# Smoke test mode: run on a tiny slice to validate logic fast.
SMOKE_TEST = int(os.environ.get("V9_SMOKE", "0"))


# ============================================================
# CAMERA INTRINSICS (from COLMAP cameras.txt)
# ============================================================

FX = 1167.4277386087481
FY = 1167.4277386087481
CX = 640.0
CY = 360.0
IMAGE_W = 1280
IMAGE_H = 720


# ============================================================
# SETTINGS
# ============================================================

# Generate one candidate every N pixels.
PIXEL_STRIDE = 8

# Ordered neighbour window for cross-view checks.
NEIGHBOR_WINDOW = 3

# Minimum NET votes across neighbours for a candidate to pass.
# net = (#track-agreements) - (#track-contradictions)
MIN_NET_VOTES = 2

# TRACK-ANCHORED CHECK
# A candidate reprojected into a neighbour agrees with the
# triangulated surface if its depth is within RELATIVE_TOLERANCE
# (or ABSOLUTE_TOLERANCE) of the nearest triangulated track's
# depth. The neighbour only votes when a track lies within
# TRACK_RADIUS_PX of the reprojected pixel.
RELATIVE_TOLERANCE = 0.18
ABSOLUTE_TOLERANCE = 0.05
TRACK_RADIUS_PX = 50

# ADAPTIVE COLMAP GATE
# Per-candidate radius = local_r(nearest sparse point) * FACTOR,
# where local_r is the 8th-nearest-neighbour distance of that
# sparse point. Tighter in dense track regions, looser in gaps.
GATE_FACTOR = 2.0
GATE_RADIUS_MIN = 0.015
GATE_RADIUS_MAX = 0.12

# FRAME CURATION
# Greedy selection: keep a frame only if its camera center is at
# least MIN_BASELINE_FRAC * median(consecutive baseline) from the
# last kept frame. Removes near-duplicate views that add noise
# but no parallax.
MIN_BASELINE_FRAC = 0.08

# CALIBRATION QUALITY GATES
CALIB_MIN_CORR = 0.55
CALIB_MIN_SAMPLES = 100
CALIB_RMSE_OUTLIER_FACTOR = 2.5

# FINAL CLEANUP
VOXEL_SIZE = 0.04
STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 1.5


# ============================================================
# COLMAP LOADERS
# ============================================================

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def load_colmap_images(path):
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
            except ValueError:
                i += 1
                continue

            R = qvec2rotmat(np.array([qw, qx, qy, qz]))
            t = np.array([tx, ty, tz], dtype=np.float64)

            observations = []
            if i + 1 < len(lines):
                obs_line = lines[i + 1].strip()
                if obs_line and not obs_line.startswith("#"):
                    obs_parts = obs_line.split()
                    for j in range(0, len(obs_parts) - 2, 3):
                        try:
                            x = float(obs_parts[j])
                            y = float(obs_parts[j + 1])
                            point_id = int(obs_parts[j + 2])
                            if point_id >= 0:
                                observations.append((x, y, point_id))
                        except ValueError:
                            pass
                    i += 2
                    images[image_id] = {
                        "id": image_id,
                        "name": name,
                        "R": R,
                        "t": t,
                        "observations": observations,
                    }
                    continue

        i += 1

    return images


def load_colmap_points(path):
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                point_id = int(parts[0])
                points[point_id] = np.array(
                    list(map(float, parts[1:4])), dtype=np.float64
                )
            except ValueError:
                continue
    return points


def camera_center(R, t):
    """C = -R^T t (COLMAP convention: X_cam = R X_world + t)."""
    return -R.T @ t


# ============================================================
# ROBUST PER-FRAME CALIBRATION
#
# Z = a * depth + b            (linear)
# Z = a / depth + b            (inverse)
#
# We fit BOTH with an iteratively re-weighted least squares that
# trims residuals beyond 3*MAD, and pick the model with the lower
# RMSE per frame. Returns None if the frame is unusable.
# ============================================================

def robust_fit(x, z, iterations=5):
    """Robust linear fit z = a*x + b on paired arrays."""
    if len(x) < 50:
        return None

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(z)
    x, z = x[valid], z[valid]
    if len(x) < 50:
        return None

    # Trim extreme tails once (fixed 2-98 percentile).
    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_z, hi_z = np.percentile(z, [2, 98])
    mask = (
        (x >= lo_x) & (x <= hi_x) &
        (z >= lo_z) & (z <= hi_z)
    )
    x, z = x[mask], z[mask]
    if len(x) < 50:
        return None

    A = np.vstack([x, np.ones_like(x)]).T

    coef = np.linalg.lstsq(A, z, rcond=None)[0]

    for _ in range(iterations):
        pred = A @ coef
        resid = z - pred
        mad = np.median(np.abs(resid - np.median(resid)))
        if mad < 1e-12:
            break
        good = np.abs(resid) <= 3.0 * 1.4826 * mad
        if good.sum() < 50:
            break
        coef = np.linalg.lstsq(A[good], z[good], rcond=None)[0]

    pred = A @ coef
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    corr = float(np.corrcoef(x, z)[0, 1])

    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "rmse": rmse,
        "corr": corr,
        "samples": len(x),
    }


def calibrate_frame(image_data, sparse_points, depth_map):
    """Collect (depth_at_obs, COLMAP camera Z at obs) pairs."""
    R = image_data["R"]
    t = image_data["t"]

    d_vals = []
    z_vals = []

    for x, y, point_id in image_data["observations"]:
        if point_id not in sparse_points:
            continue

        X_world = sparse_points[point_id]
        X_cam = R @ X_world + t
        z = X_cam[2]
        if z <= 0:
            continue

        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= IMAGE_W or yi < 0 or yi >= IMAGE_H:
            continue

        d = depth_map[yi, xi]
        if not (np.isfinite(d) and d > 0):
            continue

        d_vals.append(d)
        z_vals.append(z)

    return np.asarray(d_vals), np.asarray(z_vals)


# ============================================================
# TRACK DEPTH MAP
#
# For every pixel of a frame we want the camera-space depth Z of
# the nearest TRIANGULATED COLMAP track. Tracks are sparse
# (~1000s of pixels per image), so we rasterize them and fill the
# gaps by iterative dilation, propagating depth as the mean of the
# valid 8-neighbours. This is an approximate nearest-track depth
# map, computed with only cv2/numpy (no scipy).
# ============================================================

def build_track_maps(image_data, sparse_points):
    R = image_data["R"]
    t = image_data["t"]

    z_grid = np.full((IMAGE_H, IMAGE_W), np.nan, dtype=np.float32)
    valid = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)

    for x, y, point_id in image_data["observations"]:
        if point_id not in sparse_points:
            continue
        X_cam = R @ sparse_points[point_id] + t
        z = X_cam[2]
        if z <= 0:
            continue
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < IMAGE_W and 0 <= yi < IMAGE_H:
            z_grid[yi, xi] = z
            valid[yi, xi] = 1

    mask = valid.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dist_grid = np.full((IMAGE_H, IMAGE_W), 0, dtype=np.int32)

    step = 0
    while not mask.all() and step < 100:
        step += 1
        prev = mask
        mask = cv2.dilate(mask, kernel)
        added = (mask.astype(bool)) & (~prev.astype(bool))
        if not added.any():
            break

        zsum = cv2.filter2D(
            np.where(prev.astype(bool), z_grid, 0.0).astype(np.float64),
            -1, kernel.astype(np.float64),
            borderType=cv2.BORDER_CONSTANT,
        )
        cnt = cv2.filter2D(
            prev.astype(np.float64),
            -1, kernel.astype(np.float64),
            borderType=cv2.BORDER_CONSTANT,
        )
        fill = added & (cnt > 0)
        if not fill.any():
            break
        z_grid[fill] = (zsum[fill] / np.maximum(cnt[fill], 1.0)).astype(
            np.float32
        )
        dist_grid[added] = step

    # dist_grid = 0 on real track pixels; >0 elsewhere.
    return z_grid, dist_grid, int(np.sum(valid.astype(bool)))


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("FLIGHT2WORLD V9")
print("TRACK-ANCHORED FUSION")
print("=" * 70)
print()


# ------------------------------------------------------------
# Load COLMAP
# ------------------------------------------------------------

images = load_colmap_images(os.path.join(SPARSE_TXT, "images.txt"))
sparse_points = load_colmap_points(os.path.join(SPARSE_TXT, "points3D.txt"))

print("Registered COLMAP images:", len(images))
print("COLMAP sparse points:", len(sparse_points))

if len(images) < 20:
    raise RuntimeError("Too few registered COLMAP images.")

# ------------------------------------------------------------
# Order frames chronologically
# ------------------------------------------------------------

registered = [
    (image_id, data)
    for image_id, data in images.items()
    if os.path.exists(os.path.join(FRAME_DIR, data["name"]))
]
registered.sort(key=lambda x: x[1]["name"])

print("Usable registered frames:", len(registered))

if SMOKE_TEST:
    # Keep a CONSECUTIVE block so the ordered-neighbour window
    # reproduces the real fusion geometry (baseline ~0.38 units,
    # up to 6 neighbours per candidate).
    registered = registered[:SMOKE_TEST]
    print("SMOKE TEST: using", len(registered), "consecutive frames")


# ------------------------------------------------------------
# Greedy minimum-baseline frame curation
# ------------------------------------------------------------

centers = np.array([
    camera_center(data["R"], data["t"]) for _, data in registered
])

seq_baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
median_baseline = float(np.median(seq_baselines))
min_baseline = MIN_BASELINE_FRAC * median_baseline

print()
print("Camera path statistics:")
print("  consecutive baseline: median=%.4f min=%.4f max=%.4f"
      % (median_baseline, seq_baselines.min(), seq_baselines.max()))
print("  min baseline for curation: %.4f" % min_baseline)

curated = []
last_center = None
for (image_id, data), center in zip(registered, centers):
    if last_center is None:
        curated.append((image_id, data))
        last_center = center
        continue
    if np.linalg.norm(center - last_center) >= min_baseline:
        curated.append((image_id, data))
        last_center = center

print("  after near-duplicate curation: %d frames" % len(curated))
registered = curated


# ------------------------------------------------------------
# Sparse cloud + per-point local radius (adaptive gate)
# ------------------------------------------------------------

sparse_xyz = np.array(list(sparse_points.values()), dtype=np.float64)

sparse_pcd = o3d.geometry.PointCloud()
sparse_pcd.points = o3d.utility.Vector3dVector(sparse_xyz)
sparse_tree = o3d.geometry.KDTreeFlann(sparse_pcd)

# local_r[i] = distance from sparse point i to its 8th nearest
# sparse neighbour (a local density measure).
local_r = np.zeros(len(sparse_xyz), dtype=np.float64)
for i in range(len(sparse_xyz)):
    cnt, _, d = sparse_tree.search_knn_vector_3d(sparse_xyz[i], 8)
    if cnt >= 8:
        local_r[i] = math.sqrt(d[7])
    elif cnt >= 2:
        local_r[i] = math.sqrt(d[-1])
    else:
        local_r[i] = 0.02

print()
print("Adaptive gate local radius: "
      "median=%.5f p90=%.5f" % (
          np.median(local_r), np.percentile(local_r, 90)))


# ------------------------------------------------------------
# Load Depth Anything V2 Small (MPS)
# ------------------------------------------------------------

print()
print("Loading Depth Anything V2 Small on MPS...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="mps",
)

print("Depth model loaded.")


# ------------------------------------------------------------
# First pass: depth + robust calibration + track maps
# ------------------------------------------------------------

frame_records = []
calib_rows = []

print()
print("=" * 70)
print("PASS 1: DEPTH + CALIBRATION + TRACK MAPS")
print("=" * 70)
print()

for index, (image_id, image_data) in enumerate(registered):
    name = image_data["name"]
    print("[%02d/%02d] %s" % (index + 1, len(registered), name), flush=True)

    image = Image.open(os.path.join(FRAME_DIR, name)).convert("RGB")
    result = depth_pipe(image)

    # IMPORTANT: predicted_depth is the real geometry tensor.
    # result["depth"] is only the visualized depth image.
    predicted = result["predicted_depth"]
    if hasattr(predicted, "detach"):
        predicted = predicted.detach()
    predicted = predicted.squeeze().cpu().numpy().astype(np.float32)

    if predicted.shape != (IMAGE_H, IMAGE_W):
        predicted = cv2.resize(
            predicted, (IMAGE_W, IMAGE_H),
            interpolation=cv2.INTER_LINEAR,
        )

    d_vals, z_vals = calibrate_frame(image_data, sparse_points, predicted)
    if len(z_vals) < CALIB_MIN_SAMPLES:
        print("   too few calibration samples: %d -> SKIPPED" % len(z_vals))
        calib_rows.append({"name": name, "skipped": "few_samples",
                           "samples": int(len(z_vals))})
        continue

    # NOTE (empirically verified): Depth Anything V2 predicted_depth is
    # INVERSE depth -- higher value = closer to camera. So the physical
    # calibration is Z = a/d + b, and the SIGN of corr(d, Z) merely
    # encodes this convention. We fit BOTH models, choose the lower-RMSE
    # one, and gate on |corr| (magnitude = fit quality, sign = convention).
    linear = robust_fit(d_vals, z_vals)
    inverse = robust_fit(1.0 / np.maximum(d_vals, 1e-6), z_vals)

    candidate = None
    model = None
    if linear and inverse:
        if inverse["rmse"] <= linear["rmse"]:
            candidate, model = inverse, "inverse"
        else:
            candidate, model = linear, "linear"
    elif linear:
        candidate, model = linear, "linear"
    elif inverse:
        candidate, model = inverse, "inverse"

    if (
        candidate is None
        or abs(candidate["corr"]) < CALIB_MIN_CORR
    ):
        print("   calibration rejected (|corr|=%.3f) -> SKIPPED"
              % (abs(candidate["corr"]) if candidate else -1))
        calib_rows.append({"name": name, "skipped": "low_corr",
                           "corr": candidate["corr"] if candidate else None})
        continue

    z_grid, dist_grid, n_tracks = build_track_maps(image_data, sparse_points)
    if n_tracks < 50:
        print("   too few tracks: %d -> SKIPPED" % n_tracks)
        calib_rows.append({"name": name, "skipped": "few_tracks",
                           "tracks": n_tracks})
        continue

    frame_records.append({
        "name": name,
        "R": image_data["R"],
        "t": image_data["t"],
        "depth": predicted,
        "rgb": np.asarray(image, dtype=np.uint8),
        "a": candidate["a"],
        "b": candidate["b"],
        "model": model,
        "rmse": candidate["rmse"],
        "corr": candidate["corr"],
        "samples": candidate["samples"],
        "z_grid": z_grid,
        "dist_grid": dist_grid,
    })

    calib_rows.append({
        "name": name,
        "model": model,
        "samples": candidate["samples"],
        "corr": round(candidate["corr"], 4),
        "rmse": round(candidate["rmse"], 5),
        "tracks": n_tracks,
    })

    print("   model=%s samples=%d corr=%.3f rmse=%.5f tracks=%d"
          % (model, candidate["samples"], candidate["corr"],
             candidate["rmse"], n_tracks), flush=True)


print()
print("Calibrated frames:", len(frame_records))

min_frames = 2 if SMOKE_TEST else 10
if len(frame_records) < min_frames:
    raise RuntimeError("Too few calibrated frames.")

# Drop calibration outliers: frames with RMSE far above the pack.
rmse_all = np.array([f["rmse"] for f in frame_records])
rmse_median = float(np.median(rmse_all))
rmse_keep = rmse_all <= CALIB_RMSE_OUTLIER_FACTOR * rmse_median
frame_records = [
    f for f, ok in zip(frame_records, rmse_keep) if ok
]
print("After RMSE outlier gate: %d frames "
      "(median RMSE %.5f, threshold %.5f)"
      % (len(frame_records), rmse_median,
         CALIB_RMSE_OUTLIER_FACTOR * rmse_median))


# ------------------------------------------------------------
# Pass 2: track-anchored dense fusion
# ------------------------------------------------------------

print()
print("=" * 70)
print("PASS 2: TRACK-ANCHORED DENSE FUSION")
print("=" * 70)
print()

all_points = []
all_colors = []
vote_hist = {"agree": 0, "contradict": 0, "neutral": 0}
stats = {
    "candidates": 0,
    "gate_pass": 0,
    "votes_pass": 0,
    "frames_fused": 0,
}


def calibrated_depth(frame, d):
    if frame["model"] == "inverse":
        return frame["a"] / np.maximum(d, 1e-6) + frame["b"]
    return frame["a"] * d + frame["b"]


for frame_idx, frame in enumerate(frame_records):
    name = frame["name"]
    print("[FUSE %02d/%02d] %s" % (frame_idx + 1, len(frame_records), name),
          flush=True)

    depth = frame["depth"]
    rgb = frame["rgb"]
    R = frame["R"]
    t = frame["t"]

    # --------------------------------------------------------
    # Sample pixel grid + calibrated Z
    # --------------------------------------------------------

    ys = np.arange(0, IMAGE_H, PIXEL_STRIDE)
    xs = np.arange(0, IMAGE_W, PIXEL_STRIDE)
    u, v = np.meshgrid(xs, ys)
    u = u.reshape(-1)
    v = v.reshape(-1)

    ai = depth[v, u].astype(np.float64)
    valid = np.isfinite(ai) & (ai > 0)
    u, v, ai = u[valid], v[valid], ai[valid]

    z = calibrated_depth(frame, ai)
    valid = np.isfinite(z) & (z > 0.15) & (z < 50.0)
    u, v, z = u[valid], v[valid], z[valid]

    stats["candidates"] += len(z)

    if len(z) == 0:
        continue

    # --------------------------------------------------------
    # Backproject + world transform
    # --------------------------------------------------------

    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    cam_pts = np.column_stack([x, y, z])
    world_pts = (R.T @ (cam_pts - t).T).T

    # --------------------------------------------------------
    # ADAPTIVE COLMAP GATE
    # radius per candidate = local_r[nearest sparse point] * FACTOR
    # --------------------------------------------------------

    gate = np.zeros(len(world_pts), dtype=bool)
    for i in range(len(world_pts)):
        cnt, idx, d = sparse_tree.search_knn_vector_3d(world_pts[i], 1)
        if cnt < 1:
            continue
        nn_dist = math.sqrt(d[0])
        radius = float(np.clip(
            local_r[idx[0]] * GATE_FACTOR,
            GATE_RADIUS_MIN, GATE_RADIUS_MAX,
        ))
        if np.isfinite(nn_dist) and nn_dist <= radius:
            gate[i] = True

    stats["gate_pass"] += int(np.sum(gate))
    if not np.any(gate):
        continue

    world_pts = world_pts[gate]
    u_g, v_g = u[gate], v[gate]

    # --------------------------------------------------------
    # TRACK-ANCHORED REPROJECTION CONSISTENCY
    #
    # For each neighbour: project every candidate into it and
    # compare against the neighbour's TRIANGULATED track depth
    # (not its monocular depth). A neighbour votes +1 (agree)
    # if the candidate is within tolerance of the nearest track
    # depth, -1 (contradict) if a track is nearby but its depth
    # disagrees, 0 (neutral) if no track constrains the pixel.
    # --------------------------------------------------------

    net_votes = np.zeros(len(world_pts), dtype=np.int8)

    start = max(0, frame_idx - NEIGHBOR_WINDOW)
    end = min(len(frame_records), frame_idx + NEIGHBOR_WINDOW + 1)

    for j in range(start, end):
        if j == frame_idx:
            continue

        nb = frame_records[j]
        Rn, tn = nb["R"], nb["t"]
        z_grid_n = nb["z_grid"]
        dist_grid_n = nb["dist_grid"]

        cam_n = (Rn @ world_pts.T).T + tn
        zn = cam_n[:, 2]

        ok_z = np.isfinite(zn) & (zn > 0.15)
        if not np.any(ok_z):
            continue

        un = FX * cam_n[:, 0] / np.maximum(zn, 1e-8) + CX
        vn = FY * cam_n[:, 1] / np.maximum(zn, 1e-8) + CY

        ui = np.rint(un).astype(np.int32)
        vi = np.rint(vn).astype(np.int32)

        inside = (
            ok_z &
            (ui >= 0) & (ui < IMAGE_W) &
            (vi >= 0) & (vi < IMAGE_H)
        )
        if not np.any(inside):
            continue

        ids = np.where(inside)[0]
        ui_i, vi_i = ui[ids], vi[ids]
        zn_i = zn[ids]

        track_z = z_grid_n[vi_i, ui_i]
        track_d = dist_grid_n[vi_i, ui_i]

        constrained = np.isfinite(track_z) & (track_d <= TRACK_RADIUS_PX)

        agree = np.zeros(len(ids), dtype=bool)
        if np.any(constrained):
            tol = np.maximum(
                np.abs(track_z[constrained]) * RELATIVE_TOLERANCE,
                ABSOLUTE_TOLERANCE,
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
                    vote_hist["agree"] += len(sel_ids)
                elif delta < 0:
                    vote_hist["contradict"] += len(sel_ids)
                else:
                    vote_hist["neutral"] += len(sel_ids)

    keep = net_votes >= MIN_NET_VOTES
    stats["votes_pass"] += int(np.sum(keep))
    if not np.any(keep):
        continue

    world_keep = world_pts[keep]
    u_keep = u_g[keep]
    v_keep = v_g[keep]

    colors = rgb[v_keep, u_keep].astype(np.float64) / 255.0

    all_points.append(world_keep)
    all_colors.append(colors)
    stats["frames_fused"] += 1

    print("   candidates=%d gate=%d votes_pass=%d"
          % (len(z), int(np.sum(gate)), int(np.sum(keep))), flush=True)


print()
print("Fusion stage totals:")
for k, v in stats.items():
    print("  %-12s %d" % (k, v))
print("  vote histogram:", vote_hist)


# ------------------------------------------------------------
# Merge + voxel + statistical cleanup
# ------------------------------------------------------------

print()
print("Merging...")

if not all_points:
    raise RuntimeError("No points survived track-anchored fusion.")

points = np.vstack(all_points)
colors = np.vstack(all_colors)

print("Raw fused points:", len(points))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Voxel downsampling (%.3f)..." % VOXEL_SIZE)
pcd = pcd.voxel_down_sample(VOXEL_SIZE)
print("After voxel:", len(pcd.points))

print("Statistical cleanup (%d, %.1f)..."
      % (STAT_NB_NEIGHBORS, STAT_STD_RATIO))
if len(pcd.points) > 100:
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS,
        std_ratio=STAT_STD_RATIO,
    )
print("After cleanup:", len(pcd.points))

o3d.io.write_point_cloud(OUTPUT_RAW, pcd)
print("Saved raw:", OUTPUT_RAW)


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

diag = {
    "smoke_test": bool(SMOKE_TEST),
    "settings": {
        "pixel_stride": PIXEL_STRIDE,
        "neighbor_window": NEIGHBOR_WINDOW,
        "min_net_votes": MIN_NET_VOTES,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "track_radius_px": TRACK_RADIUS_PX,
        "gate_factor": GATE_FACTOR,
        "gate_radius_min": GATE_RADIUS_MIN,
        "gate_radius_max": GATE_RADIUS_MAX,
        "min_baseline_frac": MIN_BASELINE_FRAC,
        "voxel_size": VOXEL_SIZE,
    },
    "sparse": {
        "points": int(len(sparse_xyz)),
        "local_r_median": float(np.median(local_r)),
        "local_r_p90": float(np.percentile(local_r, 90)),
    },
    "frames": {
        "registered": int(len(images)),
        "curated": int(len(curated)),
        "calibrated_kept": len(frame_records),
    },
    "calibration": calib_rows,
    "fusion_stats": {k: int(v) for k, v in stats.items()},
    "vote_histogram": {k: int(v) for k, v in vote_hist.items()},
    "final_points": int(len(pcd.points)),
}

with open(OUTPUT_DIAG, "w") as f:
    json.dump(diag, f, indent=2)

print("Saved diagnostics:", OUTPUT_DIAG)


# ------------------------------------------------------------
# Optional DBSCAN-based clean stage (mirrors clean_v7)
# ------------------------------------------------------------

print()
print("=" * 70)
print("CLEAN STAGE (DBSCAN main-cluster retention)")
print("=" * 70)
print()

try:
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    largest_dim = float(np.max(extent))

    vox = max(0.005, min(0.10, largest_dim * 0.003))
    down = pcd.voxel_down_sample(vox)
    print("Clean voxel size: %.4f -> %d points"
          % (vox, len(down.points)))

    labels = np.array(down.cluster_dbscan(
        eps=vox * 3.5, min_points=8, print_progress=False
    ))

    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        print("No valid clusters; keeping raw cloud.")
    else:
        unique, counts = np.unique(valid_labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        main_label = unique[order[0]]
        main_mask = labels == main_label

        main_pts = np.asarray(down.points)[main_mask]
        main_center = np.mean(main_pts, axis=0)

        keep_labels = [main_label]
        for label, count in zip(unique, counts):
            if label == main_label:
                continue
            if count < max(30, int(len(main_pts) * 0.001)):
                continue
            center = np.mean(np.asarray(down.points)[labels == label], axis=0)
            rel_dist = np.linalg.norm(center - main_center) / largest_dim
            if rel_dist < 0.35:
                keep_labels.append(label)

        keep_mask = np.isin(labels, keep_labels)
        clean_pts = np.asarray(down.points)[keep_mask]
        clean_cols = None
        if down.has_colors():
            clean_cols = np.asarray(down.colors)[keep_mask]

        clean = o3d.geometry.PointCloud()
        clean.points = o3d.utility.Vector3dVector(clean_pts)
        if clean_cols is not None:
            clean.colors = o3d.utility.Vector3dVector(clean_cols)

        if len(clean.points) > 100:
            clean, _ = clean.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.5
            )

        o3d.io.write_point_cloud(OUTPUT_CLEAN, clean)
        print("Saved clean:", OUTPUT_CLEAN)
        print("Clean points:", len(clean.points))

        diag["clean_points"] = int(len(clean.points))
        with open(OUTPUT_DIAG, "w") as f:
            json.dump(diag, f, indent=2)

except Exception as exc:
    print("Clean stage failed (keeping raw):", exc)


print()
print("=" * 70)
print("V9 COMPLETE")
print("=" * 70)
print()
