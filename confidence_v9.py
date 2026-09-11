import os
import json
import math
import numpy as np
import cv2
import open3d as o3d
from PIL import Image

# ============================================================
# FLIGHT2WORLD V9 - CONFIDENCE / ERROR LAYER
#
# The fusion pass computed per-candidate votes (agree /
# contradict / neutral) against the TRIANGULATED COLMAP track
# geometry but discarded them: the surviving cloud in the PLY is
# just XYZ + RGB. This script re-computes that same evidence for
# the FINAL clean cloud and persists it as a per-point
# confidence / error layer.
#
# KEY POINT: votes on a surviving point depend only on its 3D
# position, the neighbour camera poses, and the track maps --
# NOT on monocular depth or per-frame calibration. So we can
# replay them offline, exactly, without re-running Depth
# Anything or the candidate generation.
#
# Because voxel-downsampling destroyed which source frame each
# point came from, this is a GLOBAL track-agreement measure: a
# point is scored against ALL fused frames, not just the +/-3
# source window. That is a STRICTER prior-vs-truth test than
# the fusion's local check, and therefore the right error layer:
# any view in which the point contradicts the triangulated
# surface reduces its confidence.
#
# UNITS: scene units are ARBITRARY (COLMAP reconstruction scale).
# Residuals and distances are in these units -- NOT metric. No
# metric accuracy is claimed (see memory: source video has no GPS).
# ============================================================


# ============================================================
# PATHS (same layout as fusion_v9.py)
# ============================================================

BASE = os.path.expanduser("~/flight2world")
FRAME_DIR = os.path.join(BASE, "test", "frames")
SPARSE_TXT = os.path.join(BASE, "test", "sparse_txt")
DIAG = os.path.join(BASE, "test", "fused_v9_diag.json")

INPUT_CLOUD = os.path.join(BASE, "test", "fused_v9_clean.ply")
OUTPUT_PLY = os.path.join(BASE, "test", "fused_v9_confidence.ply")
OUTPUT_JSON = os.path.join(BASE, "test", "fused_v9_confidence.json")
OUTPUT_RENDER_DIR = os.path.join(BASE, "test", "analysis")


# ============================================================
# CAMERA INTRINSICS + SETTINGS (must match fusion_v9.py)
# ============================================================

FX = 1167.4277386087481
FY = 1167.4277386087481
CX = 640.0
CY = 360.0
IMAGE_W = 1280
IMAGE_H = 720

MIN_BASELINE_FRAC = 0.08
CALIB_RMSE_OUTLIER_FACTOR = 2.5
RELATIVE_TOLERANCE = 0.18
ABSOLUTE_TOLERANCE = 0.05
TRACK_RADIUS_PX = 50


# ============================================================
# COLMAP loaders (copied verbatim from fusion_v9.py so the
# replay is bit-identical)
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
    return -R.T @ t


def build_track_maps(image_data, sparse_points):
    """Rasterise triangulated tracks + iterative-dilation depth
    fill. Identical to fusion_v9.build_track_maps."""
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

    return z_grid, dist_grid


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FLIGHT2WORLD V9 - CONFIDENCE / ERROR LAYER")
    print("=" * 70)

    # --------------------------------------------------------
    # Replay the exact fused frame set from COLMAP + diag.
    # --------------------------------------------------------

    images = load_colmap_images(os.path.join(SPARSE_TXT, "images.txt"))
    sparse_points = load_colmap_points(os.path.join(SPARSE_TXT, "points3D.txt"))
    print("\nCOLMAP registered images:", len(images))
    print("COLMAP sparse points:", len(sparse_points))

    registered = [
        (image_id, data)
        for image_id, data in images.items()
        if os.path.exists(os.path.join(FRAME_DIR, data["name"]))
    ]
    registered.sort(key=lambda x: x[1]["name"])

    centers = np.array([
        camera_center(data["R"], data["t"]) for _, data in registered
    ])
    seq_baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median_baseline = float(np.median(seq_baselines))
    min_baseline = MIN_BASELINE_FRAC * median_baseline

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

    print("After min-baseline curation:", len(curated))

    # Re-apply the RMSE outlier gate using the diag's per-frame
    # RMSE values (the fusion's gate is what produced the 48
    # fused frames).
    diag = json.load(open(DIAG))
    calib_rows = diag["calibration"]
    rmse_by_name = {r["name"]: r["rmse"] for r in calib_rows}

    rmse_all = np.array([rmse_by_name[d["name"]] for _, d in curated])
    rmse_median = float(np.median(rmse_all))
    rmse_keep = rmse_all <= CALIB_RMSE_OUTLIER_FACTOR * rmse_median

    frames = [d for (_, d), ok in zip(curated, rmse_keep) if ok]

    print("RMSE gate: median=%.5f threshold=%.5f" %
          (rmse_median, CALIB_RMSE_OUTLIER_FACTOR * rmse_median))
    print("Fused frame set:", len(frames))

    diag_kept = diag["frames"]["calibrated_kept"]
    if len(frames) != diag_kept:
        raise RuntimeError(
            "Frame replay mismatch: replayed %d, diag says %d"
            % (len(frames), diag_kept)
        )
    print("Frame replay matches diag (calibrated_kept=%d)." % diag_kept)

    # --------------------------------------------------------
    # Track maps for every fused frame (no depth / no calibration
    # needed -- pure COLMAP geometry).
    # --------------------------------------------------------

    print("\nBuilding track maps for", len(frames), "frames...")
    for data in frames:
        z_grid, dist_grid = build_track_maps(data, sparse_points)
        data["z_grid"] = z_grid
        data["dist_grid"] = dist_grid
    print("Track maps built.")

    # --------------------------------------------------------
    # Load the final clean cloud.
    # --------------------------------------------------------

    pcd = o3d.io.read_point_cloud(INPUT_CLOUD)
    P = np.asarray(pcd.points, dtype=np.float64)
    C = (np.asarray(pcd.colors) * 255.0).astype(np.uint8) if pcd.has_colors() else None
    N = len(P)
    print("\nCloud points:", N)

    # --------------------------------------------------------
    # Global track-anchored vote + residual tally.
    # For each point, project into every fused frame and score
    # it against the TRIANGULATED track depth at that pixel.
    # --------------------------------------------------------

    n_agree = np.zeros(N, dtype=np.int16)
    n_contradict = np.zeros(N, dtype=np.int16)
    n_constrained_sum = np.zeros(N, dtype=np.int16)
    residual_sum = np.zeros(N, dtype=np.float64)   # |zn - track_z| on agree
    residual_n = np.zeros(N, dtype=np.int16)
    worst_resid = np.full(N, -1.0, dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int16)          # frames point projects inside

    for k, data in enumerate(frames):
        R, t = data["R"], data["t"]
        z_grid = data["z_grid"]
        dist_grid = data["dist_grid"]

        cam = (R @ P.T).T + t          # N x 3 camera space
        zn = cam[:, 2]
        inside = zn > 0.15
        if not inside.any():
            continue
        un = FX * cam[inside, 0] / np.maximum(zn[inside], 1e-8) + CX
        vn = FY * cam[inside, 1] / np.maximum(zn[inside], 1e-8) + CY
        ui = np.rint(un).astype(np.int32)
        vi = np.rint(vn).astype(np.int32)
        in_img = (ui >= 0) & (ui < IMAGE_W) & (vi >= 0) & (vi < IMAGE_H)
        ids_all = np.where(inside)[0]
        ids = ids_all[in_img]
        if not len(ids):
            continue

        coverage[ids] += 1

        track_z = z_grid[vi[in_img], ui[in_img]]
        track_d = dist_grid[vi[in_img], ui[in_img]]
        zn_i = zn[ids]

        constrained = (np.isfinite(track_z)) & (track_d <= TRACK_RADIUS_PX)
        c_ids = ids[constrained]
        if not len(c_ids):
            continue

        tz = track_z[constrained]
        tol = np.maximum(np.abs(tz) * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
        resid = np.abs(zn_i[constrained] - tz)
        agree = resid <= tol

        n_agree[c_ids[agree]] += 1
        n_contradict[c_ids[~agree]] += 1

        # Residual evidence on AGREED views (error magnitude).
        good = c_ids[agree]
        if len(good):
            residual_sum[good] += resid[agree]
            residual_n[good] += 1

        # Largest contradiction on any view (worst-case error).
        w = c_ids[~agree]
        if len(w):
            np.maximum.at(worst_resid, w, resid[~agree])

        if (k + 1) % 8 == 0 or k == len(frames) - 1:
            print("  frames processed: %d/%d" % (k + 1, len(frames)))

    # --------------------------------------------------------
    # Per-point confidence metrics.
    # --------------------------------------------------------

    n_constrained = n_agree + n_contradict

    with np.errstate(divide="ignore", invalid="ignore"):
        agree_ratio = np.where(
            n_constrained > 0,
            n_agree / np.maximum(n_constrained, 1),
            0.0,
        )

    # Laplace-smoothed agreement ratio: pulls low-evidence points
    # toward 0.5 so n_constrained is visible in the score.
    confidence = (n_agree + 1.0) / (n_constrained + 2.0)

    med_residual = np.where(
        residual_n > 0,
        residual_sum / np.maximum(residual_n, 1),
        np.nan,
    )

    # --------------------------------------------------------
    # Complementary error: distance to the triangulated sparse
    # surface (scene units, arbitrary scale).
    # --------------------------------------------------------

    sparse_xyz = np.array(list(sparse_points.values()), dtype=np.float64)
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

    # --------------------------------------------------------
    # Write PLY with custom scalar fields (CloudCompare / MeshLab
    # compatible: confidence, evidence, net, residual, dist).
    # --------------------------------------------------------

    net = n_agree - n_contradict
    resid_fill = np.where(np.isfinite(med_residual), med_residual, -1.0)
    dist_fill = np.where(np.isfinite(dist_sparse), dist_sparse, -1.0)

    lines = ["ply", "format ascii 1.0",
             "comment FLIGHT2WORLD V9 confidence layer"]
    lines.append("element vertex %d" % N)
    for prop in ("double x", "double y", "double z",
                 "uchar red", "uchar green", "uchar blue",
                 "float confidence", "int evidence",
                 "int net_votes", "float residual",
                 "float worst_residual", "float dist_sparse"):
        lines.append("property " + prop)
    lines.append("end_header")

    rows = []
    if C is None:
        C = np.zeros((N, 3), dtype=np.uint8)
    worst_fill = np.where(worst_resid >= 0, worst_resid, -1.0)
    for i in range(N):
        rows.append("%.6f %.6f %.6f %d %d %d %.4f %d %d %.5f %.5f %.5f" % (
            P[i, 0], P[i, 1], P[i, 2],
            int(C[i, 0]), int(C[i, 1]), int(C[i, 2]),
            float(confidence[i]),
            int(n_constrained[i]),
            int(net[i]),
            float(resid_fill[i]),
            float(worst_fill[i]),
            float(dist_fill[i]),
        ))

    with open(OUTPUT_PLY, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\n".join(rows) + "\n")

    print("\nSaved:", OUTPUT_PLY)

    # --------------------------------------------------------
    # Summary diagnostics.
    # --------------------------------------------------------

    def q(a, p):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if len(a) else float("nan")

    summary = {
        "input_cloud": INPUT_CLOUD,
        "points": int(N),
        "frames_replayed": len(frames),
        "frame_replay_matches_diag": True,
        "units": "ARBITRARY COLMAP scale -- NOT metric (no GPS source)",
        "confidence": {
            "definition": "Laplace-smoothed agreement ratio "
                          "(agree+1)/(agree+contradict+2) against "
                          "triangulated COLMAP track depth over all "
                          "fused frames",
            "mean": float(np.nanmean(confidence)),
            "median": q(confidence, 50),
            "p10": q(confidence, 10),
            "p90": q(confidence, 90),
        },
        "evidence": {
            "definition": "number of fused frames whose track map "
                          "constrained this pixel",
            "p50": q(n_constrained, 50),
            "p90": q(n_constrained, 90),
            "zero_evidence_points": int(np.sum(n_constrained == 0)),
        },
        "agree_ratio_raw": {
            "median": q(agree_ratio, 50),
            "p10": q(agree_ratio, 10),
        },
        "net_votes": {
            "p10": q(net, 10),
            "median": q(net, 50),
        },
        "residual_error": {
            "definition": "mean |candidate Z - track Z| on agreeing "
                          "views, scene units (arbitrary)",
            "median": q(med_residual, 50),
            "p90": q(med_residual, 90),
        },
        "worst_residual": {
            "definition": "largest |Z - track Z| over all views "
                          "(agree or not)",
            "median": q(worst_resid, 50),
            "p90": q(worst_resid, 90),
        },
        "dist_sparse": {
            "definition": "NN distance to triangulated sparse "
                          "surface, scene units (arbitrary)",
            "median": q(dist_sparse, 50),
            "p90": q(dist_sparse, 90),
        },
        "worst_residual": {
            "definition": "largest |Z - track Z| over all constraining views "
                          "(scene units, arbitrary); -1 if no view contradicts",
            "median": q(worst_resid, 50),
            "p90": q(worst_resid, 90),
            "fraction_with_contradiction": float(np.mean(worst_resid >= 0)),
        },
        "corr_confidence_dist_sparse": float(np.corrcoef(
            confidence[np.isfinite(dist_sparse)],
            dist_sparse[np.isfinite(dist_sparse)],
        )[0, 1]),
        "corr_confidence_evidence": float(np.corrcoef(
            confidence[n_constrained > 0],
            n_constrained[n_constrained > 0],
        )[0, 1]),
        "low_conf_count": int(np.sum(confidence < 0.5)),
        "low_conf_frac": float(np.mean(confidence < 0.5)),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved:", OUTPUT_JSON)
    print(json.dumps(summary, indent=2))

    # --------------------------------------------------------
    # Confidence-colored renders (numpy splat, like analyze_v9).
    # --------------------------------------------------------

    os.makedirs(OUTPUT_RENDER_DIR, exist_ok=True)
    render_confidence(P, confidence, C,
                      os.path.join(OUTPUT_RENDER_DIR, "v9_conf_top.png"),
                      view="top")
    render_confidence(P, confidence, C,
                      os.path.join(OUTPUT_RENDER_DIR, "v9_conf_side.png"),
                      view="side_y")
    render_confidence(P, confidence, C,
                      os.path.join(OUTPUT_RENDER_DIR, "v9_conf_oblique.png"),
                      view="oblique")


def render_confidence(P, conf, C, out, view="top", size=(720, 480)):
    """Splat render coloured by confidence (red=low, green=high).
    Same projection as analyze_v9.render_view."""
    mn, mx = P.min(axis=0), P.max(axis=0)
    center = (mn + mx) / 2.0

    if view == "top":
        A = P[:, 0] - center[0]
        B = P[:, 1] - center[1]
    elif view == "side_y":
        A = P[:, 0] - center[0]
        B = P[:, 2] - center[2]
    else:
        A = 0.7 * (P[:, 0] - center[0]) + 0.7 * (P[:, 2] - center[2])
        B = P[:, 1] - center[1]

    s = max(np.ptp(A), np.ptp(B), 1e-9)
    u = ((A - A.min()) / s * (size[0] - 1)).astype(int)
    v = ((B - B.min()) / s * (size[1] - 1)).astype(int)

    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)

    # Depth ordering: top -> -Z, side_y -> -X, oblique -> -Y.
    depth_axis = P[:, 2] if view == "top" else (
        P[:, 1] if view == "side_y" else P[:, 0])
    order = np.argsort(-depth_axis)

    # Confidence -> red(green) map: low = red, high = green.
    conf_n = np.clip(conf, 0.0, 1.0)
    rcol = (255 * (1.0 - conf_n)).astype(np.uint8)
    gcol = (255 * conf_n).astype(np.uint8)
    bcol = np.full_like(rcol, 0, dtype=np.uint8)

    for i in order[:: max(1, len(order) // 20000)][::-1]:
        x, y = u[i], v[i]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            col = np.array([rcol[i], gcol[i], bcol[i]])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < size[0] and 0 <= yy < size[1]:
                        canvas[yy, xx] = col

    Image.fromarray(canvas).save(out)
    print("  rendered:", out)


if __name__ == "__main__":
    main()
