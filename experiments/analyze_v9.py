import os
import numpy as np
import open3d as o3d
from PIL import Image

# ============================================================
# FLIGHT2WORLD - V9 COMPARISON ANALYZER
# Loads V7 / V8 / V9 (raw + clean) + COLMAP sparse cloud and
# reports quantitative + coarse visual comparison.
# ============================================================

BASE = os.path.expanduser("~/flight2world/test")

CLOUDS = {
    "V7 (best baseline)": "fused_15_v7_clean.ply",
    "V8 (55-frame)": "fused_55_v8.ply",
    "V9 raw": "fused_v9.ply",
    "V9 clean": "fused_v9_clean.ply",
}

SPARSE_TXT = os.path.join(BASE, "sparse_txt")


def load_sparse(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(pts, dtype=np.float64)


def stats(pcd):
    P = np.asarray(pcd.points)
    mn, mx = P.min(axis=0), P.max(axis=0)
    ext = mx - mn
    return P, mn, mx, ext


def nn_to_sparse(P, S, k=3):
    """Nearest distances from each cloud point to the sparse cloud."""
    sp = o3d.geometry.PointCloud()
    sp.points = o3d.utility.Vector3dVector(S)
    tree = o3d.geometry.KDTreeFlann(sp)

    idx = np.linspace(0, len(P) - 1, min(len(P), 4000)).astype(int)
    dists = []
    for i in idx:
        cnt, _, d = tree.search_knn_vector_3d(P[i], 1)
        if cnt >= 1 and np.isfinite(d[0]):
            dists.append(np.sqrt(d[0]))
    return np.array(dists)


def render_view(pcd, name, out_dir, view="top", size=(720, 480)):
    """Coarse numpy splat render of a coloured point cloud."""
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors) if pcd.has_colors() else None

    mn, mx = P.min(axis=0), P.max(axis=0)
    center = (mn + mx) / 2.0
    span = np.max(mx - mn)

    # View rotations: orthographic axes -> image axes.
    if view == "top":
        # looking down Z: use X (right), Y (down)
        A = P[:, 0] - center[0]
        B = P[:, 1] - center[1]
    elif view == "side_x":
        A = P[:, 2] - center[2]
        B = P[:, 1] - center[1]
    elif view == "side_y":
        A = P[:, 0] - center[0]
        B = P[:, 2] - center[2]
    else:
        # oblique: combine X and Z onto horizontal axis
        A = 0.7 * (P[:, 0] - center[0]) + 0.7 * (P[:, 2] - center[2])
        B = P[:, 1] - center[1]

    s = max(np.ptp(A), np.ptp(B), 1e-9)
    u = ((A - A.min()) / s * (size[0] - 1)).astype(int)
    v = ((B - B.min()) / s * (size[1] - 1)).astype(int)

    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)

    # Depth-aware splat (paint far first). For top view use -Z as depth.
    depth_axis = P[:, 2] if view == "top" else (
        P[:, 1] if view in ("side_x",) else P[:, 0]
    )
    order = np.argsort(-depth_axis)

    for i in order[:: max(1, len(order) // 20000)][::-1]:
        x, y = u[i], v[i]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            if C is not None:
                col = (C[i] * 255).clip(0, 255).astype(np.uint8)
            else:
                col = np.array([40, 40, 40], dtype=np.uint8)
            # small splat
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < size[0] and 0 <= yy < size[1]:
                        canvas[yy, xx] = col

    out = os.path.join(out_dir, "v9_%s_%s.png" % (name.split()[0].lower().replace(" ", "_"), view))
    Image.fromarray(canvas).save(out)
    print("  rendered:", out)


def main():
    os.makedirs(os.path.join(BASE, "analysis"), exist_ok=True)

    sparse = load_sparse(os.path.join(SPARSE_TXT, "points3D.txt"))
    print("Sparse COLMAP points:", len(sparse))
    S = sparse

    for name, fname in CLOUDS.items():
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print("\n=== %s: MISSING (%s)" % (name, fname))
            continue
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            print("\n=== %s: EMPTY" % name)
            continue

        P, mn, mx, ext = stats(pcd)
        print("\n=== %s  (%s)" % (name, fname))
        print("  points:", len(P))
        print("  extent: X=%.3f Y=%.3f Z=%.3f  diag=%.3f"
              % (ext[0], ext[1], ext[2], np.linalg.norm(ext)))
        print("  Z range: %.3f .. %.3f (Z span %.3f)"
              % (mn[2], mx[2], ext[2]))

        # Height distribution: is there a dominant ground plane?
        z_hist, z_edges = np.histogram(P[:, 2], bins=20)
        dz = z_edges[1] - z_edges[0]
        peak = int(np.argmax(z_hist))
        peak_frac = z_hist[peak] / len(P)
        print("  ground-plane: peak Z bin at %.3f..%.3f holds %.1f%% of points"
              % (z_edges[peak], z_edges[peak + 1], 100 * peak_frac))

        # Sparse-consistency (hug the COLMAP authority?)
        d = nn_to_sparse(P, S)
        if len(d):
            print("  dist to sparse: med=%.4f p90=%.4f p99=%.4f"
                  % (np.median(d), np.percentile(d, 90), np.percentile(d, 99)))

        for view in ("top", "side_y", "oblique"):
            render_view(pcd, name, os.path.join(BASE, "analysis"), view)


if __name__ == "__main__":
    main()
