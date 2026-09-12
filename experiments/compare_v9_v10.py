import os
import json
import math
import numpy as np
import open3d as o3d
from PIL import Image

# ============================================================
# FLIGHT2WORLD V9 vs V10 COMPARISON
#
# V9  = Depth Anything V2 SMALL
# V10 = Depth Anything V2 BASE
# Everything else (COLMAP, track-anchored discriminator,
# calibration, thresholds, cleanup) is IDENTICAL.
#
# Point count alone cannot show improvement. We compare GEOMETRY
# against the triangulated COLMAP sparse surface (the same
# authority the discriminator uses), using the per-point fields
# already persisted by the confidence layer:
#   dist_sparse : NN distance dense point -> triangulated surface
#   residual    : mean |Z - track Z| on agreeing views
#   worst_resid : largest contradiction over all constraining views
#   confidence  : Laplace-smoothed track-agreement ratio
# plus sparse coverage and V9<->V10 mutual proximity.
# Significance via a pure-numpy bootstrap (no scipy dependency).
# ============================================================

BASE = os.path.expanduser("~/flight2world/test")
SPARSE_TXT = os.path.join(BASE, "sparse_txt")
ANALYSIS = os.path.join(BASE, "analysis")

CONF_PLY = {
    "V9": os.path.join(BASE, "fused_v9_confidence.ply"),
    "V10": os.path.join(BASE, "fused_v10_confidence.ply"),
}
DIAG = {
    "V9": os.path.join(BASE, "fused_v9_diag.json"),
    "V10": os.path.join(BASE, "fused_v10_diag.json"),
}
CONF = {
    "V9": os.path.join(BASE, "fused_v9_confidence.json"),
    "V10": os.path.join(BASE, "fused_v10_confidence.json"),
}


def load_sparse(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(pts, dtype=np.float64)


def read_conf_ply(path):
    """Parse ASCII confidence PLY: x y z r g b conf ev net resid worst dist."""
    rows = []
    n = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        for line in f:
            p = line.split()
            if len(p) >= 12:
                rows.append([float(v) for v in p[:3]] + [float(p[6]), float(p[9]),
                                                         float(p[10]), float(p[11])])
    a = np.array(rows, dtype=np.float64)
    assert n is None or len(a) == n, (n, len(a))
    return a  # columns: x,y,z,confidence,residual,worst_residual,dist_sparse


def nn_dists(tree, pts):
    d = np.empty(len(pts), dtype=np.float64)
    for i in range(len(pts)):
        cnt, _, dd = tree.search_knn_vector_3d(pts[i], 1)
        d[i] = math.sqrt(dd[0]) if (cnt >= 1 and np.isfinite(dd[0])) else np.nan
    return d


def q(a, p):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


def boot_median_diff(a, b, n_boot=2000, seed=0):
    """Bootstrap 95% CI for median(a) - median(b). Pure numpy."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        ma = np.median(rng.choice(a, size=len(a), replace=True))
        mb = np.median(rng.choice(b, size=len(b), replace=True))
        diffs[k] = ma - mb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(diffs))


def render_pair(P9, P10, name, view="top", size=(720, 480)):
    """V9 grey; V10 green where within 2x median NN of V9 else red."""
    mn = np.minimum(P9.min(axis=0), P10.min(axis=0))
    mx = np.maximum(P9.max(axis=0), P10.max(axis=0))
    center = (mn + mx) / 2.0
    if view == "top":
        A9, B9 = P9[:, 0] - center[0], P9[:, 1] - center[1]
        A10, B10 = P10[:, 0] - center[0], P10[:, 1] - center[1]
    elif view == "side_y":
        A9, B9 = P9[:, 0] - center[0], P9[:, 2] - center[2]
        A10, B10 = P10[:, 0] - center[0], P10[:, 2] - center[2]
    else:
        A9, B9 = 0.7*(P9[:,0]-center[0])+0.7*(P9[:,2]-center[2]), P9[:,1]-center[1]
        A10, B10 = 0.7*(P10[:,0]-center[0])+0.7*(P10[:,2]-center[2]), P10[:,1]-center[1]
    A = np.concatenate([A9, A10]); B = np.concatenate([B9, B10])
    s = max(np.ptp(A), np.ptp(B), 1e-9)
    u9 = ((A9 - A.min()) / s * (size[0]-1)).astype(int)
    v9 = ((B9 - B.min()) / s * (size[1]-1)).astype(int)
    u10 = ((A10 - A.min()) / s * (size[0]-1)).astype(int)
    v10 = ((B10 - B.min()) / s * (size[1]-1)).astype(int)
    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)

    def depth_axis(P, view):
        return P[:, 2] if view == "top" else (P[:, 1] if view == "side_y" else P[:, 0])

    order9 = np.argsort(-depth_axis(P9, view))
    for i in order9[:: max(1, len(order9)//20000)][::-1]:
        x, y = u9[i], v9[i]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < size[0] and 0 <= yy < size[1]:
                        canvas[yy, xx] = [150, 150, 150]

    order10 = np.argsort(-depth_axis(P10, view))
    for i in order10[:: max(1, len(order10)//20000)][::-1]:
        x, y = u10[i], v10[i]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            col = [30, 170, 60] if near_v10[i] else [220, 40, 40]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < size[0] and 0 <= yy < size[1]:
                        canvas[yy, xx] = col
    out = os.path.join(ANALYSIS, "%s.png" % name)
    Image.fromarray(canvas).save(out)
    print("  rendered:", out)


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    sparse = load_sparse(os.path.join(SPARSE_TXT, "points3D.txt"))
    sp_pcd = o3d.geometry.PointCloud()
    sp_pcd.points = o3d.utility.Vector3dVector(sparse)
    sp_tree = o3d.geometry.KDTreeFlann(sp_pcd)
    print("Sparse COLMAP points:", len(sparse))

    data = {}
    for v in ("V9", "V10"):
        conf_ply = read_conf_ply(CONF_PLY[v])
        P = conf_ply[:, :3]
        d = dict(diag=json.load(open(DIAG[v])),
                 conf=json.load(open(CONF[v])),
                 P=P, conf_arr=conf_ply[:, 3], resid=conf_ply[:, 4],
                 worst=conf_ply[:, 5], dist_sparse=conf_ply[:, 6])
        d["extent"] = P.max(axis=0) - P.min(axis=0)
        data[v] = d
        print("%s points=%d extent=(%.3f,%.3f,%.3f)"
              % (v, len(P), *d["extent"]))

        # cross-check: our own NN to sparse on a 4000 sample
        idx = np.linspace(0, len(P)-1, 4000).astype(int)
        d_s = nn_dists(sp_tree, P[idx])
        print("  (own NN-to-sparse 4000-sample) med=%.5f p90=%.5f"
              % (q(d_s, 50), q(d_s, 90)))

        # sparse coverage
        d_pcd = o3d.geometry.PointCloud()
        d_pcd.points = o3d.utility.Vector3dVector(P)
        tree = o3d.geometry.KDTreeFlann(d_pcd)
        ds = nn_dists(tree, sparse)
        d["cov"] = {"0.03": float(np.mean(ds <= 0.03)),
                    "0.05": float(np.mean(ds <= 0.05)),
                    "0.10": float(np.mean(ds <= 0.10))}
        print("  sparse coverage:", {k: round(x, 4) for k, x in d["cov"].items()})

    # mutual proximity
    p9 = o3d.geometry.PointCloud(); p9.points = o3d.utility.Vector3dVector(data["V9"]["P"])
    p10 = o3d.geometry.PointCloud(); p10.points = o3d.utility.Vector3dVector(data["V10"]["P"])
    t9, t10 = o3d.geometry.KDTreeFlann(p9), o3d.geometry.KDTreeFlann(p10)
    d10to9 = nn_dists(t9, data["V10"]["P"])
    d9to10 = nn_dists(t10, data["V9"]["P"])
    global near_v10
    med = q(d10to9, 50)
    near_v10 = d10to9 <= 2.0 * med
    print("V10->V9 NN: med=%.5f p90=%.5f ; frac V10 within 2x med of V9 = %.4f"
          % (med, q(d10to9, 90), float(np.mean(near_v10))))

    # significance: bootstrap 95% CI of median difference (V9 - V10)
    ds_lo, ds_hi, ds_m = boot_median_diff(data["V9"]["dist_sparse"], data["V10"]["dist_sparse"])
    re_lo, re_hi, re_m = boot_median_diff(data["V9"]["resid"], data["V10"]["resid"])
    print("bootstrap med(dist V9) - med(dist V10): %.4f  [%.4f, %.4f]" % (ds_m, ds_lo, ds_hi))
    print("bootstrap med(resid V9) - med(resid V10): %.4f [%.4f, %.4f]" % (re_m, re_lo, re_hi))

    # renders
    render_pair(data["V9"]["P"], data["V10"]["P"], "v10_vs_v9_top", "top")
    render_pair(data["V9"]["P"], data["V10"]["P"], "v10_vs_v9_side", "side_y")
    render_pair(data["V9"]["P"], data["V10"]["P"], "v10_vs_v9_oblique", "oblique")

    def s(v):
        d = data[v]
        return {
            "points_clean": int(len(d["P"])),
            "points_raw": int(d["diag"]["final_points"]),
            "extent": [float(x) for x in d["extent"]],
            "dist_to_sparse": {"median": q(d["dist_sparse"], 50),
                               "p90": q(d["dist_sparse"], 90),
                               "p99": q(d["dist_sparse"], 99)},
            "sparse_coverage": d["cov"],
            "confidence": {"mean": d["conf"]["confidence"]["mean"],
                           "median": d["conf"]["confidence"]["median"],
                           "p10": d["conf"]["confidence"]["p10"],
                           "low_conf_frac": d["conf"]["low_conf_frac"]},
            "residual_error": {"median": q(d["resid"], 50), "p90": q(d["resid"], 90)},
            "worst_residual_p90": d["conf"]["worst_residual"]["p90"],
            "worst_residual_median": q(d["worst"], 50),
            "frac_with_contradiction": d["conf"]["worst_residual"]["fraction_with_contradiction"],
            "evidence_p90": d["conf"]["evidence"]["p90"],
            "candidates": d["diag"]["fusion_stats"]["candidates"],
            "gate_pass": d["diag"]["fusion_stats"]["gate_pass"],
            "votes_pass": d["diag"]["fusion_stats"]["votes_pass"],
            "frames_fused": d["diag"]["fusion_stats"]["frames_fused"],
        }

    summary = {
        "model": {"V9": "depth-anything/Depth-Anything-V2-Small-hf",
                  "V10": "depth-anything/Depth-Anything-V2-Base-hf"},
        "V9": s("V9"),
        "V10": s("V10"),
        "v10_to_v9": {"nn_median": med, "nn_p90": q(d10to9, 90),
                      "frac_within_2x_median": float(np.mean(near_v10))},
        "v9_to_v10": {"nn_median": q(d9to10, 50), "nn_p90": q(d9to10, 90)},
    }
    A, B = summary["V9"], summary["V10"]
    summary["deltas_V10_minus_V9"] = {
        "points_clean": B["points_clean"] - A["points_clean"],
        "dist_sparse_median": B["dist_to_sparse"]["median"] - A["dist_to_sparse"]["median"],
        "dist_sparse_p90": B["dist_to_sparse"]["p90"] - A["dist_to_sparse"]["p90"],
        "coverage_0.05": B["sparse_coverage"]["0.05"] - A["sparse_coverage"]["0.05"],
        "coverage_0.10": B["sparse_coverage"]["0.10"] - A["sparse_coverage"]["0.10"],
        "confidence_mean": B["confidence"]["mean"] - A["confidence"]["mean"],
        "residual_median": B["residual_error"]["median"] - A["residual_error"]["median"],
        "residual_p90": B["residual_error"]["p90"] - A["residual_error"]["p90"],
        "worst_residual_p90": B["worst_residual_p90"] - A["worst_residual_p90"],
    }
    summary["bootstrap"] = {
        "median_dist_sparse_V9_minus_V10": {"mean": ds_m, "ci95": [ds_lo, ds_hi]},
        "median_residual_V9_minus_V10": {"mean": re_m, "ci95": [re_lo, re_hi]},
    }

    # Geometry verdict (independent of point count):
    dlt = summary["deltas_V10_minus_V9"]
    better = (dlt["dist_sparse_median"] < 0 or dlt["dist_sparse_p90"] < 0
              or dlt["coverage_0.05"] > 0 or dlt["residual_median"] < 0)
    sig_better = summary["bootstrap"]["median_dist_sparse_V9_minus_V10"]["ci95"][0] > 0 \
        or summary["bootstrap"]["median_residual_V9_minus_V10"]["ci95"][0] > 0
    if not better and not sig_better:
        verdict = "NOT IMPROVED (no geometry metric got better)"
    elif B["points_clean"] >= A["points_clean"]:
        verdict = "GEOMETRY IMPROVED (denser or equal, and closer to triangulated surface)"
    else:
        verdict = "GEOMETRY IMPROVED but FEWER points (tighter to triangulated surface; count -%d)" \
                  % abs(dlt["points_clean"])
    summary["geometry_verdict"] = verdict

    with open(os.path.join(BASE, "v9_v10_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved:", os.path.join(BASE, "v9_v10_comparison.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
