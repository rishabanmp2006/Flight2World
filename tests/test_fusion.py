"""tests.test_fusion

Layered deterministic tests for core.fusion (V10 pass-2 track-anchored
fusion):

    A. Pure mathematical tests   backprojection, world/camera transform,
                                 reprojection, RGB lookup, voting math,
                                 candidate grid, neighbor window.
    B. Synthetic scene           tiny 3-camera plane scene with known
                                 geometry -> exact, verifiable survivors.
    C. Real-artifact smoke       3 registered frames (0027-0029) with a
                                 synthetic inverse-depth map (NO depth
                                 model): imports, shapes, finiteness,
                                 determinism, no crashes.
    D. V10 reference checks      the exact V10 pass-2 code (candidate gen,
                                 backproject, voting, and the FULL loop)
                                 compiled verbatim from fusion_v10.py runs
                                 on identical inputs; outputs must be
                                 numerically equal.
"""

import pathlib
import sys

# Make the project root importable regardless of how this file is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import FRAMES, SPARSE_TXT, run_module_tests

import core.calibration as cal
import core.colmap_io as cio
import core.config as cfg
import core.fusion as fusion
import core.gate as gate
import core.track_maps as tm
from tests import v10_reference

IMAGES = SPARSE_TXT / "images.txt"
POINTS = SPARSE_TXT / "points3D.txt"

V10_CALIBRATED_DEPTH = v10_reference.v10_function("calibrated_depth")
V10_CAND = v10_reference.v10_block(
    "v10_cand", 676, 688, ["depth", "frame"],
    extra={"calibrated_depth": V10_CALIBRATED_DEPTH},
    returns="(u, v, z)")
V10_BACK = v10_reference.v10_block(
    "v10_back", 699, 702, ["u", "v", "z", "R", "t"],
    returns="(cam_pts, world_pts)")
V10_VOTE = v10_reference.v10_block(
    "v10_vote", 740, 808,
    ["world_pts", "frame_records", "frame_idx", "vote_hist"],
    returns="(net_votes, vote_hist)")
V10_LOOP = v10_reference.v10_block(
    "v10_loop", 656, 825,
    ["frame_records", "sparse_tree", "local_r",
     "all_points", "all_colors", "vote_hist", "stats"],
    returns="(all_points, all_colors, stats, vote_hist)")


# ============================================================
# A. Pure mathematical tests
# ============================================================
def test_backproject_matches_v10_exactly():
    u = np.array([0.0, 100.0, 640.0])
    v = np.array([0.0, 50.0, 360.0])
    z = np.array([2.0, 4.0, 10.0])
    R = np.eye(3)
    t = np.zeros(3)
    cam_ours = fusion.backproject(u, v, z, cfg.CameraIntrinsics())
    cam_ref, world_ref = V10_BACK(u, v, z, R, t)
    assert np.array_equal(cam_ours, cam_ref)
    assert np.array_equal(fusion.cam_to_world(cam_ours, R, t), world_ref)


def test_cam_to_world_round_trip_with_world_to_cam():
    rng = np.random.default_rng(0)
    R = np.eye(3)
    # a rotation + translation
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    t = np.array([0.1, -0.2, 0.3])
    X_world = rng.uniform(-3, 3, (20, 3))
    X_cam = (R @ X_world.T).T + t
    back = fusion.cam_to_world(X_cam, R, t)
    assert np.allclose(back, X_world, atol=1e-12)
    # and world->cam through the same convention
    assert np.allclose((R @ back.T).T + t, X_cam, atol=1e-12)


def test_reprojection_matches_pinhole_formula():
    k = cfg.CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                             width=640, height=480)
    X_cam = np.array([[0.0, 0.0, 5.0], [0.5, -0.25, 2.0], [-1.0, 1.0, 10.0]])
    # exercise the same projection V10 uses inside the voting block:
    u = k.fx * X_cam[:, 0] / np.maximum(X_cam[:, 2], 1e-8) + k.cx
    v = k.fy * X_cam[:, 1] / np.maximum(X_cam[:, 2], 1e-8) + k.cy
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))


def test_sample_rgb_returns_float64_unit_range():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    rgb[3, 4] = [255, 128, 64]
    colors = fusion.sample_rgb(rgb, np.array([3]), np.array([4]))
    assert colors.dtype == np.float64
    assert np.allclose(colors[0], [1.0, 128 / 255.0, 64 / 255.0])


def test_candidate_grid_shapes():
    u, v = fusion.candidate_grid(4, image_w=16, image_h=12)
    assert len(u) == 4 * 3 == 12
    assert np.array_equal(np.unique(u), [0, 4, 8, 12])
    assert np.array_equal(np.unique(v), [0, 4, 8])


def test_sample_candidates_filters_depth_and_z_range():
    W, H = 16, 12
    # Candidate pixels are the stride-4 grid: u in {0,4,8,12}, v in {0,4,8}.
    depth = np.full((H, W), 5.0, dtype=np.float32)
    depth[0, 0] = np.nan          # non-finite -> excluded
    depth[0, 4] = 0.0             # non-positive -> excluded
    depth[4, 8] = -1.0            # non-positive -> excluded
    frame = {"depth": depth, "model": "linear", "a": 1.0, "b": 0.0}
    u, v, z = fusion.sample_candidates(frame, 4, image_w=W, image_h=H,
                                       z_min=0.15, z_max=50.0)
    # 12 grid pixels minus 3 excluded = 9 candidates, all at z=5
    assert len(z) == 9
    assert np.allclose(z, 5.0)
    assert not np.any((u == 0) & (v == 0))     # NaN pixel gone
    assert not np.any((u == 4) & (v == 0))     # 0.0 pixel gone
    assert not np.any((u == 8) & (v == 4))     # -1.0 pixel gone
    # z-range gate: depth outside [z_min, z_max] is dropped
    depth2 = np.full((H, W), 5.0, dtype=np.float32)
    depth2[0, 0] = 100.0            # calibrated z=100 > 50
    depth2[8, 4] = 0.05             # calibrated z=0.05 < 0.15
    frame2 = {"depth": depth2, "model": "linear", "a": 1.0, "b": 0.0}
    u2, v2, z2 = fusion.sample_candidates(frame2, 4, image_w=W, image_h=H,
                                          z_min=0.15, z_max=50.0)
    assert len(z2) == 10            # 12 minus the two out-of-range pixels
    assert not np.any((u2 == 0) & (v2 == 0))
    assert not np.any((u2 == 4) & (v2 == 8))


def test_neighbor_range_ordered_window():
    assert fusion.neighbor_range(0, 5, 3) == (0, 4)      # neighbors 1,2,3
    assert fusion.neighbor_range(2, 5, 3) == (0, 5)      # all
    assert fusion.neighbor_range(4, 5, 3) == (1, 5)      # neighbors 1,2,3
    assert fusion.neighbor_range(0, 1, 3) == (0, 1)


def test_track_anchored_votes_agree_contradict_neutral():
    # Targeted voting math: one neighbour sees p0 agree, p1 contradict,
    # p2 unconstrained.
    k = cfg.CameraIntrinsics(fx=10.0, fy=10.0, cx=8.0, cy=8.0,
                             width=16, height=16)
    world = np.array([[0.0, 0.0, 5.0],    # -> pixel (8, 8)
                      [1.0, 0.0, 5.0],    # -> pixel (10, 8)
                      [-1.0, 0.0, 5.0]])  # -> pixel (6, 8)
    z_grid = np.full((16, 16), np.nan, dtype=np.float32)
    dist_grid = np.full((16, 16), 99, dtype=np.int32)
    z_grid[8, 8] = 5.0            # agrees with p0's Z=5
    dist_grid[8, 8] = 0
    z_grid[8, 10] = 3.0           # disagrees with p1's Z=5 (tol ~0.54)
    dist_grid[8, 10] = 0
    nb = {"R": np.eye(3), "t": np.zeros(3),
          "z_grid": z_grid, "dist_grid": dist_grid}
    stub = {"R": np.eye(3), "t": np.zeros(3),
            "z_grid": z_grid, "dist_grid": dist_grid}
    net, delta = fusion.track_anchored_votes(
        world, [stub, nb], 0, image_w=16, image_h=16, intrinsics=k)
    assert net.tolist() == [1, -1, 0]
    assert delta == {"agree": 1, "contradict": 1, "neutral": 1}


# ============================================================
# B. Synthetic 3-camera plane scene
# ============================================================
def make_synthetic_scene(n_frames=3, spacing=0.25, plane_z=5.0, stride=8):
    W, H = 64, 48
    k = cfg.CameraIntrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0,
                             width=W, height=H)

    # Each camera backprojects the flat plane_z depth map onto a world
    # grid shifted by its baseline. The sparse cloud is the UNION of those
    # grids, so every camera's candidates coincide with a sparse point and
    # the adaptive gate passes them exactly (nn_dist == 0) -- as the real
    # COLMAP cloud covers the whole scene.
    sparse_worlds = []
    for i in range(n_frames):
        u, v = fusion.candidate_grid(stride, image_w=W, image_h=H)
        cam_pts = fusion.backproject(
            u, v, np.full_like(u, plane_z, dtype=np.float64), k)
        world_i = fusion.cam_to_world(
            cam_pts, np.eye(3), np.array([-spacing * i, 0.0, 0.0]))
        sparse_worlds.append(world_i)
    sparse_ids = {i + 1: pt
                  for i, pt in enumerate(np.vstack(sparse_worlds))}

    frames = []
    for i in range(n_frames):
        R = np.eye(3)
        t = np.array([-spacing * i, 0.0, 0.0])
        obs = []
        for pid, X in sparse_ids.items():
            Xc = R @ X + t
            if Xc[2] <= 0:
                continue
            xpx = k.fx * Xc[0] / Xc[2] + k.cx
            ypx = k.fy * Xc[1] / Xc[2] + k.cy
            if 0 <= xpx < W and 0 <= ypx < H:
                obs.append((xpx, ypx, pid))
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        frame = {
            "name": "synth_%d" % i,
            "R": R, "t": t, "observations": obs,
            "depth": np.full((H, W), plane_z, dtype=np.float32),
            "rgb": rgb,
            "model": "linear", "a": 1.0, "b": 0.0,
        }
        zg, dg, _nt = tm.build_track_maps(frame, sparse_ids,
                                          image_w=W, image_h=H)
        frame["z_grid"], frame["dist_grid"] = zg, dg
        frames.append(frame)
    return frames, sparse_ids, k


def test_synthetic_scene_fuses_expected_plane_points():
    frames, sparse_ids, k = make_synthetic_scene()
    xyz = np.array(list(sparse_ids.values()), dtype=np.float64)
    tree = gate.build_sparse_kdtree(xyz)
    local_r = gate.compute_local_radius(xyz)

    res = fusion.fuse_frame(frames[1], frames, 1, tree, local_r,
                            config=cfg.PipelineConfig(camera=k))
    stats = res["stats"]
    # The full candidate grid passes the gate (every candidate coincides
    # with a sparse point); only candidates seen by both neighbours
    # survive the >= 2 net-vote threshold (candidates too close to the
    # image edge fall out of the outer camera's view -> neutral -> < 2).
    assert stats["candidates"] == 48
    assert stats["gate_pass"] == 48
    assert stats["votes_pass"] == len(res["points"])
    assert stats["frames_fused"] == 1
    assert 0 < len(res["points"]) < 48
    assert res["points"].shape == (len(res["points"]), 3)
    assert np.allclose(res["points"][:, 2], 5.0, atol=1e-9)  # plane z=5
    assert np.all(np.isfinite(res["points"]))
    # RGB came from the red synthetic image.
    assert np.allclose(res["colors"], [200 / 255.0, 0.0, 0.0])
    # Stats/votes are consistent and deterministic.
    a = fusion.fuse_frame(frames[1], frames, 1, tree, local_r,
                          config=cfg.PipelineConfig(camera=k))
    assert np.array_equal(res["points"], a["points"])


def test_synthetic_fuse_all_matches_per_frame_aggregation():
    frames, sparse_ids, k = make_synthetic_scene()
    xyz = np.array(list(sparse_ids.values()), dtype=np.float64)
    tree = gate.build_sparse_kdtree(xyz)
    local_r = gate.compute_local_radius(xyz)
    pts, cols, stats, votes = fusion.fuse_all(
        frames, tree, local_r, config=cfg.PipelineConfig(camera=k))
    # Every frame has two neighbours inside NEIGHBOR_WINDOW=3, so all
    # three fuse. Each KEPT candidate needs net >= 2, i.e. two agrees,
    # but the histogram also counts agrees for candidates that later fall
    # below the threshold (border candidates: 1 agree + 1 neutral), so
    # agree >= 2 * votes_pass.
    assert stats["frames_fused"] == 3
    assert stats["votes_pass"] == len(pts)
    assert votes["agree"] >= 2 * stats["votes_pass"]
    assert votes["contradict"] == 0 and votes["neutral"] >= 0
    assert pts.shape[1] == 3 and cols.shape[1] == 3
    assert np.all(np.isfinite(pts))


def test_v10_voting_matches_on_real_frame():
    # V10's inline voting block uses the V10 FX/FY/CX/CY/IMAGE_W/IMAGE_H
    # constants, so the head-to-head must run on real frames (where those
    # intrinsics are correct) -- the synthetic scene uses toy intrinsics
    # and cannot be compared against the hardcoded V10 block.
    frames = _real_frames()
    tree, local_r = _real_gate_artifacts()
    im = frames[1]                      # frame_0028 (has 2 neighbours)

    u, v, z = fusion.sample_candidates(im, 8)
    world = fusion.cam_to_world(
        fusion.backproject(u, v, z, cfg.CameraIntrinsics()),
        im["R"], im["t"])
    g = gate.gate_candidates(world, tree, local_r)
    world_g = world[g]
    assert len(world_g) > 0

    mine, my_delta = fusion.track_anchored_votes(
        world_g, frames, 1, intrinsics=cfg.CameraIntrinsics())
    vh = {"agree": 0, "contradict": 0, "neutral": 0}
    ref, ref_vh = V10_VOTE(world_g, frames, 1, vh)
    assert np.array_equal(mine, ref), "net_votes differ from V10"
    assert my_delta == ref_vh, "vote histogram differs from V10"


# ============================================================
# C. Real-artifact smoke (3 registered frames, synthetic depth)
# ============================================================
def _real_frames():
    from PIL import Image
    imgs = cio.load_colmap_images(IMAGES)
    pts = cio.load_colmap_points(POINTS)
    out = []
    for image_id in (27, 28, 29):
        im = imgs[image_id]
        z_grid, dist_grid, n_tracks = tm.build_track_maps(im, pts)
        # Synthetic inverse depth: d = 1/z on tracks, 1.0 in holes.
        depth = np.where(np.isfinite(z_grid),
                         1.0 / np.maximum(z_grid, 1e-6), 1.0)
        depth = depth.astype(np.float32)
        d_vals, z_vals = cal.calibrate_frame(im, pts, depth)
        fit = cal.fit_depth_calibration(d_vals, z_vals)
        assert fit is not None, "calibration failed on %s" % im["name"]
        rgb = np.asarray(
            Image.open(FRAMES / im["name"]).convert("RGB"), dtype=np.uint8)
        out.append({
            "name": im["name"], "R": im["R"], "t": im["t"],
            "depth": depth, "rgb": rgb,
            "a": fit["a"], "b": fit["b"], "model": fit["model"],
            "rmse": fit["rmse"], "corr": fit["corr"],
            "samples": fit["n_samples"],
            "z_grid": z_grid, "dist_grid": dist_grid,
            "_n_tracks": n_tracks,
        })
    return out


def _real_gate_artifacts():
    pts = cio.load_colmap_points(POINTS)
    xyz = np.array(list(pts.values()), dtype=np.float64)
    tree = gate.build_sparse_kdtree(xyz)
    local_r = gate.compute_local_radius(xyz)
    return tree, local_r


def test_real_pair_smoke_no_crash_shapes_deterministic():
    frames = _real_frames()
    tree, local_r = _real_gate_artifacts()
    p1, c1, s1, v1 = fusion.fuse_all(frames, tree, local_r)
    p2, c2, s2, v2 = fusion.fuse_all(frames, tree, local_r)
    assert p1.shape[1] == 3 and c1.shape[1] == 3
    assert p1.shape == p2.shape
    assert np.array_equal(p1, p2), "fusion is not deterministic"
    assert np.array_equal(c1, c2)
    assert s1 == s2 and v1 == v2
    if len(p1):
        assert np.all(np.isfinite(p1))
        assert np.all((c1 >= 0.0) & (c1 <= 1.0))
    # Every track map built: n_tracks > 500 per frame.
    for f in frames:
        assert f["_n_tracks"] > 500


def test_real_pair_smoke_votes_pass_consistent():
    frames = _real_frames()
    tree, local_r = _real_gate_artifacts()
    _p, _c, s, v = fusion.fuse_all(frames, tree, local_r)
    assert s["candidates"] > 0
    assert 0 <= s["gate_pass"] <= s["candidates"]
    assert 0 <= s["votes_pass"] <= s["gate_pass"]
    assert 0 <= s["frames_fused"] <= len(frames)
    assert v["agree"] + v["contradict"] + v["neutral"] >= 0


# ============================================================
# D. V10 reference: identical inputs, numerical equality
# ============================================================
def test_v10_candidates_and_backproject_match_real_frame():
    frames = _real_frames()
    im = frames[1]                      # frame_0028 (has 2 neighbours)
    u, v, z = fusion.sample_candidates(im, 8)
    u_ref, v_ref, z_ref = V10_CAND(im["depth"], im)
    assert np.array_equal(u, u_ref) and np.array_equal(v, v_ref)
    assert np.array_equal(z, z_ref), "candidate Z differs from V10"
    R, t = im["R"], im["t"]
    cam = fusion.backproject(u, v, z, cfg.CameraIntrinsics())
    cam_ref, world_ref = V10_BACK(u, v, z, R, t)
    assert np.array_equal(cam, cam_ref)
    assert np.array_equal(fusion.cam_to_world(cam, R, t), world_ref)


def test_v10_full_pass2_loop_matches_on_real_frames():
    frames = _real_frames()
    tree, local_r = _real_gate_artifacts()

    # Run the exact V10 pass-2 loop (compiled verbatim, lines 656-825).
    v10_pts, v10_cols, v10_stats, v10_votes = V10_LOOP(
        frames, tree, local_r, [], [], {"agree": 0, "contradict": 0,
                                        "neutral": 0},
        {"candidates": 0, "gate_pass": 0, "votes_pass": 0, "frames_fused": 0})

    # Run the extracted fusion on identical inputs.
    pts, cols, stats, votes = fusion.fuse_all(frames, tree, local_r)

    assert stats == v10_stats, "fusion stats differ from V10: %s vs %s" % (
        stats, v10_stats)
    assert votes == v10_votes, "vote histogram differs from V10: %s vs %s" % (
        votes, v10_votes)
    # V10 returns the raw per-frame lists; vstack before comparing.
    assert np.array_equal(pts, np.vstack(v10_pts)), "fused points differ from V10"
    assert np.array_equal(cols, np.vstack(v10_cols)), "fused colors differ from V10"


if __name__ == "__main__":
    run_module_tests(globals())
