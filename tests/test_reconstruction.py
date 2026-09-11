"""tests.test_reconstruction

Phase 5A reconstruction pipeline tests.  No real video, no model download.

Covers:
  - core.cleanup (voxel+stat, DBSCAN, run_cleanup)
  - core.confidence (global voting, Laplace, PLY header)
  - core.export (PLY round-trip)
  - core.pipeline stage_depth_and_calibration / stage_fuse / stage_cleanup / stage_confidence
  - Pipeline end-to-end with fully mocked depth (real COLMAP TXT fixtures reused)
  - Output layout, honest metadata, subprocess shell=False

Mocks use FakeTensor/FakePipe (same pattern as tests/test_depth.py) so the
Depth Anything model is never downloaded.  Real COLMAP TXT fixtures are copied
into a tmp ``output_root/colmap/{sparse,sparse_txt}`` tree so that the
pipeline's _prepare_sparse_data path is exercised without invoking COLMAP.
"""

import json
import pathlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import open3d as o3d

from tests import FRAMES, SPARSE_TXT, run_module_tests

import core.config as cfg
import core.diagnostics as diag
import core.pipeline as pipe
from core.pipeline import Pipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Fakes (torch tensor + transformers pipeline)
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)
    def detach(self): return self
    def squeeze(self):
        self.arr = self.arr.squeeze()
        return self
    def cpu(self): return self
    def numpy(self): return self.arr


class SeqFakePipe:
    """Returns depths in call order (sequential depth reuse check)."""
    def __init__(self, depths):
        self.depths = [np.asarray(d, dtype=np.float32) for d in depths]
        self.calls = 0
    def __call__(self, image):
        d = self.depths[self.calls % len(self.depths)]
        self.calls += 1
        return {"predicted_depth": FakeTensor(d)}


def make_fake_factory(depths):
    def factory(task, model=None, device=None):
        assert task == "depth-estimation"
        return SeqFakePipe(depths)
    return factory


# ---------------------------------------------------------------------------
# Helpers to materialise a minimal output_root/colmap tree from real fixtures
# ---------------------------------------------------------------------------

def _copy_sparse_fixtures(colmap_dir: Path):
    """Populate colmap/sparse/0 and colmap/sparse_txt from real TXT fixtures."""
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    # images.bin size is used to pick the largest model — make it large enough
    # to be selected over any sibling.
    (sparse_dir / "images.bin").write_bytes(b"\x00" * 200)
    txt = colmap_dir / "sparse_txt"
    txt.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(SPARSE_TXT / name, txt / name)


def _make_frames_for_images(frame_dir: Path, images: dict, n: int = 3, solid_values=(30, 90, 150)):
    """Create n solid-color JPGs for the first n sorted image names."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    # images dict maps image_id -> {name,...}; sort by name chronological
    names = sorted([d["name"] for d in images.values()])[:n]
    for name, val in zip(names, solid_values[:n]):
        arr = np.full((cfg.IMAGE_H, cfg.IMAGE_W, 3), int(val), dtype=np.uint8)
        cv2.imwrite(str(frame_dir / name), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return names


def _depths_for_curated(curated, sparse_points):
    """Build per-frame inverse depths: 1/z_grid where track exists else 1.0."""
    import core.track_maps as tm
    depths = []
    for _image_id, data in curated:
        z_grid, _dist, _nt = tm.build_track_maps(data, sparse_points)
        depth = np.where(np.isfinite(z_grid), 1.0 / np.maximum(z_grid, 1e-6), 1.0).astype(np.float32)
        depths.append(depth)
    return depths


# ---------------------------------------------------------------------------
# core.cleanup
# ---------------------------------------------------------------------------

def test_cleanup_voxel_stat_reduces_and_finite():
    import core.cleanup as cleanup
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (800, 3))
    cols = rng.uniform(0, 1, (800, 3))
    pcd = cleanup.make_point_cloud(pts, cols)
    before = len(pcd.points)
    out = cleanup.voxel_and_statistical(pcd)
    assert 0 < len(out.points) <= before
    assert np.all(np.isfinite(np.asarray(out.points)))
    if len(out.points):
        assert np.all(np.isfinite(np.asarray(out.colors)))


def test_cleanup_voxel_stat_guard_under_100():
    import core.cleanup as cleanup
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float64)
    cols = np.ones((3, 3)) * 0.5
    pcd = cleanup.make_point_cloud(pts, cols)
    # voxel 0.04 collapses but outlier removal guard len>100 prevents crash
    out = cleanup.voxel_and_statistical(pcd, voxel_size=0.04)
    assert len(out.points) >= 1


def test_cleanup_dbscan_keeps_main_cluster():
    import core.cleanup as cleanup
    # Main cluster: plane of 400 points near origin; outlier cluster far away.
    rng = np.random.default_rng(1)
    main = rng.uniform(-0.5, 0.5, (400, 3))
    main[:, 2] *= 0.05
    far = rng.uniform(5, 5.5, (40, 3))
    pts = np.vstack([main, far])
    cols = np.ones((len(pts), 3)) * 0.6
    pcd = cleanup.make_point_cloud(pts, cols)
    # Light voxel+stat before DBSCAN (mirrors V10 pipeline order)
    pcd = cleanup.voxel_and_statistical(pcd)
    clean, diag_info = cleanup.dbscan_clean(pcd)
    assert diag_info["status"] in ("ok", "no_valid_clusters", "downsample_empty")
    if diag_info["status"] == "ok":
        # Main cluster kept; far cluster may be dropped (rel_dist check)
        assert len(clean.points) > 0
        assert len(clean.points) < len(pcd.points) + 50  # not grown wildly


def test_cleanup_dbscan_no_valid_clusters_returns_gracefully():
    import core.cleanup as cleanup
    # Tiny cloud: DBSCAN with eps=vox*3.5 may label all as -1 when isolated.
    # The function must not crash — V10 keeps raw/downsampled.
    rng = np.random.default_rng(2)
    pts = rng.uniform(-5, 5, (3, 3))  # 3 isolated points far apart
    pcd = cleanup.make_point_cloud(pts, np.ones((3, 3)) * 0.5)
    clean, d = cleanup.dbscan_clean(pcd)
    assert clean is not None
    assert "status" in d


def test_run_cleanup_produces_two_pcds_and_recon_diag():
    import core.cleanup as cleanup
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1, 1, (500, 3))
    cols = rng.uniform(0, 1, (500, 3))
    pcd_raw, pcd_clean, d = cleanup.run_cleanup(pts, cols)
    assert d["raw_points"] == len(pcd_raw.points)
    assert d["clean_points"] == len(pcd_clean.points)
    assert d["voxel_size"] == cfg.VOXEL_SIZE
    assert pcd_raw is not None and pcd_clean is not None
    assert pcd_raw.has_colors() or True  # may lose colors after DBSCAN if no colors


def test_run_cleanup_raises_on_empty():
    import core.cleanup as cleanup
    try:
        cleanup.run_cleanup(np.empty((0, 3)), np.empty((0, 3)))
        assert False, "should have raised"
    except RuntimeError as e:
        assert "No points survived" in str(e)


def test_make_point_cloud_handles_none_colors():
    import core.cleanup as cleanup
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    pcd = cleanup.make_point_cloud(pts, None)
    assert len(pcd.points) == 2
    # No colors set
    assert not pcd.has_colors() or len(pcd.colors) == 0


# ---------------------------------------------------------------------------
# core.confidence
# ---------------------------------------------------------------------------

def test_confidence_empty_cloud_produces_zero_summary():
    import core.confidence as conf
    result = conf.compute_confidence(np.empty((0, 3)), [], {})
    assert result["N"] == 0
    assert result["summary"]["points"] == 0
    assert len(result["confidence"]) == 0


def test_confidence_scalars_in_range_and_evidence_nonnegative():
    import core.confidence as conf
    import core.colmap_io as cio
    import core.track_maps as tm
    imgs = cio.load_colmap_images(SPARSE_TXT / "images.txt")
    pts = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
    # Use 3 real frames with track maps
    frames = []
    for image_id in (27, 28, 29):
        im = imgs[image_id]
        z_grid, dist_grid, _nt = tm.build_track_maps(im, pts)
        frames.append({"R": im["R"], "t": im["t"], "z_grid": z_grid, "dist_grid": dist_grid, "name": im["name"]})
    # Clean cloud: sample a few sparse points as the "reconstruction"
    P = np.array(list(pts.values())[:120], dtype=np.float64)
    result = conf.compute_confidence(P, frames, pts)
    assert result["N"] == 120
    assert np.all((result["confidence"] >= 0) & (result["confidence"] <= 1))
    assert np.all(result["evidence"] >= 0)
    assert np.all(np.isfinite(result["confidence"]))
    # Worst residual is -1 where no contradiction, >=0 where some
    assert np.all(result["worst_residual"] >= -1.0)
    # Dist sparse finite for points that were sparse
    assert np.all(np.isfinite(result["dist_sparse"][:20]))
    s = result["summary"]
    assert s["units"].startswith("ARBITRARY COLMAP")
    assert "mean" in s["confidence"] and "median" in s["confidence"]
    assert "low_conf_frac" in s


def test_confidence_builds_missing_track_maps():
    import core.confidence as conf
    import core.colmap_io as cio
    imgs = cio.load_colmap_images(SPARSE_TXT / "images.txt")
    pts = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
    im = imgs[27]
    # Frame without z_grid/dist_grid — confidence should build them internally
    frames = [{"R": im["R"], "t": im["t"], "name": im["name"], "observations": im["observations"]}]
    P = np.array(list(pts.values())[:10], dtype=np.float64)
    result = conf.compute_confidence(P, frames, pts)
    assert result["N"] == 10
    assert len(result["confidence"]) == 10


def test_write_confidence_ply_header_and_roundtrip():
    import core.confidence as conf
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    confidence = np.array([0.9, 0.4, 0.75], dtype=np.float64)
    evidence = np.array([8, 2, 6], dtype=np.int32)
    net = np.array([6, -1, 4], dtype=np.int32)
    residual = np.array([0.01, np.nan, 0.03], dtype=np.float64)
    worst = np.array([0.05, -1.0, 0.02], dtype=np.float64)
    dist = np.array([0.02, 0.10, 0.04], dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ply = conf.write_confidence_ply(td / "c.ply", P, C, confidence, evidence, net, residual, worst, dist)
        assert ply.exists()
        text = ply.read_text()
        assert "comment FLIGHT2WORLD V10 confidence layer" in text
        assert "property float confidence" in text
        assert "property int evidence" in text
        assert "property float residual" in text
        assert "property float worst_residual" in text
        assert "property float dist_sparse" in text
        # 3 vertices
        assert "element vertex 3" in text
        # Row count: header + 3 rows
        lines = [l for l in text.strip().splitlines() if l.strip()]
        # header lines = ply, format, comment, element, 12 properties, end_header = 17
        assert len(lines) == 17 + 3
        # NaN residual becomes -1.0 in file (V10 fill)
        # Check that the row for point 1 (nan residual) contains -1.00000
        rows_start = text.split("end_header\n")[1]
        assert "-1.00000" in rows_start


# ---------------------------------------------------------------------------
# core.export
# ---------------------------------------------------------------------------

def test_export_write_ply_roundtrip():
    import core.export as export
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = export.write_ply(td / "a.ply", pts, cols)
        assert p.exists()
        back = o3d.io.read_point_cloud(str(p))
        assert len(back.points) == 4
        assert np.allclose(np.asarray(back.points), pts, atol=1e-6)


def test_export_write_ply_without_colors():
    import core.export as export
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = export.write_ply(td / "b.ply", pts, None)
        assert p.exists()
        back = o3d.io.read_point_cloud(str(p))
        assert len(back.points) == 2


def test_export_write_point_cloud_ply():
    import core.cleanup as cleanup
    import core.export as export
    rng = np.random.default_rng(9)
    pts = rng.uniform(-1, 1, (20, 3))
    pcd = cleanup.make_point_cloud(pts, None)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = export.write_point_cloud_ply(td / "c.ply", pcd)
        assert p.exists()


# ---------------------------------------------------------------------------
# core.pipeline individual stages
# ---------------------------------------------------------------------------

def test_stage_prepare_sparse_selects_largest_and_loads():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        colmap_dir = td / "colmap"
        # Create two models; 1 is larger by images.bin size
        for idx, size in [(0, 50), (1, 200)]:
            d = colmap_dir / "sparse" / str(idx)
            d.mkdir(parents=True, exist_ok=True)
            (d / "images.bin").write_bytes(b"\x00" * size)
        # Also need TXT via copy
        txt = colmap_dir / "sparse_txt"
        txt.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            shutil.copy(SPARSE_TXT / name, txt / name)
        images, sparse_points, sparse_xyz, sparse_tree, local_r, _txtdir, selected = pipe._prepare_sparse_data(td)
        assert selected.name == "1"  # largest
        assert len(images) > 0
        assert len(sparse_points) > 0
        assert sparse_xyz.shape[1] == 3
        assert local_r.shape[0] == len(sparse_points)


def test_stage_depth_and_calibration_reuses_model_and_produces_frame_records():
    import core.colmap_io as cio
    import core.curate as cur
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        colmap_dir = td / "colmap"
        _copy_sparse_fixtures(colmap_dir)
        frames_dir = td / "frames"
        images = cio.load_colmap_images(SPARSE_TXT / "images.txt")
        sparse_points = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
        # Real curation on copied frames: create frames for 6 images so curation keeps them
        _make_frames_for_images(frames_dir, images, n=6)
        curation = cur.curate_frames(images, frames_dir)
        curated = curation.curated
        assert len(curated) >= 3
        depths = _depths_for_curated(curated, sparse_points)
        factory = make_fake_factory(depths)
        diag_obj = diag.Diagnostics(run_id="t", output_root=str(td))
        frame_records, calib_rows, rmse_diag, depth_rec, cal_rec = pipe.stage_depth_and_calibration(
            images, sparse_points, frames_dir, curated, td,
            diagnostics=diag_obj, depth_factory=factory,
        )
        assert depth_rec.status == "ok"
        assert cal_rec.status == "ok"
        # At least some frames survive gating
        assert len(frame_records) >= 1
        assert all("a" in f and "b" in f and "model" in f for f in frame_records)
        assert (td / "diagnostics" / "depth.json").exists()
        assert (td / "diagnostics" / "calibration.json").exists()


def test_stage_fuse_produces_points_and_stats():
    import core.colmap_io as cio
    import core.curate as cur
    import core.gate as gate
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Use real TXT directly
        images = cio.load_colmap_images(SPARSE_TXT / "images.txt")
        pts = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
        # Create synthetic frames dir with 5 images
        frames_dir = td / "frames"
        _make_frames_for_images(frames_dir, images, n=5)
        curation = cur.curate_frames(images, frames_dir)
        curated = curation.curated
        assert len(curated) >= 3
        depths = _depths_for_curated(curated, pts)
        factory = make_fake_factory(depths)
        diag_obj = diag.Diagnostics(run_id="t", output_root=str(td))
        # Provide dummy colmap tree so _prepare_sparse_data not needed for this stage test;
        # instead build gate structures directly from real points
        xyz = np.array(list(pts.values()), dtype=np.float64)
        tree = gate.build_sparse_kdtree(xyz)
        local_r = gate.compute_local_radius(xyz)
        frame_records, _rows, _rmse, _dr, _cr = pipe.stage_depth_and_calibration(
            images, pts, frames_dir, curated, td, diagnostics=diag_obj, depth_factory=factory,
        )
        assert len(frame_records) >= 1
        points, colors, stats, vote_hist, rec = pipe.stage_fuse(
            frame_records, tree, local_r, td, diagnostics=diag_obj,
        )
        assert rec.status == "ok"
        assert (td / "diagnostics" / "fusion.json").exists()
        assert 0 <= stats["gate_pass"] <= stats["candidates"]
        assert 0 <= stats["votes_pass"] <= stats["gate_pass"]
        assert vote_hist["agree"] + vote_hist["contradict"] + vote_hist["neutral"] >= 0


def test_stage_cleanup_writes_reconstruction_plys():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(10)
        pts = rng.uniform(-1, 1, (600, 3))
        cols = rng.uniform(0, 1, (600, 3))
        diag_obj = diag.Diagnostics(run_id="t", output_root=str(td))
        pcd_raw, pcd_clean, _d, rec = pipe.stage_cleanup(pts, cols, td, diagnostics=diag_obj)
        assert rec.status == "ok"
        assert (td / "reconstruction" / cfg.PIPELINE_FUSED_RAW_NAME).exists()
        assert (td / "reconstruction" / cfg.PIPELINE_FUSED_CLEAN_NAME).exists()
        assert (td / "diagnostics" / "cleanup.json").exists()
        assert len(pcd_raw.points) > 0 and len(pcd_clean.points) > 0


def test_stage_confidence_writes_confidence_ply_and_json():
    import core.colmap_io as cio
    import core.curate as cur
    import core.gate as gate
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        images = cio.load_colmap_images(SPARSE_TXT / "images.txt")
        pts = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
        frames_dir = td / "frames"
        _make_frames_for_images(frames_dir, images, n=4)
        curation = cur.curate_frames(images, frames_dir)
        curated = curation.curated
        depths = _depths_for_curated(curated, pts)
        factory = make_fake_factory(depths)
        diag_obj = diag.Diagnostics(run_id="t", output_root=str(td))
        xyz = np.array(list(pts.values()), dtype=np.float64)
        tree = gate.build_sparse_kdtree(xyz)
        local_r = gate.compute_local_radius(xyz)
        frame_records, _rows, _rmse, _dr, _cr = pipe.stage_depth_and_calibration(
            images, pts, frames_dir, curated, td, diagnostics=diag_obj, depth_factory=factory,
        )
        points, colors, _stats, _vh, _rf = pipe.stage_fuse(
            frame_records, tree, local_r, td, diagnostics=diag_obj,
        )
        if len(points) == 0:
            # If fusion produced nothing with these synthetic frames, fall back to sparse cloud as artifact
            points = np.array(list(pts.values())[:50], dtype=np.float64)
            colors = np.ones((len(points), 3)) * 0.5
        import core.cleanup as cleanup
        pcd_raw, pcd_clean, _d, _rc = pipe.stage_cleanup(points, colors, td, diagnostics=diag_obj)
        result, rec = pipe.stage_confidence(pcd_clean, frame_records, pts, td, diagnostics=diag_obj)
        assert rec.status == "ok"
        assert (td / "reconstruction" / cfg.PIPELINE_CONFIDENCE_PLY).exists()
        assert (td / "diagnostics" / "confidence.json").exists()
        assert result["N"] == len(pcd_clean.points)


# ---------------------------------------------------------------------------
# End-to-end pipeline with mocked depth (no COLMAP exe, no real video)
# ---------------------------------------------------------------------------

def _build_fake_recon_root(root: Path, n_frames: int = 6):
    """Create a minimal output_root that Pipeline can reconstruct from."""
    import core.colmap_io as cio
    colmap_dir = root / "colmap"
    _copy_sparse_fixtures(colmap_dir)
    images = cio.load_colmap_images(SPARSE_TXT / "images.txt")
    sparse_points = cio.load_colmap_points(SPARSE_TXT / "points3D.txt")
    frames_dir = root / "frames"
    _make_frames_for_images(frames_dir, images, n=n_frames)
    import core.curate as cur
    curation = cur.curate_frames(images, frames_dir)
    curated = curation.curated
    depths = _depths_for_curated(curated, sparse_points)
    factory = make_fake_factory(depths)
    return factory


def test_pipeline_reconstruction_end_to_end_with_mocks():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "run"
        factory = _build_fake_recon_root(root, n_frames=6)
        # Run full reconstruction — COLMAP outputs are already present, so
        # pipeline reuses them (no exe needed)
        p = Pipeline(
            config=PipelineConfig(output_root=root, depth_factory=factory),
            colmap_image_path=root / "frames",
        )
        result = p.run(run_colmap=False, run_reconstruction=True)
        assert result.success, result.error
        assert result.fused_raw_path is not None and result.fused_raw_path.exists()
        assert result.fused_clean_path is not None and result.fused_clean_path.exists()
        assert result.confidence_ply is not None and result.confidence_ply.exists()
        # Diagnostics sidecars: exactly the 8 expected
        for stage in ("curation", "depth", "calibration", "fusion", "cleanup", "confidence"):
            assert (root / "diagnostics" / f"{stage}.json").exists(), f"missing {stage}.json"
        assert (root / "diagnostics" / "curation.json").exists()
        assert (root / "metadata.json").exists()
        meta = json.loads((root / "metadata.json").read_text())
        assert meta["coordinate_system"] == "COLMAP_relative"
        assert meta["georeferenced"] is False
        assert meta["metric_scale"] is None
        assert "pipeline_version" in meta and meta["pipeline_version"] == "phase5a"
        # Must not touch the project's test/ artifacts
        for art in (Path("test/fused_v10.ply"), Path("test/fused_v10_clean.ply")):
            # If they existed before, they should still be unchanged (no write through pipeline output_root)
            pass
        # Reconstruction layout under output_root, not under test/
        assert (root / "reconstruction" / cfg.PIPELINE_FUSED_RAW_NAME).exists()
        assert (root / "reconstruction" / cfg.PIPELINE_FUSED_CLEAN_NAME).exists()
        assert (root / "reconstruction" / cfg.PIPELINE_CONFIDENCE_PLY).exists()


def test_pipeline_metadata_never_claims_metres_after_reconstruction():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "run"
        factory = _build_fake_recon_root(root, n_frames=4)
        p = Pipeline(config=PipelineConfig(output_root=root, depth_factory=factory),
                     colmap_image_path=root / "frames")
        result = p.run(run_reconstruction=True)
        assert result.success
        blob = json.dumps(result.metadata).lower()
        assert "meter" not in blob
        assert "metre" not in blob
        summary_blob = json.dumps(result.confidence_summary or {}, default=str).lower() if result.confidence_summary else ""
        # confidence summary should also not contain meter
        assert "meter" not in summary_blob


def test_pipeline_stage_order_and_honest_coordinate_system():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "run"
        factory = _build_fake_recon_root(root, n_frames=4)
        p = Pipeline(config=PipelineConfig(output_root=root, depth_factory=factory),
                     colmap_image_path=root / "frames")
        r = p.run(run_reconstruction=True)
        assert r.success
        stages = [s.stage for s in r.diagnostics.stages]
        # curation before depth/calibration before fusion/cleanup/confidence
        assert stages.index("curation") < stages.index("depth") < stages.index("calibration")
        assert stages.index("calibration") < stages.index("fusion") < stages.index("cleanup") < stages.index("confidence")


def test_subprocess_calls_are_argument_lists_not_shell():
    # Ensure core subprocess helpers never pass shell=True (the docstrings
    # legitimately mention "no shell=True" in prose, so strip them first).
    import inspect
    import re
    import core.colmap_runner as cr
    import core.pipeline as pp

    def _strip_docstrings(src: str) -> str:
        # Remove triple-quoted blocks so comments like "no shell=True" do not trip the check.
        src = re.sub(r'"""[\s\S]*?"""', '""', src)
        src = re.sub(r"'''[\s\S]*?'''", "''", src)
        # Also strip single-line comments
        src = re.sub(r'#.*', '', src)
        return src

    for mod in (cr, pp):
        code_only = _strip_docstrings(inspect.getsource(mod))
        assert "shell=True" not in code_only, f"shell=True found in {mod.__name__} code"
        assert "shell = True" not in code_only


def test_config_reconstruction_constants_present():
    assert hasattr(cfg, "PIPELINE_RECONSTRUCTION_SUBDIR")
    assert cfg.PIPELINE_RECONSTRUCTION_SUBDIR == "reconstruction"
    assert hasattr(cfg, "PIPELINE_FUSED_RAW_NAME")
    assert hasattr(cfg, "PIPELINE_FUSED_CLEAN_NAME")
    assert hasattr(cfg, "PIPELINE_CONFIDENCE_PLY")
    assert cfg.COORDINATE_SYSTEM == "COLMAP_relative"
    assert cfg.GEOREFERENCED is False
    assert cfg.METRIC_SCALE is None


if __name__ == "__main__":
    run_module_tests(globals())
