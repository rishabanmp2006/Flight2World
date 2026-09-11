"""tests.test_pipeline

Deterministic tests for core.pipeline, core.diagnostics, and output metadata.

No COLMAP, no Depth Anything, no real video required. COLMAP and video
stages are exercised via mocks / synthetic fixtures.
"""

import json
import pathlib
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tests import run_module_tests

import core.config as cfg
import core.diagnostics as diag
import core.pipeline as pipe
from core.pipeline import Pipeline, PipelineConfig, build_metadata, write_metadata, load_metadata, stage_cached


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def test_stage_record_serialises():
    r = diag.StageRecord(stage="curation", status="ok", elapsed_s=0.12, details={"n_curated": 52})
    d = r.to_dict()
    assert d["stage"] == "curation" and d["status"] == "ok" and d["details"]["n_curated"] == 52

def test_diagnostics_collects_stages():
    d = diag.Diagnostics(run_id="test123", output_root="/tmp/x")
    d.add(diag.StageRecord(stage="a", status="ok", elapsed_s=0.1))
    d.add(diag.StageRecord(stage="b", status="ok", elapsed_s=0.2))
    assert len(d.stages) == 2
    assert d.stage("a").stage == "a"
    assert d.stage("missing") is None

def test_stage_timer_measures():
    import time
    with diag.StageTimer("dummy") as t:
        time.sleep(0.02)
    assert t.elapsed >= 0.015

def test_write_and_load_stage_diagnostics():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rec = diag.StageRecord(stage="colmap", status="ok", elapsed_s=1.0, details={"x": 1})
        p = diag.write_stage_diagnostics(td, rec)
        assert p.exists()
        loaded = diag.load_stage_diagnostics(td, "colmap")
        assert loaded["stage"] == "colmap" and loaded["details"]["x"] == 1
        assert diag.load_stage_diagnostics(td, "missing") is None

# ---------------------------------------------------------------------------
# metadata (coordinate system honesty)
# ---------------------------------------------------------------------------

def test_build_metadata_has_honest_coordinate_system():
    d = diag.Diagnostics(run_id="run_001", output_root="/tmp/out")
    d.add(diag.StageRecord(stage="extract_frames", status="ok", elapsed_s=0.1))
    payload = build_metadata(d)
    assert payload["coordinate_system"] == "COLMAP_relative"
    assert payload["georeferenced"] is False
    assert payload["metric_scale"] is None
    assert "run_id" in payload

def test_metadata_never_claims_metres():
    d = diag.Diagnostics(run_id="r", output_root="/tmp/o")
    payload = build_metadata(d)
    blob = json.dumps(payload).lower()
    assert "meter" not in blob
    assert "metre" not in blob

def test_metadata_extra_fields_merged():
    d = diag.Diagnostics(run_id="r", output_root="/tmp/o")
    payload = build_metadata(d, extra={"frame_count": 52, "custom": "x"})
    assert payload["frame_count"] == 52
    assert payload["custom"] == "x"

def test_write_and_load_metadata():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d = diag.Diagnostics(run_id="r", output_root=str(td))
        payload = build_metadata(d, extra={"frame_count": 10})
        p = write_metadata(td, payload)
        assert p.exists()
        loaded = load_metadata(td)
        assert loaded["coordinate_system"] == "COLMAP_relative"
        assert loaded["frame_count"] == 10
        assert loaded["metric_scale"] is None  # null in JSON -> None

def test_load_metadata_missing_returns_none():
    with tempfile.TemporaryDirectory() as td:
        assert load_metadata(td) is None

def test_config_constants_match_expectations():
    assert cfg.COORDINATE_SYSTEM == "COLMAP_relative"
    assert cfg.GEOREFERENCED is False
    assert cfg.METRIC_SCALE is None

# ---------------------------------------------------------------------------
# Pipeline stages independently callable
# ---------------------------------------------------------------------------

def test_pipeline_stage_curate_and_summarise():
    import core.colmap_io as cio
    from tests import SPARSE_TXT, FRAMES
    images = cio.load_colmap_images(SPARSE_TXT / "images.txt")
    with tempfile.TemporaryDirectory() as td:
        # Use real frames dir for existence filtering
        result, rec = pipe.stage_curate_and_summarise(images, FRAMES, output_root=td)
        assert rec.status == "ok"
        assert rec.details["n_curated"] > 0
        assert rec.details["n_curated"] <= len(images)
        # Sidecar written
        assert (Path(td) / "diagnostics" / "curation.json").exists()

def test_pipeline_stage_extract_frames_with_fake_video():
    # Synthetic video via cv2.VideoWriter
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(vp), fourcc, 10.0, (64, 48))
        for i in range(10):
            w.write(np.full((48, 64, 3), 128, dtype=np.uint8))
        w.release()
        out = td / "run"
        from core.video import ExtractionConfig
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        paths, rec = pipe.stage_extract_frames(vp, out, extraction_config=ec)
        assert rec.status == "ok"
        assert len(paths) > 0
        assert (out / "diagnostics" / "extract_frames.json").exists()

def test_pipeline_stage_run_colmap_with_fake_exe():
    # Fake colmap that creates expected outputs
    import sys
    fake_py = """\
import sys, pathlib
stage = sys.argv[1] if len(sys.argv) > 1 else ""
if stage == "feature_extractor":
    for i, a in enumerate(sys.argv):
        if a == "--database_path":
            pathlib.Path(sys.argv[i+1]).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(sys.argv[i+1]).write_text("fake")
elif stage == "mapper":
    for i, a in enumerate(sys.argv):
        if a == "--output_path":
            out = pathlib.Path(sys.argv[i+1]) / "0"
            out.mkdir(parents=True, exist_ok=True)
            (out / "images.bin").write_bytes(b"\\x00"*10)
elif stage == "model_converter":
    for i, a in enumerate(sys.argv):
        if a == "--output_path":
            out = pathlib.Path(sys.argv[i+1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cameras.txt").write_text("1 SIMPLE_RADIAL 1280 720 1167 640 360 -0.05\\n")
            (out / "images.txt").write_text("#\\n")
            (out / "points3D.txt").write_text("#\\n")
print(f"fake {stage} ok")
sys.exit(0)
"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "fake_colmap.py"
        p.write_text(fake_py)
        wrapper = td / "fake_colmap"
        wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {p} \"$@\"\n")
        wrapper.chmod(0o755)
        img = td / "frames"
        img.mkdir()
        # minimal image to satisfy COLMAP (not actually used by fake)
        cv2.imwrite(str(img / "frame_0001.jpg"), np.zeros((24, 32, 3), dtype=np.uint8))
        out = td / "run"
        from core.colmap_runner import ColmapRunConfig
        cfg_col = ColmapRunConfig(exe=str(wrapper))
        result, rec = pipe.stage_run_colmap(img, out, colmap_config=cfg_col)
        assert rec.status == "ok"
        assert (out / "diagnostics" / "colmap.json").exists()

# ---------------------------------------------------------------------------
# Pipeline orchestration / resumability / failure propagation
# ---------------------------------------------------------------------------

def test_pipeline_run_produces_metadata_json():
    with tempfile.TemporaryDirectory() as td:
        p = Pipeline(config=PipelineConfig(output_root=Path(td) / "run_001"))
        result = p.run(run_colmap=False)
        assert result.success
        assert (Path(td) / "run_001" / "metadata.json").exists()
        meta = json.loads((Path(td) / "run_001" / "metadata.json").read_text())
        assert meta["coordinate_system"] == "COLMAP_relative"
        assert meta["georeferenced"] is False

def test_pipeline_run_no_video_still_produces_metadata():
    with tempfile.TemporaryDirectory() as td:
        p = Pipeline(config=PipelineConfig(output_root=Path(td) / "run"))
        result = p.run()
        assert result.success
        assert result.metadata["coordinate_system"] == "COLMAP_relative"

def test_pipeline_run_with_video_extracts_frames():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(vp), fourcc, 10.0, (64, 48))
        for _ in range(10):
            w.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        w.release()
        out = td / "run"
        p = Pipeline(config=PipelineConfig(output_root=out), video_path=vp)
        result = p.run(run_colmap=False)
        assert result.success
        assert len(result.frame_paths) > 0
        assert (out / "metadata.json").exists()

def test_pipeline_resumability_extract_frames_cached():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(vp), fourcc, 10.0, (64, 48))
        for _ in range(10):
            w.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        w.release()
        out = td / "run"
        p1 = Pipeline(config=PipelineConfig(output_root=out), video_path=vp)
        r1 = p1.run(run_colmap=False)
        assert not r1.diagnostics.stage("extract_frames").details.get("cached")
        # Second run without force should hit cache
        p2 = Pipeline(config=PipelineConfig(output_root=out), video_path=vp)
        r2 = p2.run(run_colmap=False)
        assert r2.diagnostics.stage("extract_frames").details.get("cached") is True
        # With force=True it should re-extract (not cached)
        p3 = Pipeline(config=PipelineConfig(output_root=out, force=True), video_path=vp)
        r3 = p3.run(run_colmap=False)
        assert r3.diagnostics.stage("extract_frames").details.get("cached") is not True

def test_pipeline_failure_propagation_extract():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Non-existent video => extract fails
        p = Pipeline(config=PipelineConfig(output_root=td / "run"), video_path=td / "nope.mp4")
        result = p.run()
        assert not result.success
        assert result.error is not None
        # Metadata still written even on failure
        assert (td / "run" / "metadata.json").exists()

def test_pipeline_failure_propagation_colmap():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = td / "frames"
        img.mkdir()
        cv2.imwrite(str(img / "frame_0001.jpg"), np.zeros((24, 32, 3), dtype=np.uint8))
        out = td / "run"
        p = Pipeline(config=PipelineConfig(output_root=out), colmap_image_path=img)
        # Patch stage_run_colmap to simulate failure
        with mock.patch("core.pipeline.stage_run_colmap", side_effect=RuntimeError("COLMAP boom")):
            result = p.run(run_colmap=True)
            assert not result.success
            assert "boom" in result.error

def test_stage_cached_helper():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        assert not stage_cached(td, "extract_frames")
        # Write a fake ok sidecar
        (td / "diagnostics").mkdir(parents=True)
        (td / "diagnostics" / "extract_frames.json").write_text(json.dumps({"status": "ok"}))
        assert stage_cached(td, "extract_frames")
        # Failed stage is not considered cached
        (td / "diagnostics" / "curation.json").write_text(json.dumps({"status": "failed"}))
        assert not stage_cached(td, "curation")

def test_pipeline_diagnostics_stages_ordered():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(vp), fourcc, 10.0, (64, 48))
        for _ in range(10):
            w.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        w.release()
        out = td / "run"
        p = Pipeline(config=PipelineConfig(output_root=out), video_path=vp)
        result = p.run()
        stages = [s.stage for s in result.diagnostics.stages]
        assert "extract_frames" in stages

if __name__ == "__main__":
    run_module_tests(globals())
