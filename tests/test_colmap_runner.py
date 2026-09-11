"""tests.test_colmap_runner

Deterministic tests for core.colmap_runner.

No real COLMAP invocation is performed. A tiny fake "colmap" Python script
is used as the executable so command construction, path handling, and error
propagation can be verified without a GPU, database, or image set.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests import run_module_tests

import core.colmap_runner as cr
from core.colmap_runner import ColmapError
import core.config as cfg

# ---------------------------------------------------------------------------
# Helpers: fake colmap executable
# ---------------------------------------------------------------------------

FAKE_COLMAP_PY = """\
import sys, pathlib, os
stage = sys.argv[1] if len(sys.argv) > 1 else ""
# Simulate success for known stages; fail stage name contains "fail"
if "fail" in stage:
    print("fake error", file=sys.stderr)
    sys.exit(1)
# feature_extractor creates a fake database
if stage == "feature_extractor":
    for i, a in enumerate(sys.argv):
        if a == "--database_path" and i + 1 < len(sys.argv):
            pathlib.Path(sys.argv[i+1]).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(sys.argv[i+1]).write_text("fake db")
# mapper creates sparse/0/images.bin
if stage == "mapper":
    for i, a in enumerate(sys.argv):
        if a == "--output_path" and i + 1 < len(sys.argv):
            out = pathlib.Path(sys.argv[i+1]) / "0"
            out.mkdir(parents=True, exist_ok=True)
            (out / "images.bin").write_bytes(b"\\x00" * 10)
            (out / "cameras.bin").write_bytes(b"\\x00" * 10)
# model_converter creates cameras.txt
if stage == "model_converter":
    for i, a in enumerate(sys.argv):
        if a == "--output_path" and i + 1 < len(sys.argv):
            out = pathlib.Path(sys.argv[i+1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cameras.txt").write_text("1 SIMPLE_RADIAL 1280 720 1167 640 360 -0.05\\n")
            (out / "images.txt").write_text("# dummy\\n")
            (out / "points3D.txt").write_text("# dummy\\n")
print(f"fake {stage} ok")
sys.exit(0)
"""

def _fake_exe(tmpdir: Path) -> str:
    p = tmpdir / "fake_colmap.py"
    p.write_text(FAKE_COLMAP_PY)
    p.chmod(0o755)
    # Use "python fake_colmap.py" via a small shell wrapper or just python
    # For simplicity, return a wrapper path that invokes python + script
    wrapper = tmpdir / "fake_colmap"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {p} \"$@\"\n")
    wrapper.chmod(0o755)
    return str(wrapper)

# ---------------------------------------------------------------------------
# resolve / availability
# ---------------------------------------------------------------------------

def test_resolve_explicit_path_exists():
    with tempfile.TemporaryDirectory() as td:
        fake = _fake_exe(Path(td))
        resolved = cr.resolve_colmap_exe(fake)
        assert resolved == fake

def test_resolve_explicit_path_missing_raises():
    try:
        cr.resolve_colmap_exe("/tmp/does_not_exist_colmap_12345")
        assert False
    except FileNotFoundError:
        pass

def test_resolve_bare_name_on_path():
    # Use "python3" or "sh" as a known bare name on PATH
    resolved = cr.resolve_colmap_exe("sh")
    assert "sh" in resolved

def test_resolve_bare_name_not_on_path_raises():
    try:
        cr.resolve_colmap_exe("definitely_not_a_real_exe_12345")
        assert False
    except FileNotFoundError:
        pass

def test_check_colmap_available_with_fake():
    with tempfile.TemporaryDirectory() as td:
        fake = _fake_exe(Path(td))
        resolved = cr.check_colmap_available(fake)
        assert resolved == fake

def test_check_colmap_available_missing_raises():
    try:
        cr.check_colmap_available("/tmp/no_such_colmap_xyz")
        assert False
    except FileNotFoundError:
        pass

# ---------------------------------------------------------------------------
# Command construction (no shell, correct flags)
# ---------------------------------------------------------------------------

def test_feature_extraction_command_construction():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "db" / "database.db"
        img = td / "imgs"
        img.mkdir()
        result = cr.feature_extraction(db, img, exe=fake)
        assert result.succeeded
        assert "--database_path" in result.command
        assert "--ImageReader.camera_model" in result.command
        assert "SIMPLE_RADIAL" in result.command
        assert "--ImageReader.single_camera" in result.command
        assert db.exists()

def test_feature_extraction_uses_config_defaults():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        img = td / "imgs"
        img.mkdir()
        result = cr.feature_extraction(db, img, exe=fake, camera_model="PINHOLE", single_camera=0)
        assert "PINHOLE" in result.command
        assert "0" in result.command  # single_camera 0

def test_feature_extraction_extra_args_appended():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        img = td / "imgs"
        img.mkdir()
        result = cr.feature_extraction(db, img, exe=fake, extra_args=["--SiftExtraction.max_image_size", "3200"])
        assert "--SiftExtraction.max_image_size" in result.command

def test_sequential_matching_command_construction():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        db.write_text("x")
        result = cr.sequential_matching(db, exe=fake, overlap=10)
        assert result.succeeded
        assert "--SequentialMatching.overlap" in result.command
        assert "10" in result.command

def test_sequential_matching_custom_overlap():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        db.write_text("x")
        result = cr.sequential_matching(db, exe=fake, overlap=5)
        assert "5" in result.command

def test_sparse_mapping_command_construction():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        db.write_text("x")
        img = td / "imgs"
        img.mkdir()
        out = td / "sparse"
        result = cr.sparse_mapping(db, img, out, exe=fake)
        assert result.succeeded
        assert "--output_path" in result.command
        assert (out / "0" / "images.bin").exists()

def test_sparse_mapping_ba_flags():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        db.write_text("x")
        img = td / "imgs"
        img.mkdir()
        result = cr.sparse_mapping(db, img, td / "sparse2", exe=fake,
                                   min_num_matches=20, ba_refine_extra_params=0)
        assert "20" in result.command
        # 0 for ba_refine_extra_params should appear
        idx = result.command.index("--Mapper.ba_refine_extra_params")
        assert result.command[idx + 1] == "0"

def test_model_converter_command_construction():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        src = td / "sparse" / "0"
        src.mkdir(parents=True)
        (src / "images.bin").write_bytes(b"\x00")
        out = td / "sparse_txt"
        result = cr.model_converter(src, out, exe=fake)
        assert result.succeeded
        assert (out / "cameras.txt").exists()
        assert "--output_type" in result.command
        assert "TXT" in result.command

def test_model_converter_binary_output_type():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        src = td / "sparse" / "0"
        src.mkdir(parents=True)
        result = cr.model_converter(src, td / "out_bin", output_type="BIN", exe=fake)
        assert "BIN" in result.command

# ---------------------------------------------------------------------------
# Error propagation (subprocess failure -> ColmapError)
# ---------------------------------------------------------------------------

def test_feature_extraction_failure_raises_ColmapError():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Create a failing fake: map "feature_extractor" to a name containing "fail"
        # Simpler: mock _run_colmap to return failure
        with mock.patch("core.colmap_runner._run_colmap") as m:
            m.return_value = cr.ColmapCommandResult(command=["colmap","feature_extractor"], returncode=1, stdout="", stderr="boom")
            try:
                cr.feature_extraction(td / "db.db", td / "imgs", exe="colmap")
                assert False
            except ColmapError as e:
                assert e.stage == "feature_extraction"
                assert "boom" in str(e)

def test_sequential_matching_failure_raises():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with mock.patch("core.colmap_runner._run_colmap") as m:
            m.return_value = cr.ColmapCommandResult(command=["colmap","sequential_matcher"], returncode=1, stdout="", stderr="fail seq")
            try:
                cr.sequential_matching(td / "db.db", exe="colmap")
                assert False
            except ColmapError as e:
                assert e.stage == "sequential_matching"

def test_sparse_mapping_failure_raises():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with mock.patch("core.colmap_runner._run_colmap") as m:
            m.return_value = cr.ColmapCommandResult(command=["colmap","mapper"], returncode=1, stdout="", stderr="fail map")
            try:
                cr.sparse_mapping(td / "db.db", td / "imgs", td / "sparse", exe="colmap")
                assert False
            except ColmapError:
                pass

def test_model_converter_failure_raises():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with mock.patch("core.colmap_runner._run_colmap") as m:
            m.return_value = cr.ColmapCommandResult(command=["colmap","model_converter"], returncode=1, stdout="", stderr="fail conv")
            try:
                cr.model_converter(td / "a", td / "b", exe="colmap")
                assert False
            except ColmapError as e:
                assert e.stage == "model_converter"

# ---------------------------------------------------------------------------
# Safe subprocess (no shell)
# ---------------------------------------------------------------------------

def test_commands_are_lists_not_shell_strings():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        img = td / "imgs"
        img.mkdir()
        result = cr.feature_extraction(db, img, exe=fake)
        assert isinstance(result.command, list)
        assert all(isinstance(c, str) for c in result.command)
        # Paths with spaces must not be shell-split
        spaced = Path(td) / "my imgs"
        spaced.mkdir()
        db2 = td / "my db" / "database.db"
        result2 = cr.feature_extraction(db2, spaced, exe=fake)
        assert result2.succeeded
        assert str(spaced) in result2.command

# ---------------------------------------------------------------------------
# High-level run_colmap (mocked stages)
# ---------------------------------------------------------------------------

def test_run_colmap_orchestrates_stages_in_order():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        img = td / "frames"
        img.mkdir()
        (img / "frame_0001.jpg").write_bytes(b"\x00")
        out = td / "output"
        from core.colmap_runner import ColmapRunConfig
        cfg_obj = ColmapRunConfig(exe=fake)
        result = cr.run_colmap(img, out, config=cfg_obj, export_txt=True)
        assert "feature_extraction" in result.stages
        assert "sequential_matching" in result.stages
        assert "sparse_mapping" in result.stages
        assert result.database_path.exists()
        assert result.sparse_dir.exists()

def test_run_colmap_propagates_stage_failure():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = td / "frames"
        img.mkdir()
        out = td / "output"
        with mock.patch("core.colmap_runner.feature_extraction") as m:
            m.side_effect = ColmapError("feature_extraction",
                cr.ColmapCommandResult(command=["x"], returncode=1, stdout="", stderr="fail"))
            try:
                cr.run_colmap(img, out)
                assert False
            except ColmapError:
                pass

def test_config_defaults_match_validated_benchmark():
    assert cfg.COLMAP_CAMERA_MODEL == "SIMPLE_RADIAL"
    assert cfg.COLMAP_SINGLE_CAMERA == 1
    assert cfg.COLMAP_SEQUENTIAL_OVERLAP == 10
    from core.colmap_runner import ColmapRunConfig
    c = ColmapRunConfig()
    assert c.camera_model == "SIMPLE_RADIAL"
    assert c.single_camera == 1
    assert c.sequential_overlap == 10

# ---------------------------------------------------------------------------
# Regression: COLMAP 4.1.1 compatibility — unsupported GPU flags must NOT be emitted
# ---------------------------------------------------------------------------

def test_feature_extraction_does_not_emit_SiftExtraction_use_gpu():
    """Regression: COLMAP 4.1.1 has no --SiftExtraction.use_gpu (see base_option_manager.cc:265)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        img = td / "imgs"
        img.mkdir()
        result = cr.feature_extraction(db, img, exe=fake)
        assert "--SiftExtraction.use_gpu" not in result.command, result.command
        assert "--FeatureExtraction.use_gpu" in result.command

def test_sequential_matching_does_not_emit_SiftMatching_use_gpu():
    """Regression: COLMAP 4.1.1 has no --SiftMatching.use_gpu."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        db = td / "database.db"
        db.write_text("x")
        result = cr.sequential_matching(db, exe=fake)
        assert "--SiftMatching.use_gpu" not in result.command, result.command
        assert "--FeatureMatching.use_gpu" in result.command

def test_no_unsupported_Sift_gpu_flags_in_run_colmap():
    """End-to-end: run_colmap must not emit either deprecated Sift* gpu flag."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = _fake_exe(td)
        img = td / "frames"
        img.mkdir()
        (img / "frame_0001.jpg").write_bytes(b"\x00")
        out = td / "output"
        from core.colmap_runner import ColmapRunConfig
        result = cr.run_colmap(img, out, config=ColmapRunConfig(exe=fake), export_txt=True)
        all_flags = []
        for r in result.stages.values():
            all_flags.extend(r.command)
        assert "--SiftExtraction.use_gpu" not in all_flags
        assert "--SiftMatching.use_gpu" not in all_flags

if __name__ == "__main__":
    run_module_tests(globals())
