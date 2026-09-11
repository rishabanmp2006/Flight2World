"""tests.test_video

Deterministic tests for core.video.

No real drone video is required. Tests create a tiny synthetic video via
cv2.VideoWriter, then exercise inspect/extract/cache. No model download,
no COLMAP, no large allocation.
"""

import json
import pathlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tests import run_module_tests

import core.video as vid
from core.video import ExtractionConfig, VideoInfo
import core.config as cfg

# ---------------------------------------------------------------------------
# Helpers: synthetic video creation
# ---------------------------------------------------------------------------

def _make_video(path: Path, width=64, height=48, fps=10.0, n_frames=20, color=(10, 20, 30)):
    """Write a tiny synthetic video using cv2.VideoWriter (mp4v)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {path}")
    for i in range(n_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        # Vary one pixel so frames are not all identical (helps dedup checks)
        frame[0, 0] = [(color[0] + i) % 256, color[1], color[2]]
        writer.write(frame)
    writer.release()
    return path

# ---------------------------------------------------------------------------
# VideoInfo / ExtractionConfig
# ---------------------------------------------------------------------------

def test_video_info_to_dict():
    info = VideoInfo(path=Path("/tmp/x.mp4"), width=1280, height=720, fps=30.0, frame_count=100, duration=3.33, codec="h264")
    d = info.to_dict()
    assert d["width"] == 1280 and d["codec"] == "h264" and isinstance(d["path"], str)


def test_extraction_config_cache_key_stable():
    a = ExtractionConfig(target_fps=1.0, target_width=1280, target_height=720)
    b = ExtractionConfig(target_fps=1.0, target_width=1280, target_height=720)
    assert a.cache_key() == b.cache_key()
    c = ExtractionConfig(target_fps=2.0, target_width=1280, target_height=720)
    assert a.cache_key() != c.cache_key()

# ---------------------------------------------------------------------------
# inspect_video
# ---------------------------------------------------------------------------

def test_inspect_video_missing_raises():
    try:
        vid.inspect_video("/tmp/does_not_exist_12345.mp4")
        assert False, "should have raised"
    except FileNotFoundError:
        pass

def test_inspect_video_synthetic():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "test.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=20)
        info = vid.inspect_video(vp)
        assert info.width == 64
        assert info.height == 48
        assert abs(info.fps - 10.0) < 0.5
        assert info.frame_count >= 18  # mp4v may report slightly off
        assert info.duration > 0


def test_inspect_garbage_file_raises():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "garbage.mp4"
        p.write_bytes(b"not a video")
        try:
            vid.inspect_video(p)
            assert False, "should have raised"
        except RuntimeError:
            pass

# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

def test_extract_frames_basic():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=20)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=2.0, target_width=32, target_height=24, jpeg_quality=95)
        result = vid.extract_frames(vp, out, config=ec)
        # 20 frames @ 10 fps = 2s duration; at 2 fps we expect ~5 frames
        assert len(result.frame_paths) >= 4
        assert not result.skipped
        for p in result.frame_paths:
            assert p.exists()
            img = cv2.imread(str(p))
            assert img is not None
            assert img.shape[1] == 32 and img.shape[0] == 24
        # Deterministic naming
        assert result.frame_paths[0].name == "frame_0001.jpg"


def test_extract_frames_no_resize_when_same_size():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=10.0, target_width=64, target_height=48)
        result = vid.extract_frames(vp, out, config=ec)
        assert len(result.frame_paths) >= 8
        for p in result.frame_paths:
            img = cv2.imread(str(p))
            assert img.shape[1] == 64 and img.shape[0] == 48


def test_extract_frames_deterministic():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=20)
        out1 = td / "frames1"
        out2 = td / "frames2"
        ec = ExtractionConfig(target_fps=2.0, target_width=32, target_height=24)
        r1 = vid.extract_frames(vp, out1, config=ec)
        r2 = vid.extract_frames(vp, out2, config=ec)
        assert len(r1.frame_paths) == len(r2.frame_paths)
        # Same extracted pixel content (lossy but deterministic)
        for p1, p2 in zip(r1.frame_paths, r2.frame_paths):
            a = cv2.imread(str(p1))
            b = cv2.imread(str(p2))
            assert np.array_equal(a, b)


def test_extract_frames_missing_video_raises():
    with tempfile.TemporaryDirectory() as td:
        try:
            vid.extract_frames("/tmp/no_such_video.mp4", Path(td) / "out")
            assert False
        except FileNotFoundError:
            pass


def test_frames_cache_valid_after_extract():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        vid.extract_frames(vp, out, config=ec)
        assert vid.frames_cache_valid(out, vp, ec) is True
        # Different config => invalid
        ec2 = ExtractionConfig(target_fps=2.0, target_width=32, target_height=24)
        assert vid.frames_cache_valid(out, vp, ec2) is False
        # Different video path => invalid
        assert vid.frames_cache_valid(out, td / "other.mp4", ec) is False


def test_extract_frames_cache_hit_skips_rewrite():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        r1 = vid.extract_frames(vp, out, config=ec)
        mtime1 = (out / ".cache_manifest.json").stat().st_mtime
        # Second call with same config and overwrite=False should hit cache
        r2 = vid.extract_frames(vp, out, config=ec)
        assert r2.skipped is True
        assert len(r2.frame_paths) == len(r1.frame_paths)
        # Manifest mtime should not change on cache hit
        mtime2 = (out / ".cache_manifest.json").stat().st_mtime
        assert mtime1 == mtime2


def test_extract_frames_overwrite_forces_rewrite():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        vid.extract_frames(vp, out, config=ec)
        ec_over = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24, overwrite=True)
        r2 = vid.extract_frames(vp, out, config=ec_over)
        assert r2.skipped is False


def test_cache_invalid_when_frame_deleted():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        out = td / "frames"
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        vid.extract_frames(vp, out, config=ec)
        # Delete one frame => cache invalid
        list(out.glob("frame_*.jpg"))[0].unlink()
        assert vid.frames_cache_valid(out, vp, ec) is False


def test_extract_frames_auto_creates_output_dir():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vp = td / "in.mp4"
        _make_video(vp, width=64, height=48, fps=10.0, n_frames=10)
        nested = td / "a" / "b" / "frames"
        ec = ExtractionConfig(target_fps=5.0, target_width=32, target_height=24)
        result = vid.extract_frames(vp, nested, config=ec)
        assert nested.exists()
        assert len(result.frame_paths) > 0


def test_inspect_real_benchmark_frame_is_jpeg():
    # Lightweight real-artifact check: frame_0001.jpg must be readable
    p = Path(__file__).resolve().parent.parent / "test" / "frames" / "frame_0001.jpg"
    if not p.exists():
        return  # skip when benchmark not present
    img = cv2.imread(str(p))
    assert img is not None
    assert img.shape[1] == cfg.IMAGE_W and img.shape[0] == cfg.IMAGE_H


if __name__ == "__main__":
    run_module_tests(globals())
