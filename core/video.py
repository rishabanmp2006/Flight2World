"""flight2world.core.video

Lightweight video utilities for the FLIGHT2WORLD pipeline.

Uses OpenCV (cv2) for inspection/extraction and, when available,
ffprobe for richer metadata. No extra video framework is introduced.

Responsibilities:
  - :class:`VideoInfo`  structured metadata for a video file
  - :func:`inspect_video`  probe width/height/fps/frame_count/duration/codec
  - :func:`extract_frames`  deterministic FPS-controlled extraction to a
    directory with configurable resize and JPEG quality
  - :func:`frames_cache_valid`  cache-friendly check so extraction is
    skipped when cached frames already match the requested config
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import (
    PIPELINE_CACHE_MANIFEST,
    VIDEO_FRAME_PATTERN,
    VIDEO_JPEG_QUALITY,
    VIDEO_TARGET_HEIGHT,
    VIDEO_TARGET_WIDTH,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    """Structured metadata for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds (frame_count / fps or container duration)
    codec: Optional[str] = None  # fourcc string when available

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(d["path"])
        return d


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for :func:`extract_frames`."""

    target_fps: float = 1.0
    target_width: int = VIDEO_TARGET_WIDTH
    target_height: int = VIDEO_TARGET_HEIGHT
    frame_pattern: str = VIDEO_FRAME_PATTERN  # must contain {idx:04d}
    jpeg_quality: int = VIDEO_JPEG_QUALITY
    overwrite: bool = False  # if False, cache check can skip extraction

    def cache_key(self) -> dict:
        """Stable dict for cache-manifest comparison (excludes overwrite)."""
        return {
            "target_fps": float(self.target_fps),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "frame_pattern": str(self.frame_pattern),
            "jpeg_quality": int(self.jpeg_quality),
        }


@dataclass(frozen=True)
class ExtractionResult:
    """Result of :func:`extract_frames`."""

    frame_paths: list[Path]
    n_extracted: int
    skipped: bool  # True if served from cache
    video_info: VideoInfo
    output_dir: Path


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def _fourcc_str(fourcc: int) -> Optional[str]:
    if fourcc == 0:
        return None
    try:
        chars = [chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]
        s = "".join(chars).strip()
        return s if s and s.isprintable() else None
    except Exception:
        return None


def inspect_video(path: str | Path) -> VideoInfo:
    """Probe video metadata via OpenCV (and ffprobe when available for codec).

    Raises:
        FileNotFoundError: if *path* does not exist.
        RuntimeError: if OpenCV cannot open the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    # Try ffprobe for codec (best-effort, never fails the call)
    codec: Optional[str] = None
    if shutil.which("ffprobe") is not None:
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=nw=1:nk=1",
                    str(path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                codec = out.stdout.strip().splitlines()[0].strip() or None
        except Exception:
            pass

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if codec is None:
            codec = _fourcc_str(fourcc)
    finally:
        cap.release()

    # Duration from container when available via cap, else frame_count/fps
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        duration = 0.0

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=codec,
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _manifest_path(output_dir: Path) -> Path:
    return output_dir / PIPELINE_CACHE_MANIFEST


def frames_cache_valid(
    output_dir: str | Path,
    video_path: str | Path,
    config: ExtractionConfig,
) -> bool:
    """Return True if cached frames in *output_dir* match the requested config.

    The check requires:
      - manifest file exists and is valid JSON,
      - ``video_path`` matches,
      - extraction config cache key matches,
      - every listed frame file still exists.
    Any failure => cache is invalid (caller should re-extract).
    """
    output_dir = Path(output_dir)
    manifest = _manifest_path(output_dir)
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return False
    if str(data.get("video_path")) != str(Path(video_path)):
        return False
    if data.get("extraction_config") != config.cache_key():
        return False
    frame_list = data.get("frames") or []
    if not frame_list:
        return False
    for rel in frame_list:
        if not (output_dir / rel).exists():
            return False
    return True


def _write_manifest(
    output_dir: Path,
    video_path: Path,
    config: ExtractionConfig,
    frame_rel_paths: list[str],
    video_info: VideoInfo,
) -> None:
    payload = {
        "video_path": str(video_path),
        "video_info": video_info.to_dict(),
        "extraction_config": config.cache_key(),
        "frames": frame_rel_paths,
    }
    _manifest_path(output_dir).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _resize_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    # Downscale uses INTER_AREA (better quality); upscale uses INTER_LINEAR
    interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interp)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    config: Optional[ExtractionConfig] = None,
) -> ExtractionResult:
    """Extract frames from *video_path* into *output_dir* at ``config.target_fps``.

    - Deterministic naming: ``frame_{idx:04d}.jpg`` (1-based) via
      ``config.frame_pattern``.
    - Deterministic frame selection: source frames are sampled by timestamp,
      not by wall clock. For each target timestamp ``k/target_fps`` the
      nearest source frame is selected via ``round(ts * src_fps)``.
    - Resize to ``target_width × target_height`` when they differ from the
      source.
    - JPEG quality controlled by ``config.jpeg_quality``.
    - Cache-friendly: when ``config.overwrite is False`` and
      :func:`frames_cache_valid` is True, no frames are rewritten and
      ``skipped=True`` is returned.

    Raises:
        FileNotFoundError: if *video_path* does not exist.
        RuntimeError: if the video cannot be opened or has no FPS.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if config is None:
        config = ExtractionConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = inspect_video(video_path)
    if info.fps <= 0:
        raise RuntimeError(f"video has no FPS metadata: {video_path}")
    if info.frame_count <= 0:
        # Some containers report 0 frame_count; fall back to duration*fps estimate
        # and let the read loop handle it.
        pass

    # Cache check
    if not config.overwrite and frames_cache_valid(output_dir, video_path, config):
        manifest = json.loads(_manifest_path(output_dir).read_text())
        frame_paths = [output_dir / rel for rel in manifest["frames"]]
        return ExtractionResult(
            frame_paths=frame_paths,
            n_extracted=len(frame_paths),
            skipped=True,
            video_info=info,
            output_dir=output_dir,
        )

    # Read all frames into a list of (index, frame) to allow timestamp sampling.
    # For very long videos this could be memory-heavy; we instead do a
    # two-pass selection: first enumerate frame indices to select, then read.
    # Simpler and streaming-friendly: read once and pick on the fly.
    #
    # Deterministic sampling: build the set of source indices to keep.
    total_src = info.frame_count if info.frame_count > 0 else None
    if total_src is not None and info.duration > 0:
        # Build target timestamps 0, 1/fps, 2/fps, ... < duration
        n_targets = int(info.duration * config.target_fps + 1e-9) + 1
        # Clamp to source frame count
        wanted: set[int] = set()
        for k in range(n_targets):
            ts = k / config.target_fps
            src_idx = int(round(ts * info.fps))
            if 0 <= src_idx < total_src:
                wanted.add(src_idx)
            elif src_idx == total_src and total_src > 0:
                # Last timestamp may map to one-past-end due to rounding
                wanted.add(total_src - 1)
        wanted_sorted = sorted(wanted)
    else:
        # No reliable duration/frame_count: read every frame and throttle by FPS ratio
        wanted_sorted = None  # signal to throttle loop

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    # Clean existing JPGs when overwriting or when cache is invalid
    if not (not config.overwrite and frames_cache_valid(output_dir, video_path, config)):
        for p in output_dir.glob("frame_*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass

    frame_paths: list[Path] = []
    written_idx = 0
    src_idx = 0
    wanted_ptr = 0  # index into wanted_sorted

    # Throttle factor for the fallback path
    step = max(1, int(round(info.fps / config.target_fps))) if wanted_sorted is None and info.fps > 0 else 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            keep = False
            if wanted_sorted is not None:
                if wanted_ptr < len(wanted_sorted) and src_idx == wanted_sorted[wanted_ptr]:
                    keep = True
                    wanted_ptr += 1
                    # Skip duplicates (can happen with rounding)
                    while wanted_ptr < len(wanted_sorted) and wanted_sorted[wanted_ptr] == src_idx:
                        wanted_ptr += 1
            else:
                keep = (src_idx % step == 0)
            if keep:
                written_idx += 1
                out_name = config.frame_pattern.format(idx=written_idx)
                out_path = output_dir / out_name
                resized = _resize_frame(frame, config.target_width, config.target_height)
                ok_write = cv2.imwrite(
                    str(out_path),
                    resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)],
                )
                if not ok_write:
                    raise RuntimeError(f"failed to write frame: {out_path}")
                frame_paths.append(out_path)
            src_idx += 1
    finally:
        cap.release()

    # Write cache manifest
    rels = [p.name for p in frame_paths]
    _write_manifest(output_dir, video_path, config, rels, info)

    return ExtractionResult(
        frame_paths=frame_paths,
        n_extracted=len(frame_paths),
        skipped=False,
        video_info=info,
        output_dir=output_dir,
    )
