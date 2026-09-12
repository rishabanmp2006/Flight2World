"""flight2world.core.pipeline

Pipeline orchestration for FLIGHT2WORLD.

Conceptual flow:

    video → frame extraction → curation → COLMAP → sparse model →
    depth → calibration → fusion → cleanup → output

Each stage is independently callable; :class:`Pipeline` composes them with
filesystem-based resumability:

    {output_root}/
        frames/            extracted JPG frames (video → frames)
        colmap/            database.db + sparse/ + sparse_txt/ (COLMAP)
        reconstruction/    fused_raw.ply, fused_clean.ply, confidence.ply (Phase 5A)
        diagnostics/       per-stage JSON sidecars
        metadata.json      run metadata (coordinate system etc.)

A stage is skipped when its outputs already exist and the cache manifest
matches. The full pipeline can also be driven stage-by-stage for testing.

This module is architecture/orchestration — it does NOT change the
reconstruction mathematics. Depth estimation itself is still provided by
``core.depth`` with the V10 defaults.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.config import (
    COORDINATE_SYSTEM,
    DEFAULT_DEPTH_DEVICE,
    DEFAULT_DEPTH_MODEL,
    GEOREFERENCED,
    METRIC_SCALE,
    PIPELINE_COLMAP_SUBDIR,
    PIPELINE_CONFIDENCE_JSON,
    PIPELINE_CONFIDENCE_PLY,
    PIPELINE_DIAGNOSTICS_SUBDIR,
    PIPELINE_FRAMES_SUBDIR,
    PIPELINE_FUSED_CLEAN_NAME,
    PIPELINE_FUSED_RAW_NAME,
    PIPELINE_METADATA_NAME,
    PIPELINE_RECONSTRUCTION_SUBDIR,
    PIPELINE_SPARSE_SUBDIR,
)
from core.diagnostics import Diagnostics, StageRecord, StageTimer, write_stage_diagnostics


# ---------------------------------------------------------------------------
# Run-level types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """High-level pipeline configuration.

    Thin wrapper over ``core.config.PipelineConfig`` (V10 thresholds) plus
    pipeline/output-level knobs. Defaults preserve the benchmark.
    """

    # Where the pipeline writes its outputs (run root)
    output_root: Path = field(default_factory=lambda: Path("outputs/run_001"))
    # Number of COLMAP / calibrated frames to use (None => all)
    max_frames: Optional[int] = None
    # Force re-run even when caches are valid
    force: bool = False
    # Depth model configuration (Phase 5A). ``depth_factory`` is injectable
    # for tests (a callable mimicking transformers.pipeline).
    depth_model: str = DEFAULT_DEPTH_MODEL
    depth_device: str = DEFAULT_DEPTH_DEVICE
    depth_factory: Optional[Any] = None

    # Video extraction is configured via core.video.ExtractionConfig;
    # COLMAP via core.colmap_runner.ColmapRunConfig — callers pass them
    # explicitly to the relevant stage rather than nesting them here.


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    success: bool
    output_root: Path
    diagnostics: Diagnostics
    metadata: dict
    # Per-stage outputs (populated when the stage ran / was cached)
    frame_paths: list[Path] = field(default_factory=list)
    colmap_result: Optional[Any] = None  # ColmapRunResult when COLMAP ran
    sparse_summary: Optional[dict] = None
    calibration_summary: Optional[dict] = None
    fusion_summary: Optional[dict] = None
    cleanup_summary: Optional[dict] = None
    confidence_summary: Optional[dict] = None
    # Final PLY outputs (Phase 5A)
    fused_raw_path: Optional[Path] = None
    fused_clean_path: Optional[Path] = None
    confidence_ply: Optional[Path] = None
    # Final outputs
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def build_metadata(
    diagnostics: Diagnostics,
    extra: Optional[dict] = None,
) -> dict:
    """Build the run ``metadata.json`` payload.

    The current benchmark has no GPS/RTK/GCP source — all geometry is in
    COLMAP's arbitrary coordinate system. This is recorded explicitly so
    that downstream consumers cannot mistake scene units for metres.

    Returns a JSON-serialisable dict. Callers are responsible for writing
    it to ``{output_root}/metadata.json`` (see :func:`write_metadata`).
    """
    payload: dict[str, Any] = {
        "coordinate_system": COORDINATE_SYSTEM,   # "COLMAP_relative"
        "georeferenced": GEOREFERENCED,             # False
        "metric_scale": METRIC_SCALE,               # None -> null in JSON
        "pipeline_version": "phase5a",
        "run_id": diagnostics.run_id,
        "created_at": diagnostics.created_at,
        "output_root": diagnostics.output_root,
        "stages": [s.to_dict() for s in diagnostics.stages],
    }
    if extra:
        payload.update(extra)
    # Honest per-stage counts for the summary
    payload.setdefault("frame_count", None)
    payload.setdefault("registered_images", None)
    payload.setdefault("sparse_points", None)
    return payload


def write_metadata(output_root: str | Path, payload: dict) -> Path:
    """Write ``metadata.json`` under *output_root*."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / PIPELINE_METADATA_NAME
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_metadata(output_root: str | Path) -> Optional[dict]:
    """Load ``metadata.json`` if it exists, else None."""
    p = Path(output_root) / PIPELINE_METADATA_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage cache helpers
# ---------------------------------------------------------------------------

def _cache_manifest_path(output_root: Path, stage: str) -> Path:
    return output_root / PIPELINE_DIAGNOSTICS_SUBDIR / f"{stage}.json"


def stage_cached(output_root: str | Path, stage: str) -> bool:
    """Return True if the diagnostics sidecar for *stage* exists and marks it ok."""
    data = _load_stage_sidecar(output_root, stage)
    return bool(data and data.get("status") == "ok")


def _load_stage_sidecar(output_root: str | Path, stage: str) -> Optional[dict]:
    p = Path(output_root) / PIPELINE_DIAGNOSTICS_SUBDIR / f"{stage}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual stages (independently testable)
# ---------------------------------------------------------------------------

def stage_extract_frames(
    video_path: str | Path,
    output_root: str | Path,
    extraction_config=None,
    diagnostics: Optional[Diagnostics] = None,
) -> tuple[list[Path], StageRecord]:
    """Stage 1: video → frames (via :mod:`core.video`).

    Respects extraction cache; on a cache hit no frames are rewritten.
    """
    from core.video import ExtractionConfig, extract_frames

    output_root = Path(output_root)
    frames_dir = output_root / PIPELINE_FRAMES_SUBDIR
    cfg = extraction_config or ExtractionConfig()

    timer = StageTimer("extract_frames")
    timer.__enter__()
    try:
        result = extract_frames(video_path, frames_dir, config=cfg)
        timer.__exit__(None, None, None)
        rec = StageRecord(
            stage="extract_frames",
            status="ok",
            elapsed_s=timer.elapsed,
            details={
                "n_frames": len(result.frame_paths),
                "skipped": result.skipped,
                "video": str(video_path),
                "output_dir": str(frames_dir),
                "target_fps": cfg.target_fps,
                "target_size": [cfg.target_width, cfg.target_height],
            },
        )
    except Exception as e:
        timer.__exit__(None, None, None)
        rec = StageRecord(
            stage="extract_frames", status="failed",
            elapsed_s=timer.elapsed, error=str(e),
        )
        # persist diagnostics even on failure
        if diagnostics is not None:
            diagnostics.add(rec)
            write_stage_diagnostics(output_root, rec)
        else:
            write_stage_diagnostics(output_root, rec)
        raise

    if diagnostics is not None:
        diagnostics.add(rec)
        write_stage_diagnostics(output_root, rec)
    else:
        write_stage_diagnostics(output_root, rec)

    return result.frame_paths, rec


def stage_run_colmap(
    image_path: str | Path,
    output_root: str | Path,
    colmap_config=None,
    export_txt: bool = True,
    diagnostics: Optional[Diagnostics] = None,
) -> tuple[Any, StageRecord]:
    """Stage 2: COLMAP sparse reconstruction (via :mod:`core.colmap_runner`)."""
    from core.colmap_runner import ColmapRunConfig, run_colmap

    output_root = Path(output_root)
    colmap_dir = output_root / PIPELINE_COLMAP_SUBDIR
    cfg = colmap_config or ColmapRunConfig()

    t = StageTimer("colmap")
    t.__enter__()
    try:
        result = run_colmap(
            image_path=image_path,
            output_dir=colmap_dir,
            config=cfg,
            export_txt=export_txt,
        )
        t.__exit__(None, None, None)
        rec = StageRecord(
            stage="colmap", status="ok", elapsed_s=t.elapsed,
            details={
                "database": str(result.database_path),
                "sparse_dir": str(result.sparse_dir),
                "sparse_txt": str(result.sparse_txt_dir) if result.sparse_txt_dir else None,
                "stages": list(result.stages.keys()),
            },
        )
    except Exception as e:
        t.__exit__(None, None, None)
        rec = StageRecord(
            stage="colmap", status="failed",
            elapsed_s=t.elapsed, error=str(e),
            details={"error_type": type(e).__name__},
        )
        if diagnostics is not None:
            diagnostics.add(rec)
            write_stage_diagnostics(output_root, rec)
        else:
            write_stage_diagnostics(output_root, rec)
        raise

    if diagnostics is not None:
        diagnostics.add(rec)
    write_stage_diagnostics(output_root, rec)
    return result, rec


def stage_curate_and_summarise(
    images: dict,
    frame_dir: str | Path,
    min_baseline_frac: Optional[float] = None,
    output_root: Optional[str | Path] = None,
    diagnostics: Optional[Diagnostics] = None,
) -> tuple[Any, StageRecord]:
    """Stage: curation bookkeeping (ordering + min-baseline).

    Wraps :func:`core.curate.curate_frames` and records diagnostics.
    """
    from core.curate import curate_frames
    kwargs: dict[str, Any] = {}
    if min_baseline_frac is not None:
        kwargs["min_baseline_frac"] = min_baseline_frac

    with StageTimer("curation") as t:
        result = curate_frames(images, frame_dir, **kwargs)
    rec = StageRecord(
        stage="curation", status="ok", elapsed_s=t.elapsed,
        details={
            "n_registered": len(images),
            "n_curated": len(result.curated),
            "n_dropped": len(result.dropped),
            "baseline_stats": result.baseline_stats,
            "min_baseline": result.min_baseline,
        },
    )

    if output_root is not None:
        if diagnostics is not None:
            diagnostics.add(rec)
        write_stage_diagnostics(Path(output_root), rec)
    elif diagnostics is not None:
        diagnostics.add(rec)

    return result, rec


# ---------------------------------------------------------------------------
# Phase 5A: sparse preparation (locate + ensure TXT + build gate)
# ---------------------------------------------------------------------------

def _prepare_sparse_data(output_root: str | Path):
    """Locate the selected/largest sparse model, ensure TXT export, and
    build gate structures.

    Returns (images, sparse_points, sparse_xyz, sparse_tree, local_r,
             sparse_txt_dir, selected_model_dir).

    Raises FileNotFoundError / RuntimeError on missing data.
    """
    import numpy as np

    from core.colmap_io import load_colmap_images, load_colmap_points
    from core.gate import build_sparse_kdtree, compute_local_radius

    output_root = Path(output_root)
    colmap_dir = output_root / PIPELINE_COLMAP_SUBDIR
    sparse_dir = colmap_dir / PIPELINE_SPARSE_SUBDIR
    sparse_txt_dir = colmap_dir / "sparse_txt"

    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse dir not found: {sparse_dir}")

    candidates = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no sparse models in {sparse_dir}")

    def _score(p: Path) -> int:
        img = p / "images.bin"
        return int(img.stat().st_size) if img.exists() else -1

    selected = max(candidates, key=_score)

    # Ensure TXT export exists.
    need_export = not (sparse_txt_dir / "images.txt").exists() or not (sparse_txt_dir / "points3D.txt").exists()
    if need_export:
        from core.colmap_runner import model_converter
        model_converter(input_path=selected, output_path=sparse_txt_dir, output_type="TXT")

    images = load_colmap_images(sparse_txt_dir / "images.txt")
    sparse_points = load_colmap_points(sparse_txt_dir / "points3D.txt")

    if not sparse_points:
        raise RuntimeError(f"no sparse points in {sparse_txt_dir / 'points3D.txt'}")

    sparse_xyz = np.array(list(sparse_points.values()), dtype=np.float64)
    sparse_tree = build_sparse_kdtree(sparse_xyz)
    local_r = compute_local_radius(sparse_xyz)

    return images, sparse_points, sparse_xyz, sparse_tree, local_r, sparse_txt_dir, selected


# ---------------------------------------------------------------------------
# Phase 5A: depth + calibration + track maps (Pass 1)
# ---------------------------------------------------------------------------

def stage_depth_and_calibration(
    images: dict,
    sparse_points: dict,
    frame_dir: str | Path,
    curated,
    output_root: str | Path,
    diagnostics: Optional[Diagnostics] = None,
    depth_model: Optional[str] = None,
    depth_device: Optional[str] = None,
    depth_factory: Optional[Any] = None,
):
    """Pass 1: sequential depth inference, calibration selection, track maps,
    and RMSE outlier gating.

    * Reuses a single DepthAnythingV2 instance.
    * Processes frames sequentially (no batching / no multiprocessing).
    * Writes diagnostics/depth.json and diagnostics/calibration.json.

    Returns (frame_records, calib_rows, rmse_diag, depth_rec, cal_rec).
    """
    import numpy as np
    from PIL import Image

    from core.calibration import calibrate_frame, filter_rmse_outliers, fit_depth_calibration
    from core.config import CALIB_MIN_SAMPLES
    from core.track_maps import build_track_maps

    output_root = Path(output_root)
    frame_dir = Path(frame_dir)
    depth_model = depth_model or DEFAULT_DEPTH_MODEL
    depth_device = depth_device or DEFAULT_DEPTH_DEVICE

    # ---- depth stage ----
    t_depth = StageTimer("depth")
    t_depth.__enter__()
    depth_maps: dict[str, np.ndarray] = {}
    rgb_maps: dict[str, np.ndarray] = {}
    depth_per_frame: list[dict] = []
    pipe = None
    actual_model = depth_model
    actual_device = depth_device
    try:
        from core.depth import DepthAnythingV2
        pipe = DepthAnythingV2(model=depth_model, device=depth_device, pipeline_factory=depth_factory)
        actual_model = getattr(pipe, "model", depth_model)
        actual_device = getattr(pipe, "device", depth_device)
        for _image_id, data in curated:
            name = data["name"]
            img_path = frame_dir / name
            if not img_path.exists():
                depth_per_frame.append({"name": name, "status": "missing_image"})
                continue
            # Keep file handle scoped; Image.open is lazy.
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                # Hold a copy for fusion's RGB sampling; do not keep PIL object.
                rgb = np.asarray(im_rgb, dtype=np.uint8)
                depth = pipe.predict_depth(im_rgb)
            depth_maps[name] = depth
            rgb_maps[name] = rgb
            depth_per_frame.append({"name": name, "status": "ok", "shape": list(depth.shape), "dtype": str(depth.dtype)})
        t_depth.__exit__(None, None, None)
        depth_rec = StageRecord(
            stage="depth", status="ok", elapsed_s=t_depth.elapsed,
            details={
                "model": actual_model,
                "device": actual_device,
                "n_curated": len(curated),
                "n_depth_ok": len(depth_maps),
                "per_frame": depth_per_frame,
            },
        )
    except Exception as e:
        t_depth.__exit__(None, None, None)
        depth_rec = StageRecord(
            stage="depth", status="failed", elapsed_s=t_depth.elapsed,
            error=str(e), details={"model": depth_model, "device": depth_device, "error_type": type(e).__name__},
        )
        write_stage_diagnostics(output_root, depth_rec)
        if diagnostics is not None:
            diagnostics.add(depth_rec)
        raise
    write_stage_diagnostics(output_root, depth_rec)
    if diagnostics is not None:
        diagnostics.add(depth_rec)

    # ---- calibration + track maps + RMSE gate ----
    t_cal = StageTimer("calibration")
    t_cal.__enter__()
    frame_records: list[dict] = []
    calib_rows: list[dict] = []
    rmse_diag: dict = {}
    try:
        for _image_id, data in curated:
            name = data["name"]
            if name not in depth_maps:
                # Already recorded as missing/depth-failed; if depth succeeded but this frame was curated,
                # it will have an entry; otherwise skip with a synthetic calib row if not already present.
                # Check if we already emitted a depth_per_frame entry for this name with non-ok.
                continue
            depth = depth_maps[name]
            rgb = rgb_maps[name]
            d_vals, z_vals = calibrate_frame(data, sparse_points, depth)
            if len(z_vals) < CALIB_MIN_SAMPLES:
                calib_rows.append({"name": name, "skipped": "few_samples", "samples": int(len(z_vals))})
                continue
            fit = fit_depth_calibration(d_vals, z_vals)
            if fit is None:
                calib_rows.append({"name": name, "skipped": "low_corr", "samples": int(len(z_vals))})
                continue
            z_grid, dist_grid, n_tracks = build_track_maps(data, sparse_points)
            if n_tracks < 50:
                calib_rows.append({"name": name, "skipped": "few_tracks", "tracks": int(n_tracks)})
                continue
            frame_records.append({
                "name": name,
                "R": data["R"],
                "t": data["t"],
                "depth": depth,
                "rgb": rgb,
                "a": fit["a"],
                "b": fit["b"],
                "model": fit["model"],
                "rmse": fit["rmse"],
                "corr": fit["corr"],
                "samples": fit["n_samples"],
                "z_grid": z_grid,
                "dist_grid": dist_grid,
            })
            calib_rows.append({
                "name": name,
                "model": fit["model"],
                "samples": int(fit["n_samples"]),
                "corr": round(float(fit["corr"]), 4),
                "rmse": round(float(fit["rmse"]), 5),
                "tracks": int(n_tracks),
            })

        kept, dropped, rmse_diag = filter_rmse_outliers(frame_records)
        frame_records = kept
        t_cal.__exit__(None, None, None)
        cal_rec = StageRecord(
            stage="calibration", status="ok", elapsed_s=t_cal.elapsed,
            details={
                "n_curated": len(curated),
                "n_calibrated_before_rmse": len(kept) + len(dropped),
                "n_kept": len(kept),
                "n_dropped_rmse": len(dropped),
                "rmse_median": rmse_diag.get("median"),
                "rmse_threshold": rmse_diag.get("threshold"),
                "calibration_rows": calib_rows,
                "rmse_diagnostics": rmse_diag,
            },
        )
    except Exception as e:
        t_cal.__exit__(None, None, None)
        cal_rec = StageRecord(
            stage="calibration", status="failed", elapsed_s=t_cal.elapsed,
            error=str(e), details={"error_type": type(e).__name__},
        )
        write_stage_diagnostics(output_root, cal_rec)
        if diagnostics is not None:
            diagnostics.add(cal_rec)
        raise
    write_stage_diagnostics(output_root, cal_rec)
    if diagnostics is not None:
        diagnostics.add(cal_rec)

    return frame_records, calib_rows, rmse_diag, depth_rec, cal_rec


# ---------------------------------------------------------------------------
# Phase 5A: fusion (Pass 2)
# ---------------------------------------------------------------------------

def stage_fuse(
    frame_records,
    sparse_tree,
    local_r,
    output_root: str | Path,
    diagnostics: Optional[Diagnostics] = None,
    config=None,
):
    """Track-anchored dense fusion (delegates to core.fusion.fuse_all)."""
    output_root = Path(output_root)
    with StageTimer("fusion") as t:
        from core.fusion import fuse_all
        points, colors, stats, vote_hist = fuse_all(frame_records, sparse_tree, local_r, config=config)
    rec = StageRecord(
        stage="fusion", status="ok", elapsed_s=t.elapsed,
        details={
            "candidates": int(stats.get("candidates", 0)),
            "gate_pass": int(stats.get("gate_pass", 0)),
            "votes_pass": int(stats.get("votes_pass", 0)),
            "frames_fused": int(stats.get("frames_fused", 0)),
            "vote_hist": {k: int(v) for k, v in vote_hist.items()},
            "stats": {k: int(v) for k, v in stats.items()},
        },
    )
    write_stage_diagnostics(output_root, rec)
    if diagnostics is not None:
        diagnostics.add(rec)
    return points, colors, stats, vote_hist, rec


# ---------------------------------------------------------------------------
# Phase 5A: cleanup (voxel+stat → raw, DBSCAN → clean)
# ---------------------------------------------------------------------------

def stage_cleanup(
    points,
    colors,
    output_root: str | Path,
    diagnostics: Optional[Diagnostics] = None,
):
    """Cleanup: voxel+stat then DBSCAN, writing reconstruction PLYs."""
    output_root = Path(output_root)
    with StageTimer("cleanup") as t:
        from core.cleanup import run_cleanup
        from core.export import write_point_cloud_ply

        pcd_raw, pcd_clean, cleanup_diag = run_cleanup(points, colors)

        recon_dir = output_root / PIPELINE_RECONSTRUCTION_SUBDIR
        recon_dir.mkdir(parents=True, exist_ok=True)
        raw_path = recon_dir / PIPELINE_FUSED_RAW_NAME
        clean_path = recon_dir / PIPELINE_FUSED_CLEAN_NAME
        write_point_cloud_ply(raw_path, pcd_raw)
        write_point_cloud_ply(clean_path, pcd_clean)
    rec = StageRecord(
        stage="cleanup", status="ok", elapsed_s=t.elapsed,
        details={
            "raw_points": int(len(pcd_raw.points)),
            "clean_points": int(len(pcd_clean.points)),
            "raw_path": str(raw_path),
            "clean_path": str(clean_path),
            "diagnostics": cleanup_diag,
        },
    )
    write_stage_diagnostics(output_root, rec)
    if diagnostics is not None:
        diagnostics.add(rec)
    return pcd_raw, pcd_clean, cleanup_diag, rec


# ---------------------------------------------------------------------------
# Phase 5A: confidence (global voting)
# ---------------------------------------------------------------------------

def stage_confidence(
    pcd_clean,
    frame_records,
    sparse_points: dict,
    output_root: str | Path,
    diagnostics: Optional[Diagnostics] = None,
):
    """Global confidence replay on the clean cloud."""
    import numpy as np

    output_root = Path(output_root)
    with StageTimer("confidence") as t:
        from core.confidence import compute_confidence, write_confidence_ply

        P = np.asarray(pcd_clean.points, dtype=np.float64)
        if pcd_clean.has_colors():
            colors_uint8 = (np.asarray(pcd_clean.colors) * 255.0).astype(np.uint8)
        else:
            colors_uint8 = None

        result = compute_confidence(P, frame_records, sparse_points)

        conf = result["confidence"]
        evidence = result["evidence"]
        net_votes = result["net_votes"]
        residual = result["residual"]
        worst = result["worst_residual"]
        dist_sparse = result["dist_sparse"]
        summary = result["summary"]

        recon_dir = output_root / PIPELINE_RECONSTRUCTION_SUBDIR
        recon_dir.mkdir(parents=True, exist_ok=True)
        conf_ply = recon_dir / PIPELINE_CONFIDENCE_PLY
        write_confidence_ply(conf_ply, P, colors_uint8, conf, evidence, net_votes, residual, worst, dist_sparse)
        try:
            (recon_dir / PIPELINE_CONFIDENCE_JSON).write_text(json.dumps(summary, indent=2))
        except Exception:
            pass
    rec = StageRecord(
        stage="confidence", status="ok", elapsed_s=t.elapsed,
        details={
            "points": int(result["N"]),
            "confidence_ply": str(conf_ply),
            "summary": summary,
        },
    )
    write_stage_diagnostics(output_root, rec)
    if diagnostics is not None:
        diagnostics.add(rec)
    return result, rec


# ---------------------------------------------------------------------------
# Full pipeline (with simple resumability)
# ---------------------------------------------------------------------------

class Pipeline:
    """Simple resumable pipeline.

    Each external stage (video extraction, COLMAP) is skipped when its
    cached outputs are still valid — unless ``force=True`` is set on the
    :class:`PipelineConfig`.

    Heavy reconstruction stages (depth, calibration, fusion) are not
    executed by default; this class provides the orchestration skeleton
    and the metadata/diagnostics plumbing so that those stages can be
    added incrementally in future phases without changing the output
    layout or cache contract.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        video_path: Optional[str | Path] = None,
        colmap_image_path: Optional[str | Path] = None,
        run_id: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.video_path = Path(video_path) if video_path is not None else None
        # When no video is supplied the COLMAP images live wherever the
        # caller points; default to the pipeline's own frames dir.
        if colmap_image_path is not None:
            self.colmap_image_path = Path(colmap_image_path)
        elif self.video_path is not None:
            self.colmap_image_path = self.config.output_root / PIPELINE_FRAMES_SUBDIR
        else:
            self.colmap_image_path = self.config.output_root / PIPELINE_FRAMES_SUBDIR
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())

    def _diagnostics(self) -> Diagnostics:
        return Diagnostics(run_id=self.run_id, output_root=str(self.config.output_root))

    def run(
        self,
        run_colmap: bool = False,
        run_reconstruction: bool = False,
        write_metadata_flag: bool = True,
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            run_colmap: whether to run the COLMAP stage (expensive; off
                by default so lightweight tests do not invoke COLMAP).
            run_reconstruction: whether to run the V10 reconstruction
                stages (depth → calibration → fusion → cleanup →
                confidence → PLY export). Off by default so lightweight
                tests stay fast; when true, COLMAP outputs are reused if
                they already exist (unless force=True).
            write_metadata_flag: whether to write ``metadata.json`` at the end.
        """
        output_root = Path(self.config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        diag = self._diagnostics()
        frame_paths: list[Path] = []
        colmap_result = None
        error: Optional[str] = None
        success = True

        sparse_summary: Optional[dict] = None
        calibration_summary: Optional[dict] = None
        fusion_summary: Optional[dict] = None
        cleanup_summary: Optional[dict] = None
        confidence_summary: Optional[dict] = None
        fused_raw_path: Optional[Path] = None
        fused_clean_path: Optional[Path] = None
        confidence_ply: Optional[Path] = None

        # Stage 1: video → frames (only when a video is supplied)
        if self.video_path is not None:
            if not self.config.force and stage_cached(output_root, "extract_frames"):
                # Serve from cache diagnostics (no re-extraction)
                cached = _load_stage_sidecar(output_root, "extract_frames")
                n = (cached or {}).get("details", {}).get("n_frames", 0)
                rec = StageRecord(
                    stage="extract_frames", status="ok", elapsed_s=0.0,
                    details={"n_frames": n, "skipped": True, "cached": True},
                )
                diag.add(rec)
                # Best-effort: list existing frames on disk
                frames_dir = output_root / PIPELINE_FRAMES_SUBDIR
                if frames_dir.exists():
                    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
            else:
                try:
                    from core.video import ExtractionConfig
                    fps = getattr(self.config, "video_fps", None)
                    cfg = ExtractionConfig(target_fps=fps) if fps is not None else None
                    frame_paths, _rec = stage_extract_frames(
                        self.video_path, output_root,
                        extraction_config=cfg, diagnostics=diag,
                    )
                except Exception as e:
                    error = f"extract_frames failed: {e}"
                    success = False
                    if write_metadata_flag:
                        meta = build_metadata(diag, {"error": error, "frame_count": len(frame_paths)})
                        write_metadata(output_root, meta)
                    return PipelineResult(
                        success=False, output_root=output_root, diagnostics=diag,
                        metadata=meta, frame_paths=frame_paths, error=error,
                    )
        else:
            # No video: COLMAP may operate on an existing frames dir
            frames_dir = self.colmap_image_path
            if frames_dir.exists():
                frame_paths = sorted(frames_dir.glob("*.jpg"))

        # Stage 2: COLMAP (opt-in; expensive)
        if run_colmap:
            if not self.config.force and stage_cached(output_root, "colmap"):
                cached = _load_stage_sidecar(output_root, "colmap")
                details = (cached or {}).get("details", {})
                rec = StageRecord(
                    stage="colmap", status="ok", elapsed_s=0.0,
                    details={**details, "cached": True},
                )
                diag.add(rec)
            else:
                try:
                    from core.colmap_runner import ColmapRunConfig  # noqa: F401
                    colmap_result, _rec = stage_run_colmap(
                        self.colmap_image_path, output_root,
                        diagnostics=diag,
                    )
                except Exception as e:
                    error = f"colmap failed: {e}"
                    success = False
                    if write_metadata_flag:
                        meta = build_metadata(diag, {"error": error, "frame_count": len(frame_paths)})
                        write_metadata(output_root, meta)
                    return PipelineResult(
                        success=False, output_root=output_root, diagnostics=diag,
                        metadata=meta, frame_paths=frame_paths, error=error,
                    )

        # Stages 3-...: V10 reconstruction (opt-in)
        if run_reconstruction:
            try:
                # C/D: locate sparse model, ensure TXT, build gate structures
                import numpy as np  # for median local_r etc.

                images, sparse_points, sparse_xyz, sparse_tree, local_r, sparse_txt_dir, selected_model = _prepare_sparse_data(output_root)

                sparse_summary = {
                    "n_images": len(images),
                    "n_points": len(sparse_points),
                    "sparse_txt": str(sparse_txt_dir),
                    "selected_model": str(selected_model),
                    "local_r_median": float(np.median(local_r)) if len(local_r) else 0.0,
                    "local_r_p90": float(np.percentile(local_r, 90)) if len(local_r) else 0.0,
                }

                # E: curation
                curation_result, _rec_curation = stage_curate_and_summarise(
                    images, self.colmap_image_path, output_root=output_root, diagnostics=diag,
                )
                curated = curation_result.curated
                if self.config.max_frames is not None and len(curated) > self.config.max_frames:
                    curated = curated[: self.config.max_frames]

                if not curated:
                    raise RuntimeError("no curated frames (registered frames missing or baseline filter removed all)")

                # F/G/H: depth + calibration + track maps + RMSE gate
                frame_records, calib_rows, rmse_diag, _depth_rec, _cal_rec = stage_depth_and_calibration(
                    images, sparse_points, self.colmap_image_path, curated,
                    output_root, diagnostics=diag,
                    depth_model=self.config.depth_model,
                    depth_device=self.config.depth_device,
                    depth_factory=self.config.depth_factory,
                )

                calibration_summary = {
                    "n_curated": len(curated),
                    "n_kept": len(frame_records),
                    "calibration_rows": calib_rows,
                    "rmse_diagnostics": rmse_diag,
                }

                if not frame_records:
                    raise RuntimeError("no calibrated frames after gating (all frames failed depth/calibration/track gates)")

                # I: fusion
                points, colors, stats, vote_hist, _rec_fusion = stage_fuse(
                    frame_records, sparse_tree, local_r, output_root, diagnostics=diag,
                )

                fusion_summary = {
                    "stats": {k: int(v) for k, v in stats.items()},
                    "vote_hist": {k: int(v) for k, v in vote_hist.items()},
                }

                if len(points) == 0:
                    raise RuntimeError("No points survived track-anchored fusion.")

                # J: cleanup
                pcd_raw, pcd_clean, cleanup_diag, _rec_cleanup = stage_cleanup(
                    points, colors, output_root, diagnostics=diag,
                )
                cleanup_summary = cleanup_diag
                fused_raw_path = output_root / PIPELINE_RECONSTRUCTION_SUBDIR / PIPELINE_FUSED_RAW_NAME
                fused_clean_path = output_root / PIPELINE_RECONSTRUCTION_SUBDIR / PIPELINE_FUSED_CLEAN_NAME

                # K: confidence
                conf_result, _rec_conf = stage_confidence(
                    pcd_clean, frame_records, sparse_points, output_root, diagnostics=diag,
                )
                confidence_summary = conf_result["summary"]
                confidence_ply = output_root / PIPELINE_RECONSTRUCTION_SUBDIR / PIPELINE_CONFIDENCE_PLY

            except Exception as e:
                error = f"reconstruction failed: {e}"
                success = False
                if write_metadata_flag:
                    meta = build_metadata(diag, {
                        "error": error,
                        "frame_count": len(frame_paths),
                        "video_path": str(self.video_path) if self.video_path else None,
                        "image_path": str(self.colmap_image_path),
                        "registered_images": sparse_summary.get("n_images") if sparse_summary else None,
                        "sparse_points": sparse_summary.get("n_points") if sparse_summary else None,
                        "sparse_summary": sparse_summary,
                        "calibration_summary": calibration_summary,
                        "fusion_summary": fusion_summary,
                        "cleanup_summary": cleanup_summary,
                        "confidence_summary": confidence_summary,
                    })
                    write_metadata(output_root, meta)
                return PipelineResult(
                    success=False,
                    output_root=output_root,
                    diagnostics=diag,
                    metadata=meta,
                    frame_paths=frame_paths,
                    colmap_result=colmap_result,
                    sparse_summary=sparse_summary,
                    calibration_summary=calibration_summary,
                    fusion_summary=fusion_summary,
                    cleanup_summary=cleanup_summary,
                    confidence_summary=confidence_summary,
                    fused_raw_path=fused_raw_path,
                    fused_clean_path=fused_clean_path,
                    confidence_ply=confidence_ply,
                    error=error,
                )

        # Always produce metadata at the end (even on partial success)
        # Enrich with reconstruction counts when available
        extra: dict[str, Any] = {
            "frame_count": len(frame_paths),
            "video_path": str(self.video_path) if self.video_path else None,
            "image_path": str(self.colmap_image_path),
        }
        if sparse_summary is not None:
            extra["registered_images"] = sparse_summary.get("n_images")
            extra["sparse_points"] = sparse_summary.get("n_points")
            extra["sparse_summary"] = sparse_summary
        if calibration_summary is not None:
            extra["calibration_summary"] = calibration_summary
        if fusion_summary is not None:
            extra["fusion_summary"] = fusion_summary
        if cleanup_summary is not None:
            extra["cleanup_summary"] = cleanup_summary
        if confidence_summary is not None:
            extra["confidence_summary"] = confidence_summary
        if fused_raw_path is not None:
            extra["fused_raw_path"] = str(fused_raw_path)
        if fused_clean_path is not None:
            extra["fused_clean_path"] = str(fused_clean_path)
        if confidence_ply is not None:
            extra["confidence_ply"] = str(confidence_ply)

        metadata = build_metadata(diag, extra)
        if write_metadata_flag:
            write_metadata(output_root, metadata)

        return PipelineResult(
            success=success,
            output_root=output_root,
            diagnostics=diag,
            metadata=metadata,
            frame_paths=frame_paths,
            colmap_result=colmap_result,
            sparse_summary=sparse_summary,
            calibration_summary=calibration_summary,
            fusion_summary=fusion_summary,
            cleanup_summary=cleanup_summary,
            confidence_summary=confidence_summary,
            fused_raw_path=fused_raw_path,
            fused_clean_path=fused_clean_path,
            confidence_ply=confidence_ply,
            error=error,
        )
