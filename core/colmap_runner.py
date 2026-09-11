"""flight2world.core.colmap_runner

Controlled wrappers around the native COLMAP executable.

No pipeline logic lives here — only subprocess-safe primitives for the
individual COLMAP stages that the pipeline orchestrates.

Validated benchmark defaults (from ``core.config``):

    ImageReader.camera_model  = SIMPLE_RADIAL
    ImageReader.single_camera = 1
    SequentialMatching.overlap = 10

The COLMAP executable is configurable; the default ``"colmap"`` relies on
PATH (colmap 4.1.1 was present during Phase 3). Every command is built as
a list of arguments (no ``shell=True``).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import (
    COLMAP_CAMERA_MODEL,
    COLMAP_EXE,
    COLMAP_MAPPER_BA_REFINE_EXTRA_PARAMS,
    COLMAP_MAPPER_BA_REFINE_FOCAL_LENGTH,
    COLMAP_MAPPER_BA_REFINE_PRINCIPAL_POINT,
    COLMAP_MAPPER_MIN_NUM_MATCHES,
    COLMAP_SEQUENTIAL_OVERLAP,
    COLMAP_SINGLE_CAMERA,
    COLMAP_USE_GPU,
    PIPELINE_DATABASE_NAME,
    PIPELINE_SPARSE_SUBDIR,
)


# ---------------------------------------------------------------------------
# Results / errors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColmapCommandResult:
    """Result of a single COLMAP subprocess invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class ColmapError(RuntimeError):
    """Raised when a COLMAP stage fails."""

    def __init__(self, stage: str, result: ColmapCommandResult):
        super().__init__(
            f"COLMAP stage {stage!r} failed (exit {result.returncode}): "
            f"{result.stderr[:800]}"
        )
        self.stage = stage
        self.result = result


# ---------------------------------------------------------------------------
# Executable resolution
# ---------------------------------------------------------------------------

def resolve_colmap_exe(exe: Optional[str] = None) -> str:
    """Resolve the COLMAP executable.

    Returns the executable string to invoke. Raises ``FileNotFoundError``
    when it cannot be found on PATH (or at the explicit path).
    """
    candidate = exe or COLMAP_EXE
    # Absolute / explicit path: check directly
    if "/" in candidate or "\\" in candidate:
        p = Path(candidate)
        if not p.exists():
            raise FileNotFoundError(f"COLMAP executable not found: {candidate}")
        return str(p)
    # Bare name: look up on PATH
    found = shutil.which(candidate)
    if found is None:
        raise FileNotFoundError(
            f"COLMAP executable {candidate!r} not found on PATH. "
            "Install COLMAP or pass an explicit path."
        )
    return candidate


def check_colmap_available(exe: Optional[str] = None) -> str:
    """Return resolved exe if COLMAP is callable, else raise."""
    resolved = resolve_colmap_exe(exe)
    try:
        proc = subprocess.run(
            [resolved, "--help"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"COLMAP not runnable: {resolved}") from e
    # --help may exit non-zero on some builds; only fail if not found
    if proc.returncode not in (0, 1):
        # Still consider it available if help text was produced
        if not (proc.stdout or proc.stderr):
            raise RuntimeError(f"COLMAP help check failed: {proc.stderr[:500]}")
    return resolved


# ---------------------------------------------------------------------------
# Subprocess helper (safe, no shell)
# ---------------------------------------------------------------------------

def _run_colmap(
    exe: str,
    args: list[str],
    timeout: Optional[float] = None,
) -> ColmapCommandResult:
    """Run ``exe`` with *args* as a subprocess list (no shell)."""
    cmd = [exe] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ColmapCommandResult(
        command=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


# ---------------------------------------------------------------------------
# COLMAP stages
# ---------------------------------------------------------------------------

def feature_extraction(
    database_path: str | Path,
    image_path: str | Path,
    exe: Optional[str] = None,
    camera_model: str = COLMAP_CAMERA_MODEL,
    single_camera: int = COLMAP_SINGLE_CAMERA,
    use_gpu: int = COLMAP_USE_GPU,
    extra_args: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> ColmapCommandResult:
    """Run ``colmap feature_extractor``.

    Args:
        database_path: path to the COLMAP SQLite database (created if needed).
        image_path: directory containing the input images.
        exe: COLMAP executable (default from ``core.config``).
        camera_model: COLMAP camera model (default SIMPLE_RADIAL).
        single_camera: 1 to force a single shared camera.
        use_gpu: 1 to enable GPU SIFT (0 for CPU on the M3 reference).
        extra_args: additional ``--key value`` pairs appended verbatim.
        timeout: optional subprocess timeout in seconds.
    """
    resolved = resolve_colmap_exe(exe)
    db = Path(database_path)
    img = Path(image_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(img),
        "--ImageReader.camera_model", str(camera_model),
        "--ImageReader.single_camera", str(int(single_camera)),
        "--FeatureExtraction.use_gpu", str(int(use_gpu)),
    ]
    if extra_args:
        args.extend(extra_args)

    result = _run_colmap(resolved, args, timeout=timeout)
    if not result.succeeded:
        raise ColmapError("feature_extraction", result)
    return result


def sequential_matching(
    database_path: str | Path,
    exe: Optional[str] = None,
    overlap: int = COLMAP_SEQUENTIAL_OVERLAP,
    use_gpu: int = COLMAP_USE_GPU,
    extra_args: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> ColmapCommandResult:
    """Run ``colmap sequential_matcher``.

    Args:
        database_path: COLMAP database path.
        exe: COLMAP executable.
        overlap: SequentialMatching.overlap (default 10 from the benchmark).
        extra_args: additional ``--key value`` pairs.
        timeout: optional timeout in seconds.
    """
    resolved = resolve_colmap_exe(exe)
    args = [
        "sequential_matcher",
        "--database_path", str(Path(database_path)),
        "--SequentialMatching.overlap", str(int(overlap)),
        "--FeatureMatching.use_gpu", str(int(use_gpu)),
    ]
    if extra_args:
        args.extend(extra_args)

    result = _run_colmap(resolved, args, timeout=timeout)
    if not result.succeeded:
        raise ColmapError("sequential_matching", result)
    return result


def sparse_mapping(
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    exe: Optional[str] = None,
    min_num_matches: int = COLMAP_MAPPER_MIN_NUM_MATCHES,
    ba_refine_focal_length: int = COLMAP_MAPPER_BA_REFINE_FOCAL_LENGTH,
    ba_refine_principal_point: int = COLMAP_MAPPER_BA_REFINE_PRINCIPAL_POINT,
    ba_refine_extra_params: int = COLMAP_MAPPER_BA_REFINE_EXTRA_PARAMS,
    extra_args: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> ColmapCommandResult:
    """Run ``colmap mapper`` (sparse reconstruction).

    Args:
        database_path: COLMAP database path.
        image_path: image directory.
        output_path: directory where sparse models (``0/``, ``1/`` …) are written.
        exe: COLMAP executable.
        extra_args: additional ``--key value`` pairs.
        timeout: optional timeout in seconds.
    """
    resolved = resolve_colmap_exe(exe)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    args = [
        "mapper",
        "--database_path", str(Path(database_path)),
        "--image_path", str(Path(image_path)),
        "--output_path", str(out),
        "--Mapper.min_num_matches", str(int(min_num_matches)),
        "--Mapper.ba_refine_focal_length", str(int(ba_refine_focal_length)),
        "--Mapper.ba_refine_principal_point", str(int(ba_refine_principal_point)),
        "--Mapper.ba_refine_extra_params", str(int(ba_refine_extra_params)),
    ]
    if extra_args:
        args.extend(extra_args)

    result = _run_colmap(resolved, args, timeout=timeout)
    if not result.succeeded:
        raise ColmapError("sparse_mapping", result)
    return result


def model_converter(
    input_path: str | Path,
    output_path: str | Path,
    output_type: str = "TXT",
    exe: Optional[str] = None,
    timeout: Optional[float] = None,
) -> ColmapCommandResult:
    """Run ``colmap model_converter`` to export a sparse model to text.

    The pipeline's ``core.colmap_io`` readers expect the text format
    (``cameras.txt``, ``images.txt``, ``points3D.txt``); this helper wraps
    the binary-to-text conversion.
    """
    resolved = resolve_colmap_exe(exe)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    args = [
        "model_converter",
        "--input_path", str(Path(input_path)),
        "--output_path", str(out),
        "--output_type", str(output_type),
    ]
    result = _run_colmap(resolved, args, timeout=timeout)
    if not result.succeeded:
        raise ColmapError("model_converter", result)
    return result


# ---------------------------------------------------------------------------
# High-level orchestration (optional convenience)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColmapRunConfig:
    """Configuration for :func:`run_colmap`."""

    exe: str = COLMAP_EXE
    camera_model: str = COLMAP_CAMERA_MODEL
    single_camera: int = COLMAP_SINGLE_CAMERA
    sequential_overlap: int = COLMAP_SEQUENTIAL_OVERLAP
    use_gpu: int = COLMAP_USE_GPU
    min_num_matches: int = COLMAP_MAPPER_MIN_NUM_MATCHES
    ba_refine_focal_length: int = COLMAP_MAPPER_BA_REFINE_FOCAL_LENGTH
    ba_refine_principal_point: int = COLMAP_MAPPER_BA_REFINE_PRINCIPAL_POINT
    ba_refine_extra_params: int = COLMAP_MAPPER_BA_REFINE_EXTRA_PARAMS


@dataclass(frozen=True)
class ColmapRunResult:
    """Collected outputs of :func:`run_colmap`."""

    database_path: Path
    sparse_dir: Path
    sparse_txt_dir: Optional[Path]
    stages: dict  # stage name -> ColmapCommandResult


def run_colmap(
    image_path: str | Path,
    output_dir: str | Path,
    config: Optional[ColmapRunConfig] = None,
    export_txt: bool = True,
) -> ColmapRunResult:
    """Run the full COLMAP pipeline: extraction → matching → mapping.

    This is the high-level orchestration used by the pipeline. Individual
    stages remain independently callable via the stage functions above.

    On success, ``output_dir`` will contain ``database.db`` and
    ``sparse/0/`` (and optionally a text export). Raises :class:`ColmapError`
    on any stage failure.
    """
    if config is None:
        config = ColmapRunConfig()
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / PIPELINE_DATABASE_NAME
    sparse_dir = output_dir / PIPELINE_SPARSE_SUBDIR

    stages: dict[str, ColmapCommandResult] = {}

    stages["feature_extraction"] = feature_extraction(
        database_path=database_path,
        image_path=image_path,
        exe=config.exe,
        camera_model=config.camera_model,
        single_camera=config.single_camera,
        use_gpu=config.use_gpu,
    )
    stages["sequential_matching"] = sequential_matching(
        database_path=database_path,
        exe=config.exe,
        overlap=config.sequential_overlap,
        use_gpu=config.use_gpu,
    )
    stages["sparse_mapping"] = sparse_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_dir,
        exe=config.exe,
        min_num_matches=config.min_num_matches,
        ba_refine_focal_length=config.ba_refine_focal_length,
        ba_refine_principal_point=config.ba_refine_principal_point,
        ba_refine_extra_params=config.ba_refine_extra_params,
    )

    sparse_txt_dir: Optional[Path] = None
    if export_txt:
        # Use the largest model (usually sparse/0) for text export
        model_src = _largest_sparse_model(sparse_dir)
        if model_src is not None:
            sparse_txt_dir = output_dir / "sparse_txt"
            stages["model_converter"] = model_converter(
                input_path=model_src,
                output_path=sparse_txt_dir,
                output_type="TXT",
                exe=config.exe,
            )

    return ColmapRunResult(
        database_path=database_path,
        sparse_dir=sparse_dir,
        sparse_txt_dir=sparse_txt_dir,
        stages=stages,
    )


def _largest_sparse_model(sparse_dir: Path) -> Optional[Path]:
    """Return the sparse model subdir with the most points (or None)."""
    if not sparse_dir.exists():
        return None
    candidates = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Prefer the model whose images.bin is largest, fallback to first
    def _score(p: Path) -> int:
        img = p / "images.bin"
        return img.stat().st_size if img.exists() else -1
    return max(candidates, key=_score)
