"""flight2world.core.diagnostics

Structured diagnostics for the FLIGHT2WORLD pipeline.

Each stage writes a small JSON sidecar under
``{output_root}/diagnostics/``; :func:`collect_diagnostics` merges them.

The :class:`Diagnostics` object is intentionally simple: a list of
:class:`StageRecord` entries plus a run-level summary. No framework is
introduced.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """One pipeline stage's diagnostic record."""

    stage: str
    status: str  # "ok" | "skipped" | "failed"
    elapsed_s: float
    details: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnostics:
    """Collected diagnostics for a full pipeline run."""

    run_id: str
    output_root: str
    stages: list[StageRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def add(self, record: StageRecord) -> None:
        self.stages.append(record)

    def stage(self, name: str) -> Optional[StageRecord]:
        for s in self.stages:
            if s.stage == name:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "output_root": self.output_root,
            "created_at": self.created_at,
            "stages": [s.to_dict() for s in self.stages],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def write_stage_diagnostics(
    output_root: str | Path,
    record: StageRecord,
) -> Path:
    """Write a single stage JSON under ``{output_root}/diagnostics/``."""
    output_root = Path(output_root)
    diag_dir = output_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    path = diag_dir / f"{record.stage}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2))
    return path


def load_stage_diagnostics(output_root: str | Path, stage: str) -> Optional[dict]:
    """Load a single stage JSON if it exists, else None."""
    p = Path(output_root) / "diagnostics" / f"{stage}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class StageTimer:
    """Context manager that measures a stage's wall time.

    Usage:
        with StageTimer("frames") as t:
            ... do work ...
        record = StageRecord(stage="frames", status="ok",
                             elapsed_s=t.elapsed, details={...})
    """

    def __init__(self, stage: str):
        self.stage = stage
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed = time.monotonic() - self._t0
