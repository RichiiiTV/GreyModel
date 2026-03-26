from __future__ import annotations

from pathlib import Path
from typing import Optional

from .runners import StageResult, run_prediction_stage


def run_batch_prediction_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    evidence_policy: str = "bad",
) -> StageResult:
    return run_prediction_stage(
        manifest_path=manifest_path,
        index_path=index_path,
        variant=variant,
        run_root=run_root,
        evidence_policy=evidence_policy,
    )
