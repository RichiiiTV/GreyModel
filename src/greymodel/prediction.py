from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from .data import build_dataset_manifest
from .runners import StageResult, run_prediction_stage


def run_batch_prediction_stage(
    manifest_path: Path | str | None = None,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    evidence_policy: str = "bad",
    input_dir: Path | str | None = None,
    model_profile: str | Path | Mapping[str, object] | None = None,
    model_registry_root: Path | str | None = None,
) -> StageResult:
    if manifest_path is None:
        if input_dir is None:
            raise ValueError("run_batch_prediction_stage requires either manifest_path or input_dir.")
        staged_root = Path(run_root) / "_prediction_input"
        index = build_dataset_manifest(input_dir, output_dir=staged_root)
        manifest_path = index.manifest_path
        if index_path is None:
            index_path = index.index_path or Path(manifest_path).with_name("dataset_index.json")
    return run_prediction_stage(
        manifest_path=manifest_path,
        index_path=index_path,
        variant=variant,
        run_root=run_root,
        evidence_policy=evidence_policy,
        model_profile=model_profile,
        model_registry_root=model_registry_root,
    )
