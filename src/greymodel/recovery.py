from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Mapping, Optional, Sequence

from .tracking import RunContext, create_run_context, update_run_status
from .types import FailureRecord
from .utils import ensure_dir, utc_timestamp, write_json
from .version import __version__


def _failure_id(stage: str) -> str:
    return "%s-%s" % (stage, utc_timestamp().replace("-", "").replace(":", "").replace("T", "-").replace("Z", ""))


def write_failure_bundle(
    context: RunContext,
    *,
    stage: str,
    variant: str,
    exc: BaseException,
    manifest_path: Optional[Path | str] = None,
    index_path: Optional[Path | str] = None,
    latest_checkpoint_path: Optional[Path | str] = None,
    best_checkpoint_path: Optional[Path | str] = None,
    epoch: int = 0,
    global_step: int = 0,
    offending_sample_ids: Sequence[str] = (),
    partial_artifacts: Optional[Mapping[str, Any]] = None,
    resume_metadata: Optional[Mapping[str, Any]] = None,
    config_snapshot_path: Optional[Path | str] = None,
    metrics_path: Optional[Path | str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    status: str = "failed",
) -> FailureRecord:
    failure_id = _failure_id(stage)
    failure_dir = ensure_dir(context.session_failures_dir / failure_id)
    traceback_path = failure_dir / "traceback.txt"
    traceback_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    timestamp = utc_timestamp()
    payload = FailureRecord(
        failure_id=failure_id,
        stage=stage,
        variant=variant,
        status=status,
        error_type=type(exc).__name__,
        error_message=str(exc),
        run_dir=str(context.run_dir),
        failure_dir=str(failure_dir),
        traceback_path=str(traceback_path),
        timestamp=timestamp,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        index_path=str(index_path) if index_path is not None else None,
        latest_checkpoint_path=str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
        best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        epoch=int(epoch),
        global_step=int(global_step),
        offending_sample_ids=tuple(str(value) for value in offending_sample_ids),
        partial_artifacts=dict(partial_artifacts or {}),
        resume_metadata=dict(resume_metadata or {}),
        config_snapshot_path=str(config_snapshot_path) if config_snapshot_path is not None else None,
        metrics_path=str(metrics_path) if metrics_path is not None else None,
        metadata={"model_version": __version__, **dict(metadata or {})},
    )
    write_json(failure_dir / "failure.json", asdict(payload))
    update_run_status(
        context,
        {
            "status": status,
            "failed_at": timestamp,
            "latest_checkpoint_path": payload.latest_checkpoint_path,
            "best_checkpoint_path": payload.best_checkpoint_path,
            "latest_usable_checkpoint_path": payload.latest_checkpoint_path or payload.best_checkpoint_path,
            "epoch": payload.epoch,
            "global_step": payload.global_step,
            "failure_dir": str(failure_dir),
            "manifest_path": payload.manifest_path,
            "index_path": payload.index_path,
            "metrics_path": payload.metrics_path,
            "metadata": {"last_failure_id": failure_id},
        },
    )
    try:
        setattr(exc, "_greymodel_failure_written", True)
        setattr(exc, "_greymodel_failure_path", str(failure_dir / "failure.json"))
    except Exception:
        pass
    return payload


def ensure_failure_bundle(
    *,
    run_root: Path | str,
    stage: str,
    variant: str,
    exc: BaseException,
    manifest_path: Optional[Path | str] = None,
    index_path: Optional[Path | str] = None,
    latest_checkpoint_path: Optional[Path | str] = None,
    best_checkpoint_path: Optional[Path | str] = None,
    epoch: int = 0,
    global_step: int = 0,
    offending_sample_ids: Sequence[str] = (),
    partial_artifacts: Optional[Mapping[str, Any]] = None,
    resume_metadata: Optional[Mapping[str, Any]] = None,
    config_snapshot_path: Optional[Path | str] = None,
    metrics_path: Optional[Path | str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    status: str = "failed",
) -> FailureRecord:
    context = create_run_context(run_root=run_root, stage=stage, variant=variant)
    return write_failure_bundle(
        context,
        stage=stage,
        variant=variant,
        exc=exc,
        manifest_path=manifest_path,
        index_path=index_path,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        epoch=epoch,
        global_step=global_step,
        offending_sample_ids=offending_sample_ids,
        partial_artifacts=partial_artifacts,
        resume_metadata=resume_metadata,
        config_snapshot_path=config_snapshot_path,
        metrics_path=metrics_path,
        metadata=metadata,
        status=status,
    )
