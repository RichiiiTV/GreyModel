from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .utils import append_jsonl, ensure_dir, utc_timestamp, write_json


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    explanations_dir: Path
    metrics_path: Path
    step_metrics_path: Path
    epoch_metrics_path: Path
    config_snapshot_path: Path
    manifest_snapshot_path: Path
    summary_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path


def create_run_context(run_root: Path | str, stage: str, variant: str) -> RunContext:
    run_dir = ensure_dir(Path(run_root) / ("%s-%s" % (stage, variant)))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    reports_dir = ensure_dir(run_dir / "reports")
    explanations_dir = ensure_dir(run_dir / "explanations")
    metrics_path = run_dir / "metrics.jsonl"
    epoch_metrics_path = run_dir / "epoch_metrics.jsonl"
    return RunContext(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
        explanations_dir=explanations_dir,
        metrics_path=metrics_path,
        step_metrics_path=metrics_path,
        epoch_metrics_path=epoch_metrics_path,
        config_snapshot_path=run_dir / "config_snapshot.json",
        manifest_snapshot_path=run_dir / "manifest_snapshot.json",
        summary_path=reports_dir / "training_summary.json",
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
    )


def snapshot_run_config(context: RunContext, payload: Mapping[str, Any]) -> Path:
    data = dict(payload)
    data["captured_at"] = utc_timestamp()
    return write_json(context.config_snapshot_path, data)


def snapshot_manifest(context: RunContext, payload: Mapping[str, Any]) -> Path:
    return write_json(context.manifest_snapshot_path, payload)


def log_metrics(context: RunContext, payload: Mapping[str, Any]) -> Path:
    record = dict(payload)
    record["timestamp"] = utc_timestamp()
    return append_jsonl(context.metrics_path, record)


def log_step_metrics(context: RunContext, payload: Mapping[str, Any]) -> Path:
    record = dict(payload)
    record["timestamp"] = utc_timestamp()
    record.setdefault("event", "step")
    return append_jsonl(context.step_metrics_path, record)


def log_epoch_metrics(context: RunContext, payload: Mapping[str, Any]) -> Path:
    record = dict(payload)
    record["timestamp"] = utc_timestamp()
    record.setdefault("event", "epoch")
    return append_jsonl(context.epoch_metrics_path, record)


def write_summary(context: RunContext, payload: Mapping[str, Any]) -> Path:
    data = dict(payload)
    data["written_at"] = utc_timestamp()
    return write_json(context.summary_path, data)
