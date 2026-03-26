from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

from .utils import append_jsonl, ensure_dir, read_json, utc_timestamp, write_json
from .version import __version__


@dataclass(frozen=True)
class RunContext:
    run_root: Path
    run_dir: Path
    session_id: str
    session_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    explanations_dir: Path
    predictions_dir: Path
    failures_dir: Path
    session_reports_dir: Path
    session_explanations_dir: Path
    session_predictions_dir: Path
    session_failures_dir: Path
    metrics_path: Path
    step_metrics_path: Path
    epoch_metrics_path: Path
    config_snapshot_path: Path
    manifest_snapshot_path: Path
    summary_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    status_path: Path
    session_status_path: Path
    registry_path: Path


def _new_session_id() -> str:
    return "%s-%d" % (utc_timestamp().replace("-", "").replace(":", "").replace("T", "-").replace("Z", ""), time.time_ns() % 1000000)


def _status_payload(context: RunContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    data.setdefault("run_root", str(context.run_root))
    data.setdefault("run_dir", str(context.run_dir))
    data.setdefault("session_id", context.session_id)
    data.setdefault("updated_at", utc_timestamp())
    data.setdefault("model_version", __version__)
    return data


def _merge_existing_status(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = read_json(path) if path.exists() else {}
    merged = dict(existing)
    merged.update(dict(payload))
    return merged


def _write_status_copy(context: RunContext, payload: Mapping[str, Any]) -> None:
    write_json(context.status_path, payload)
    write_json(context.session_status_path, payload)
    append_jsonl(
        context.registry_path,
        {
            "event": "run_status",
            "timestamp": payload.get("updated_at", utc_timestamp()),
            "run_dir": str(context.run_dir),
            "session_id": context.session_id,
            "stage": payload.get("stage"),
            "variant": payload.get("variant"),
            "status": payload.get("status"),
            "report_path": payload.get("report_path"),
            "summary_path": payload.get("summary_path"),
            "latest_checkpoint_path": payload.get("latest_checkpoint_path"),
            "best_checkpoint_path": payload.get("best_checkpoint_path"),
            "failed_at": payload.get("failed_at"),
            "completed_at": payload.get("completed_at"),
            "epoch": payload.get("epoch", 0),
            "global_step": payload.get("global_step", 0),
            "model_version": payload.get("model_version", __version__),
        },
    )


def create_run_context(run_root: Path | str, stage: str, variant: str) -> RunContext:
    run_root = ensure_dir(Path(run_root))
    run_dir = ensure_dir(run_root / ("%s-%s" % (stage, variant)))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    reports_dir = ensure_dir(run_dir / "reports")
    explanations_dir = ensure_dir(run_dir / "explanations")
    predictions_dir = ensure_dir(run_dir / "predictions")
    failures_dir = ensure_dir(run_dir / "failures")
    session_id = _new_session_id()
    session_dir = ensure_dir(run_dir / "sessions" / session_id)
    session_reports_dir = ensure_dir(session_dir / "reports")
    session_explanations_dir = ensure_dir(session_dir / "explanations")
    session_predictions_dir = ensure_dir(session_dir / "predictions")
    session_failures_dir = ensure_dir(session_dir / "failures")
    metrics_path = run_dir / "metrics.jsonl"
    epoch_metrics_path = run_dir / "epoch_metrics.jsonl"
    context = RunContext(
        run_root=run_root,
        run_dir=run_dir,
        session_id=session_id,
        session_dir=session_dir,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
        explanations_dir=explanations_dir,
        predictions_dir=predictions_dir,
        failures_dir=failures_dir,
        session_reports_dir=session_reports_dir,
        session_explanations_dir=session_explanations_dir,
        session_predictions_dir=session_predictions_dir,
        session_failures_dir=session_failures_dir,
        metrics_path=metrics_path,
        step_metrics_path=metrics_path,
        epoch_metrics_path=epoch_metrics_path,
        config_snapshot_path=run_dir / "config_snapshot.json",
        manifest_snapshot_path=run_dir / "manifest_snapshot.json",
        summary_path=reports_dir / "training_summary.json",
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        status_path=run_dir / "run_status.json",
        session_status_path=session_dir / "run_status.json",
        registry_path=run_root / "run_registry.jsonl",
    )
    update_run_status(
        context,
        {
            "stage": stage,
            "variant": variant,
            "status": "created",
            "started_at": utc_timestamp(),
            "latest_checkpoint_path": str(context.latest_checkpoint_path) if context.latest_checkpoint_path.exists() else None,
            "best_checkpoint_path": str(context.best_checkpoint_path) if context.best_checkpoint_path.exists() else None,
            "metrics_path": str(context.metrics_path),
            "summary_path": str(context.summary_path),
        },
    )
    return context


def update_run_status(context: RunContext, payload: Mapping[str, Any]) -> Path:
    data = _status_payload(context, payload)
    merged = _merge_existing_status(context.status_path, data)
    merged.setdefault("started_at", data.get("started_at", utc_timestamp()))
    merged["updated_at"] = data.get("updated_at", utc_timestamp())
    _write_status_copy(context, merged)
    return context.status_path


def snapshot_run_config(context: RunContext, payload: Mapping[str, Any]) -> Path:
    data = dict(payload)
    data["captured_at"] = utc_timestamp()
    write_json(context.session_dir / "config_snapshot.json", data)
    return write_json(context.config_snapshot_path, data)


def snapshot_manifest(context: RunContext, payload: Mapping[str, Any]) -> Path:
    write_json(context.session_dir / "manifest_snapshot.json", dict(payload))
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
    data.setdefault("model_version", __version__)
    write_json(context.session_reports_dir / "training_summary.json", data)
    path = write_json(context.summary_path, data)
    update_run_status(
        context,
        {
            "summary_path": str(path),
            "report_path": str(data.get("report_path")) if data.get("report_path") else None,
            "latest_checkpoint_path": data.get("latest_checkpoint_path"),
            "best_checkpoint_path": data.get("best_checkpoint_path"),
            "latest_usable_checkpoint_path": data.get("latest_checkpoint_path"),
            "epoch": int(data.get("epoch", 0) or 0),
            "global_step": int(data.get("global_step", 0) or 0),
        },
    )
    return path


def load_run_status(path: Path | str) -> Mapping[str, Any]:
    return read_json(Path(path))
