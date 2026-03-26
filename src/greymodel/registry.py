from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .types import FailureRecord, RunStatusRecord
from .utils import read_json


def _coerce_run_status(payload: Mapping[str, Any]) -> RunStatusRecord:
    return RunStatusRecord(
        run_dir=str(payload.get("run_dir")),
        stage=str(payload.get("stage", "unknown")),
        variant=str(payload.get("variant", "base")),
        status=str(payload.get("status", "unknown")),
        run_root=payload.get("run_root"),
        session_id=payload.get("session_id"),
        manifest_path=payload.get("manifest_path"),
        index_path=payload.get("index_path"),
        started_at=payload.get("started_at"),
        updated_at=payload.get("updated_at"),
        completed_at=payload.get("completed_at"),
        failed_at=payload.get("failed_at"),
        latest_checkpoint_path=payload.get("latest_checkpoint_path"),
        best_checkpoint_path=payload.get("best_checkpoint_path"),
        latest_usable_checkpoint_path=payload.get("latest_usable_checkpoint_path"),
        report_path=payload.get("report_path"),
        summary_path=payload.get("summary_path"),
        metrics_path=payload.get("metrics_path"),
        epoch=int(payload.get("epoch", 0) or 0),
        global_step=int(payload.get("global_step", 0) or 0),
        model_version=payload.get("model_version"),
        distributed_strategy=payload.get("distributed_strategy"),
        extra_paths=dict(payload.get("extra_paths", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def list_run_statuses(run_root: Path | str) -> list[RunStatusRecord]:
    run_root = Path(run_root)
    session_status_paths = sorted(run_root.glob("*-*/sessions/*/run_status.json"))
    if not session_status_paths:
        session_status_paths = sorted(run_root.glob("*-*/run_status.json"))
    rows = [_coerce_run_status(read_json(path)) for path in session_status_paths]
    rows.sort(key=lambda row: (row.updated_at or "", row.run_dir, row.session_id or ""), reverse=True)
    return rows


def latest_run_status(run_root: Path | str, stage: str | None = None, variant: str | None = None) -> RunStatusRecord | None:
    rows = list_run_statuses(run_root)
    for row in rows:
        if stage is not None and row.stage != stage:
            continue
        if variant is not None and row.variant != variant:
            continue
        return row
    return None


def list_failure_records(run_root: Path | str) -> list[FailureRecord]:
    run_root = Path(run_root)
    failure_paths = sorted(run_root.glob("*-*/sessions/*/failures/*/failure.json"))
    records: list[FailureRecord] = []
    for path in failure_paths:
        payload = read_json(path)
        records.append(FailureRecord(**payload))
    records.sort(key=lambda row: row.timestamp, reverse=True)
    return records


def compare_run_reports(left_report_path: Path | str, right_report_path: Path | str) -> dict[str, Any]:
    left = read_json(Path(left_report_path))
    right = read_json(Path(right_report_path))
    metric_keys = ("accuracy", "far", "frr", "precision", "recall", "auroc")
    left_overall = dict(left.get("overall", {}))
    right_overall = dict(right.get("overall", {}))
    delta = {}
    for key in metric_keys:
        if key not in left_overall and key not in right_overall:
            continue
        left_value = left_overall.get(key)
        right_value = right_overall.get(key)
        if left_value is None or right_value is None:
            delta[key] = None
            continue
        delta[key] = float(right_value) - float(left_value)
    return {
        "left_report_path": str(left_report_path),
        "right_report_path": str(right_report_path),
        "left_overall": left_overall,
        "right_overall": right_overall,
        "delta": delta,
    }
