from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from greymodel.data import load_dataset_index, load_dataset_manifest, load_station_configs_from_index, station_config_for_record
from greymodel.explainability import build_explanation_bundle
from greymodel.registry import compare_run_reports, list_failure_records, list_run_statuses
from greymodel.ui import UIExecutionDefaults, build_greymodel_job_command, build_slurm_submission_command, format_shell_command
from greymodel.ui_models import benchmark_profile_runtime, build_runtime_for_profile, predict_record_with_profile, save_benchmark_result
from greymodel.ui_workspace import (
    ModelProfile,
    WorkspaceConfig,
    delete_model_profile,
    load_workspace,
    save_workspace,
    set_recent_dataset_index,
    set_recent_run_dir,
    upsert_model_profile,
    workspace_path_for,
)
from greymodel.utils import ensure_dir, load_uint8_grayscale, read_json, read_jsonl, utc_timestamp, write_json, write_jsonl


PAGE_ORDER = ["Home", "Datasets", "Models", "Train", "Runs", "Predict & Review", "Explain", "Failures", "Settings"]

STAGE_LABELS = {
    "pretrain": "1. Public Pretraining",
    "domain-adapt": "2. Adapt To Production Images",
    "finetune": "3. Train Final Inspector",
    "calibrate": "4. Calibrate Decision Thresholds",
}

STAGE_HELP = {
    "pretrain": "Use a public or unlabeled dataset to teach the model general grayscale defect structure before it sees production labels.",
    "domain-adapt": "Use unlabeled production images to adapt the model to your real station, camera, and product appearance.",
    "finetune": "Use labeled production data to train the final good-vs-bad inspector and defect-family outputs.",
    "calibrate": "Build station-specific thresholds after training so the final pass/fail decision is stable on your line.",
}


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", default="artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--workspace-path", default=None)
    parser.add_argument("--default-execution-backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument("--slurm-mem", default="50G")
    parser.add_argument("--slurm-gres", default="gpu:8")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-queue", default="")
    parser.add_argument("--slurm-nproc-per-node", type=int, default=8)
    parser.add_argument("--slurm-python", default=sys.executable)
    return parser.parse_known_args(argv)


@lru_cache(maxsize=32)
def _scan_dataset_indexes_cached(data_root_value: str, deep: bool = False) -> tuple[str, ...]:
    data_root = Path(data_root_value)
    if not data_root.exists():
        return ()
    if deep:
        return tuple(sorted(str(path) for path in data_root.rglob("dataset_index.json")))
    patterns = (
        "dataset_index.json",
        "*/dataset_index.json",
        "*/*/dataset_index.json",
        "*/*/*/dataset_index.json",
    )
    discovered: set[str] = set()
    for pattern in patterns:
        for path in data_root.glob(pattern):
            discovered.add(str(path))
    return tuple(sorted(discovered))


def _find_dataset_indexes(data_root: Path, *, deep: bool = False) -> list[Path]:
    return [Path(value) for value in _scan_dataset_indexes_cached(str(data_root), deep=deep)]


def _known_dataset_indexes(workspace: WorkspaceConfig, data_root: Path) -> list[Path]:
    discovered = {path for path in _find_dataset_indexes(data_root, deep=False)}
    active_candidates = [workspace.active_dataset_index, *workspace.recent_dataset_indexes]
    for candidate in active_candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            discovered.add(path)
    return sorted(discovered)


def _job_root(run_root: Path) -> Path:
    return ensure_dir(run_root / "ui_jobs")


def _review_root(run_root: Path) -> Path:
    return ensure_dir(run_root / "ui_reviews")


def _execution_defaults_from_args(args: argparse.Namespace) -> UIExecutionDefaults:
    return UIExecutionDefaults(
        execution_backend=args.default_execution_backend,
        slurm_cpus=int(args.slurm_cpus),
        slurm_mem=str(args.slurm_mem),
        slurm_gres=str(args.slurm_gres),
        slurm_partition=str(args.slurm_partition),
        slurm_queue=str(args.slurm_queue),
        slurm_nproc_per_node=int(args.slurm_nproc_per_node),
        slurm_python=str(args.slurm_python),
    )


def _execution_defaults_from_workspace(workspace: WorkspaceConfig) -> UIExecutionDefaults:
    return UIExecutionDefaults(
        execution_backend=workspace.default_execution_backend,
        slurm_cpus=workspace.slurm_cpus,
        slurm_mem=workspace.slurm_mem,
        slurm_gres=workspace.slurm_gres,
        slurm_partition=workspace.slurm_partition,
        slurm_queue=workspace.slurm_queue,
        slurm_nproc_per_node=workspace.slurm_nproc_per_node,
        slurm_python=sys.executable,
    )


def _create_ui_job_paths(run_root: Path, kind: str) -> tuple[Path, Path]:
    job_dir = _job_root(run_root)
    stamp = utc_timestamp().replace(":", "").replace("-", "").replace("T", "-").replace("Z", "")
    return job_dir / ("%s-%s.log" % (kind, stamp)), job_dir / ("%s-%s.json" % (kind, stamp))


def _launch_local_job(command: list[str], cwd: Path, run_root: Path, kind: str, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
    log_path, metadata_path = _create_ui_job_paths(run_root, kind)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=handle, stderr=handle)
    payload = {"backend": "local", "pid": int(process.pid), "kind": kind, "command": command, "log_path": str(log_path), "metadata_path": str(metadata_path), "created_at": utc_timestamp(), **dict(metadata or {})}
    write_json(metadata_path, payload)
    return payload


def _launch_slurm_job(
    command: list[str],
    cwd: Path,
    run_root: Path,
    kind: str,
    *,
    cpus: int,
    mem: str,
    gres: str,
    partition: str = "",
    queue: str = "",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    log_path, metadata_path = _create_ui_job_paths(run_root, kind)
    submit_command = build_slurm_submission_command(
        inner_command=command,
        repo_root=cwd,
        cpus=cpus,
        mem=mem,
        gres=gres,
        partition=partition or None,
        queue=queue or None,
        job_name="greymodel-%s" % kind,
        log_path=log_path,
    )
    result = subprocess.run(submit_command, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("Slurm submission failed with exit code %d: %s" % (int(result.returncode), (result.stderr or result.stdout).strip()))
    stdout = (result.stdout or "").strip()
    payload = {"backend": "slurm", "job_id": stdout.split(";", 1)[0] if stdout else "", "kind": kind, "command": command, "submit_command": submit_command, "log_path": str(log_path), "metadata_path": str(metadata_path), "created_at": utc_timestamp(), "submit_stdout": stdout, "submit_stderr": (result.stderr or "").strip(), **dict(metadata or {})}
    write_json(metadata_path, payload)
    return payload


def _launch_managed_job(
    task_tokens: list[str],
    *,
    cwd: Path,
    run_root: Path,
    kind: str,
    execution_backend: str,
    nproc_per_node: int = 1,
    execution_defaults: UIExecutionDefaults,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    python_executable = execution_defaults.slurm_python if execution_backend == "slurm" else sys.executable
    command = build_greymodel_job_command(task_tokens, python_executable=python_executable, nproc_per_node=max(int(nproc_per_node), 1))
    if execution_backend == "slurm":
        return _launch_slurm_job(command, cwd, run_root, kind, cpus=execution_defaults.slurm_cpus, mem=execution_defaults.slurm_mem, gres=execution_defaults.slurm_gres, partition=execution_defaults.slurm_partition, queue=execution_defaults.slurm_queue, metadata=metadata)
    return _launch_local_job(command, cwd, run_root, kind, metadata=metadata)


def _ui_jobs(run_root: Path) -> list[dict[str, object]]:
    jobs = []
    for metadata_path in sorted(_job_root(run_root).glob("*.json"), reverse=True):
        try:
            jobs.append(read_json(metadata_path))
        except Exception:
            continue
    return jobs


def _tail_text(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-max(int(lines), 1) :])


def _selectbox_index(options: Iterable[str], default_value: str) -> int:
    option_list = list(options)
    try:
        return option_list.index(default_value)
    except ValueError:
        return 0


def _inject_theme(st) -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        .gm-title { font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.1rem; }
        .gm-subtitle { color: #475569; margin-bottom: 0.9rem; }
        .gm-card {
            border-radius: 16px;
            padding: 0.9rem 1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.92));
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }
        .gm-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #64748b;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(st, label: str, value: object, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="gm-card">
          <div class="gm-kicker">{label}</div>
          <div style="font-size: 1.6rem; font-weight: 800; margin-top: 0.2rem;">{value}</div>
          <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.25rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _simple_mode(workspace: WorkspaceConfig) -> bool:
    return str(getattr(workspace, "ui_mode", "vision_engineer")) != "advanced"


def _profile_summary(profile: ModelProfile) -> dict[str, str]:
    if profile.profile_id == "prod_fast_native":
        return {
            "title": "Production Fast Path",
            "purpose": "Best choice for line-side screening and low-latency review.",
            "details": "Fast screen first, then patch refinement only on uncertain or suspicious samples.",
        }
    if profile.profile_id == "review_native_base":
        return {
            "title": "Review Base",
            "purpose": "Best choice for offline review and richer evidence.",
            "details": "Higher-capacity native model with slower but richer outputs.",
        }
    if profile.profile_id == "review_native_lite":
        return {
            "title": "Review Lite",
            "purpose": "Good default when you want a lighter review model.",
            "details": "Compact native model with the same main output contract as Base.",
        }
    if profile.is_huggingface:
        return {
            "title": profile.display_name or profile.profile_id,
            "purpose": "External Hugging Face model profile.",
            "details": "Useful for experiments, comparisons, or customer-specific backends.",
        }
    return {
        "title": profile.display_name or profile.profile_id,
        "purpose": profile.notes or "Model profile",
        "details": "Backend: %s | Runtime: %s" % (profile.backend_family, profile.runtime_engine),
    }


def _bundle_preview_paths(bundle_payload: Mapping[str, object]) -> list[Path]:
    keys = (
        "image_path",
        "valid_mask_path",
        "attribution_path",
        "heatmap_path",
        "local_heatmap_path",
        "global_heatmap_path",
    )
    paths = []
    for key in keys:
        value = bundle_payload.get(key)
        if value not in (None, ""):
            path = Path(str(value))
            if path.exists():
                paths.append(path)
    return paths


def collect_ui_state(run_root: Path | str = "artifacts", data_root: Path | str = "data", workspace_path: Path | str | None = None) -> dict[str, object]:
    run_root = Path(run_root)
    data_root = Path(data_root)
    workspace = load_workspace(run_root=run_root, data_root=data_root, workspace_path=workspace_path)
    return {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "workspace_path": str(Path(workspace_path) if workspace_path is not None else workspace_path_for(run_root)),
        "workspace": {
            "workspace_name": workspace.workspace_name,
            "ui_mode": workspace.ui_mode,
            "active_dataset_index": workspace.active_dataset_index,
            "active_model_profile": workspace.active_model_profile,
            "profile_count": len(workspace.model_profiles),
        },
        "runs": [row.__dict__ for row in list_run_statuses(run_root)],
        "failures": [row.__dict__ for row in list_failure_records(run_root)],
        "datasets": [str(path) for path in _find_dataset_indexes(data_root)] if data_root.exists() else [],
        "jobs": _ui_jobs(run_root),
        "model_profiles": [profile.__dict__ for profile in workspace.model_profiles.values()],
    }


def _render_execution_settings(
    st,
    *,
    key_prefix: str,
    execution_defaults: UIExecutionDefaults,
    show_nproc: bool = False,
    nproc_default: int = 1,
) -> dict[str, object]:
    backend_options = ["local", "slurm"]
    execution_backend = st.selectbox(
        "Execution Backend",
        backend_options,
        index=_selectbox_index(backend_options, execution_defaults.execution_backend),
        key="%s_backend" % key_prefix,
    )
    nproc_per_node = 1
    if show_nproc:
        nproc_per_node = int(st.number_input("Processes / GPUs", min_value=1, value=max(int(nproc_default), 1), key="%s_nproc" % key_prefix))
    if execution_backend != "slurm":
        return {"execution_backend": execution_backend, "nproc_per_node": int(nproc_per_node), "execution_defaults": execution_defaults}
    slurm_defaults = UIExecutionDefaults(
        execution_backend="slurm",
        slurm_cpus=int(st.number_input("Slurm CPUs (-c)", min_value=1, value=max(int(execution_defaults.slurm_cpus), 1), key="%s_slurm_cpus" % key_prefix)),
        slurm_mem=st.text_input("Slurm Memory (--mem)", execution_defaults.slurm_mem, key="%s_slurm_mem" % key_prefix),
        slurm_gres=st.text_input("Slurm GRES (--gres)", execution_defaults.slurm_gres, key="%s_slurm_gres" % key_prefix),
        slurm_partition=st.text_input("Slurm Partition (-p)", execution_defaults.slurm_partition, key="%s_slurm_partition" % key_prefix),
        slurm_queue=st.text_input("Slurm Queue (-q)", execution_defaults.slurm_queue, key="%s_slurm_queue" % key_prefix),
        slurm_nproc_per_node=max(int(nproc_per_node), 1),
        slurm_python=st.text_input("Slurm Python", execution_defaults.slurm_python, key="%s_slurm_python" % key_prefix),
    )
    return {"execution_backend": execution_backend, "nproc_per_node": int(nproc_per_node) if show_nproc else int(slurm_defaults.slurm_nproc_per_node), "execution_defaults": slurm_defaults}


def _preview_job_commands(
    task_tokens: list[str],
    *,
    repo_root: Path,
    run_root: Path,
    kind: str,
    execution_backend: str,
    nproc_per_node: int,
    execution_defaults: UIExecutionDefaults,
) -> tuple[list[str], list[str]]:
    python_executable = execution_defaults.slurm_python if execution_backend == "slurm" else sys.executable
    inner_command = build_greymodel_job_command(task_tokens, python_executable=python_executable, nproc_per_node=max(int(nproc_per_node), 1))
    if execution_backend != "slurm":
        return inner_command, inner_command
    preview_log_path = Path(run_root) / "ui_jobs" / ("%s-preview.log" % kind)
    submit_command = build_slurm_submission_command(
        inner_command=inner_command,
        repo_root=repo_root,
        cpus=execution_defaults.slurm_cpus,
        mem=execution_defaults.slurm_mem,
        gres=execution_defaults.slurm_gres,
        partition=execution_defaults.slurm_partition or None,
        queue=execution_defaults.slurm_queue or None,
        job_name="greymodel-%s" % kind,
        log_path=preview_log_path,
    )
    return inner_command, submit_command


def _render_job_history(st, run_root: Path, kind: str) -> None:
    jobs = [row for row in _ui_jobs(run_root) if row.get("kind") == kind]
    if not jobs:
        return
    st.subheader("Submitted Jobs")
    st.dataframe(jobs[:20], use_container_width=True)
    log_choices = [job["log_path"] for job in jobs if job.get("log_path")]
    if log_choices:
        selected_log = st.selectbox("%s Log" % kind.title(), log_choices, format_func=str, key="%s_log" % kind)
        if selected_log:
            st.code(_tail_text(Path(str(selected_log))), language="text")


def _read_last_jsonl_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        rows = read_jsonl(path)
    except Exception:
        return {}
    return dict(rows[-1]) if rows else {}


def _run_status_for_dir(run_dir: Path) -> dict[str, object]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return {}
    try:
        return dict(read_json(status_path))
    except Exception:
        return {}


def _latest_autofit_run_status(run_root: Path) -> dict[str, object]:
    rows = [row for row in list_run_statuses(run_root) if row.stage == "autofit"]
    if not rows:
        return {}
    latest = rows[0]
    return {
        "run_dir": latest.run_dir,
        "status": latest.status,
        "updated_at": latest.updated_at,
        "epoch": latest.epoch,
        "global_step": latest.global_step,
        "best_checkpoint_path": latest.best_checkpoint_path,
        "latest_usable_checkpoint_path": latest.latest_usable_checkpoint_path,
        "report_path": latest.report_path,
        "summary_path": latest.summary_path,
        "metadata": dict(latest.metadata or {}),
    }


def _autofit_child_run_dirs(parent_run_dir: Path, variant: str) -> dict[str, Path]:
    stage_root = parent_run_dir / "stages"
    return {
        "finetune": stage_root / ("finetune-%s" % variant),
        "calibrate": stage_root / ("calibrate-%s" % variant),
        "benchmark": stage_root / ("benchmark-%s" % variant),
    }


def _active_autofit_stage(parent_status: Mapping[str, object]) -> tuple[str, dict[str, object], Path | None]:
    run_dir = parent_status.get("run_dir")
    if not run_dir:
        return "autofit", {}, None
    variant = str(Path(str(run_dir)).name).split("-", 1)[1] if "-" in Path(str(run_dir)).name else "base"
    child_dirs = _autofit_child_run_dirs(Path(str(run_dir)), variant)
    stage_rows: list[tuple[str, dict[str, object], Path]] = []
    for stage_name, stage_dir in child_dirs.items():
        status = _run_status_for_dir(stage_dir)
        if status:
            stage_rows.append((stage_name, status, stage_dir))
    if not stage_rows:
        return "autofit", dict(parent_status), Path(str(run_dir))
    running = [row for row in stage_rows if str(row[1].get("status", "")).lower() in {"created", "running"}]
    if running:
        running.sort(key=lambda item: str(item[1].get("updated_at", "")), reverse=True)
        return running[0]
    stage_rows.sort(key=lambda item: str(item[1].get("updated_at", "")), reverse=True)
    return stage_rows[0]


def _autofit_live_payload(run_root: Path) -> dict[str, object]:
    parent_status = _latest_autofit_run_status(run_root)
    if not parent_status:
        return {}
    stage_name, stage_status, stage_dir = _active_autofit_stage(parent_status)
    metrics_path = Path(str(stage_status.get("metrics_path") or (stage_dir / "metrics.jsonl" if stage_dir is not None else "")))
    epoch_metrics_path = stage_dir / "epoch_metrics.jsonl" if stage_dir is not None else Path()
    latest_step = _read_last_jsonl_record(metrics_path) if metrics_path else {}
    latest_epoch = _read_last_jsonl_record(epoch_metrics_path) if stage_dir is not None else {}
    summary_path = _latest_autofit_summary(run_root)
    summary = read_json(summary_path) if summary_path is not None and summary_path.exists() else {}
    return {
        "parent_status": parent_status,
        "stage_name": stage_name,
        "stage_status": stage_status,
        "stage_dir": str(stage_dir) if stage_dir is not None else None,
        "latest_step": latest_step,
        "latest_epoch": latest_epoch,
        "summary": summary,
    }


def _render_autofit_live_monitor(st, run_root: Path, *, key_prefix: str) -> None:
    st.markdown("### 4. Watch Progress")
    live_updates = st.checkbox("Live updates", value=True, key="%s_live" % key_prefix)
    refresh_seconds = int(
        st.selectbox(
            "Refresh every",
            [2, 5, 10],
            index=0,
            format_func=lambda value: "%ss" % value,
            key="%s_refresh_seconds" % key_prefix,
        )
    )

    def _body() -> None:
        _render_job_history(st, run_root, "autofit")
        payload = _autofit_live_payload(run_root)
        if not payload:
            st.info("No AutoFit run found yet.")
            return
        parent_status = dict(payload["parent_status"])
        stage_status = dict(payload["stage_status"])
        latest_step = dict(payload["latest_step"])
        latest_epoch = dict(payload["latest_epoch"])
        summary = dict(payload["summary"])
        stage_name = str(payload["stage_name"])
        cols = st.columns(5)
        _metric_card(cols[0], "Run Status", parent_status.get("status", "unknown"), "AutoFit workflow state")
        _metric_card(cols[1], "Current Stage", STAGE_LABELS.get(stage_name, stage_name), "Stage writing logs right now")
        _metric_card(cols[2], "Epoch", stage_status.get("epoch", latest_epoch.get("epoch", 0)), "Latest completed epoch")
        _metric_card(cols[3], "Step", stage_status.get("global_step", latest_step.get("global_step", 0)), "Latest global step")
        _metric_card(cols[4], "Updated", parent_status.get("updated_at", "n/a"), "Last artifact update")
        st.markdown("### Current Performance")
        perf_cols = st.columns(5)
        _metric_card(perf_cols[0], "Train Loss", latest_step.get("loss", latest_epoch.get("train_loss", "n/a")), "Latest optimizer step")
        _metric_card(perf_cols[1], "Val Loss", latest_step.get("val_loss", latest_epoch.get("val_loss", "n/a")), "Latest validation value")
        _metric_card(perf_cols[2], "Learning Rate", latest_step.get("learning_rate", latest_epoch.get("learning_rate", "n/a")), "Current LR")
        _metric_card(perf_cols[3], "Samples / Sec", latest_step.get("samples_per_second", "n/a"), "Throughput")
        if summary:
            overall = dict(summary.get("overall", {}))
            _metric_card(perf_cols[4], "Val AUROC", overall.get("auroc", "n/a"), "Final validation benchmark")
        else:
            _metric_card(perf_cols[4], "Val AUROC", "n/a", "Available after benchmark")
        log_path = None
        for job in _ui_jobs(run_root):
            if job.get("kind") == "autofit" and job.get("log_path"):
                log_path = Path(str(job["log_path"]))
                break
        if log_path is not None and log_path.exists():
            st.code(_tail_text(log_path), language="text")
        if summary:
            overall = dict(summary.get("overall", {}))
            defect_family = dict(summary.get("defect_family_bad_only", {}))
            st.markdown("### 5. Review Performance")
            cols = st.columns(5)
            _metric_card(cols[0], "Accuracy", overall.get("accuracy", "n/a"), "Validation set")
            _metric_card(cols[1], "AUROC", overall.get("auroc", "n/a"), "Binary quality")
            _metric_card(cols[2], "FAR", overall.get("far", "n/a"), "False accept rate")
            _metric_card(cols[3], "FRR", overall.get("frr", "n/a"), "False reject rate")
            _metric_card(cols[4], "Bad Top1", defect_family.get("top1_accuracy", "n/a"), "Bad-only defect family")
            st.write(
                {
                    "best_checkpoint_path": summary.get("best_checkpoint_path"),
                    "report_path": summary.get("report_path"),
                    "calibration_report_path": summary.get("calibration_report_path"),
                    "log_path": summary.get("log_path"),
                }
            )

    if live_updates and isinstance(st, types.ModuleType) and callable(getattr(st, "fragment", None)):
        @st.fragment(run_every=refresh_seconds)
        def _live_fragment() -> None:
            _body()

        _live_fragment()
    else:
        _body()


def _default_autofit_data_path(workspace: WorkspaceConfig) -> str:
    if workspace.active_manifest:
        return str(workspace.active_manifest)
    if workspace.active_dataset_index:
        try:
            return str(load_dataset_index(workspace.active_dataset_index).manifest_path)
        except Exception:
            pass
    return str(Path(workspace.data_root) / "production")


def _latest_autofit_summary(run_root: Path) -> Path | None:
    candidates = sorted(run_root.glob("autofit-*/reports/autofit_summary.json"))
    return candidates[-1] if candidates else None


def _render_autofit_launcher(
    st,
    *,
    repo_root: Path,
    run_root: Path,
    workspace: WorkspaceConfig,
    execution_defaults: UIExecutionDefaults,
    title: str,
    description: str,
) -> None:
    st.markdown("### %s" % title)
    st.caption(description)
    model_options = ["base", "lite"]
    active_profile = _active_profile(workspace)
    default_model = "lite" if str(getattr(active_profile, "native_variant", "")).lower() == "lite" else "base"
    simple_mode = _simple_mode(workspace)
    data_path = st.text_input("1. Data", value=_default_autofit_data_path(workspace), key="%s_data" % title)
    model_choice = st.selectbox("2. Model", model_options, index=_selectbox_index(model_options, default_model), key="%s_model" % title)
    if simple_mode:
        st.caption("Execution backend: `%s` from workspace settings. Open `Advanced` only if you need to change resources or hyperparameters." % execution_defaults.execution_backend)
        execution_settings = {
            "execution_backend": execution_defaults.execution_backend,
            "nproc_per_node": int(max(execution_defaults.slurm_nproc_per_node, 1)),
            "execution_defaults": execution_defaults,
        }
    else:
        execution_settings = _render_execution_settings(
            st,
            key_prefix="autofit_%s" % title.lower().replace(" ", "_"),
            execution_defaults=execution_defaults,
            show_nproc=True,
            nproc_default=max(int(execution_defaults.slurm_nproc_per_node), 1),
        )
    with st.expander("Advanced", expanded=False):
        if simple_mode:
            execution_settings = _render_execution_settings(
                st,
                key_prefix="autofit_%s" % title.lower().replace(" ", "_"),
                execution_defaults=execution_defaults,
                show_nproc=True,
                nproc_default=max(int(execution_defaults.slurm_nproc_per_node), 1),
            )
        epochs = int(st.number_input("Epochs", min_value=1, value=1, key="%s_epochs" % title))
        batch_size = int(st.number_input("Images Per Step", min_value=1, value=2, key="%s_batch" % title))
        learning_rate = float(st.number_input("Learning Rate", min_value=1e-6, value=1e-3, format="%.6f", key="%s_lr" % title))
        num_workers = int(st.number_input("Workers", min_value=0, value=0, key="%s_workers" % title))
        precision = st.selectbox("Precision", ["auto", "fp32", "fp16", "bf16"], index=0, key="%s_precision" % title)
    task_tokens = [
        "auto",
        "fit",
        "--data",
        data_path,
        "--model",
        model_choice,
        "--run-root",
        workspace.run_root,
        "--execution",
        "local",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--num-workers",
        str(num_workers),
        "--precision",
        precision,
    ]
    _, preview_command = _preview_job_commands(
        task_tokens,
        repo_root=repo_root,
        run_root=run_root,
        kind="autofit",
        execution_backend=str(execution_settings["execution_backend"]),
        nproc_per_node=int(execution_settings["nproc_per_node"]),
        execution_defaults=execution_settings["execution_defaults"],
    )
    if not _simple_mode(workspace):
        st.code(format_shell_command(preview_command))
    if st.button("3. Start Automated Training", key="%s_start" % title):
        payload = _launch_managed_job(
            task_tokens,
            cwd=repo_root,
            run_root=run_root,
            kind="autofit",
            execution_backend=str(execution_settings["execution_backend"]),
            nproc_per_node=int(execution_settings["nproc_per_node"]),
            execution_defaults=execution_settings["execution_defaults"],
            metadata={"workflow": "autofit", "model": model_choice},
        )
        if payload.get("backend") == "slurm":
            st.success("Submitted AutoFit Slurm job %s" % payload.get("job_id"))
        else:
            st.success("Started AutoFit process %s" % payload.get("pid"))
    _render_autofit_live_monitor(st, run_root, key_prefix="%s_monitor" % title.lower().replace(" ", "_"))


def _active_profile(workspace: WorkspaceConfig) -> ModelProfile:
    if workspace.active_model_profile and workspace.active_model_profile in workspace.model_profiles:
        return workspace.model_profiles[workspace.active_model_profile]
    return next(iter(workspace.model_profiles.values()))


def _profile_ids(workspace: WorkspaceConfig) -> list[str]:
    return sorted(workspace.model_profiles.keys())


def _persist_workspace(workspace: WorkspaceConfig, workspace_path: Path | None) -> None:
    save_workspace(workspace, workspace_path)


def _render_home(st, repo_root: Path, workspace: WorkspaceConfig, run_root: Path, data_root: Path, execution_defaults: UIExecutionDefaults) -> None:
    st.markdown('<div class="gm-title">GreyModel Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="gm-subtitle">A vision-inspection console for preparing datasets, choosing models, reviewing predictions, and launching jobs.</div>', unsafe_allow_html=True)
    runs = list_run_statuses(run_root)
    failures = list_failure_records(run_root)
    datasets = _known_dataset_indexes(workspace, data_root)
    profile = _active_profile(workspace)
    cols = st.columns(4)
    _metric_card(cols[0], "Datasets", len(datasets), "Detected bundles")
    _metric_card(cols[1], "Runs", len(runs), "Tracked sessions")
    _metric_card(cols[2], "Failures", len(failures), "Quarantined bundles")
    _metric_card(cols[3], "Active Profile", profile.display_name or profile.profile_id, profile.backend_family)
    if _simple_mode(workspace):
        st.markdown("### Recommended Workflow")
        st.markdown(
            "\n".join(
                [
                    "1. Select your labeled dataset or manifest below.",
                    "2. Choose `base` or `lite` depending on the speed/accuracy tradeoff you want.",
                    "3. Start AutoFit and watch the live job log.",
                    "4. Review the final checkpoint and validation metrics once training finishes.",
                ]
            )
        )
        _render_autofit_launcher(
            st,
            repo_root=repo_root,
            run_root=run_root,
            workspace=workspace,
            execution_defaults=execution_defaults,
            title="Automated Training",
            description="Point GreyModel at your labeled data, choose a native model, and let it build manifests, split data, train, calibrate, and score the result.",
        )
    st.markdown("### Recent Runs")
    if runs:
        st.dataframe([{"stage": row.stage, "variant": row.variant, "status": row.status, "updated_at": row.updated_at, "checkpoint": row.latest_usable_checkpoint_path, "report": row.report_path} for row in runs[:10]], use_container_width=True)
    else:
        st.info("No runs yet.")
    st.markdown("### Recent Failures")
    if failures:
        st.dataframe([{"stage": row.stage, "variant": row.variant, "timestamp": row.timestamp, "error_type": row.error_type, "sample_ids": ", ".join(row.offending_sample_ids[:3])} for row in failures[:10]], use_container_width=True)
    else:
        st.info("No failure bundles found.")


def _render_datasets(st, workspace: WorkspaceConfig, data_root: Path, workspace_path: Path | None) -> None:
    st.header("Datasets")
    if _simple_mode(workspace):
        st.caption("Choose the image bundle you want to inspect or train on.")
    if st.button("Deep Scan Dataset Root", key="gm_dataset_deep_scan"):
        _scan_dataset_indexes_cached.cache_clear()
        dataset_indexes = _find_dataset_indexes(data_root, deep=True)
    else:
        dataset_indexes = _known_dataset_indexes(workspace, data_root)
    if not dataset_indexes:
        st.info("No dataset bundles found under %s." % data_root)
        return
    selected = st.selectbox("Dataset bundle", dataset_indexes, index=_selectbox_index([str(path) for path in dataset_indexes], workspace.active_dataset_index or str(dataset_indexes[0])), format_func=lambda path: str(path.relative_to(data_root)))
    workspace = set_recent_dataset_index(workspace, str(selected))
    index = load_dataset_index(selected)
    records = load_dataset_manifest(index.manifest_path)
    _persist_workspace(workspace, workspace_path)
    cols = st.columns(4)
    _metric_card(cols[0], "Records", len(records), "Manifest rows")
    _metric_card(cols[1], "Stations", len(index.station_configs), "Station configs")
    _metric_card(cols[2], "Splits", len(index.split_assignments), "Leakage-safe assignments")
    _metric_card(cols[3], "Ontology", Path(index.ontology_path).name, "Defect taxonomy")
    if _simple_mode(workspace):
        try:
            bundle_value = str(selected.relative_to(data_root))
        except Exception:
            bundle_value = str(selected)
        st.write(
            {
                "bundle": bundle_value,
                "manifest": index.manifest_path,
                "ontology": index.ontology_path,
            }
        )
    else:
        st.json({"manifest_path": index.manifest_path, "ontology_path": index.ontology_path, "hard_negatives_path": index.hard_negatives_path, "grouping_keys": list(index.grouping_keys)})
    if records:
        sample_record = records[0]
        preview = st.columns([1, 1.2])
        preview[0].image(load_uint8_grayscale(Path(sample_record.image_path)), clamp=True, caption=sample_record.sample_id)
        preview[1].write({"sample_id": sample_record.sample_id, "station_id": sample_record.station_id, "product_family": sample_record.product_family, "geometry_mode": str(sample_record.geometry_mode), "split": sample_record.split, "review_state": sample_record.review_state})
        if preview[1].button("Set as active dataset", key="gm_set_active_dataset"):
            workspace.active_dataset_index = str(selected)
            workspace.active_manifest = index.manifest_path
            _persist_workspace(workspace, workspace_path)
            st.success("Active dataset updated.")


def _render_models(st, workspace: WorkspaceConfig, data_root: Path, workspace_path: Path | None) -> None:
    st.header("Models")
    profiles = sorted(workspace.model_profiles.values(), key=lambda item: item.profile_id)
    if _simple_mode(workspace):
        st.caption("Pick the model profile that matches your goal. The advanced profile editor is hidden by default.")
    if profiles:
        if _simple_mode(workspace):
            st.dataframe(
                [
                    {
                        "profile": p.display_name or p.profile_id,
                        "profile_id": p.profile_id,
                        "use_for": _profile_summary(p)["purpose"],
                        "backend": p.backend_family,
                        "target_ms": p.latency_target_ms,
                    }
                    for p in profiles
                ],
                use_container_width=True,
            )
        else:
            st.dataframe([{"profile_id": p.profile_id, "name": p.display_name or p.profile_id, "backend_family": p.backend_family, "task_type": p.task_type, "model_id": p.model_id, "runtime_engine": p.runtime_engine, "latency_target_ms": p.latency_target_ms} for p in profiles], use_container_width=True)
    selected_id = st.selectbox("Selected profile", _profile_ids(workspace) or ["prod_fast_native"], index=_selectbox_index(_profile_ids(workspace) or ["prod_fast_native"], workspace.active_model_profile or "prod_fast_native"), key="gm_selected_profile")
    profile = workspace.model_profiles.get(selected_id, _active_profile(workspace))
    summary = _profile_summary(profile)
    st.write(
        {
            "profile": summary["title"],
            "purpose": summary["purpose"],
            "details": summary["details"],
            "latency_target_ms": float(profile.latency_target_ms),
            "runtime_engine": profile.runtime_engine,
        }
    )
    show_editor = not _simple_mode(workspace)
    if _simple_mode(workspace):
        show_editor = st.checkbox("Show advanced profile editor", value=False, key="gm_show_advanced_profile_editor")
    if show_editor:
        with st.form("model_profile_form"):
            profile_id = st.text_input("Profile ID", value=profile.profile_id)
            display_name = st.text_input("Display Name", value=profile.display_name)
            backend_family = st.selectbox("Backend Family", ["native", "hf_classification", "hf_detection", "hf_segmentation"], index=_selectbox_index(["native", "hf_classification", "hf_detection", "hf_segmentation"], profile.backend_family))
            task_type = st.selectbox("Task Type", ["train", "predict", "review", "benchmark", "calibrate"], index=_selectbox_index(["train", "predict", "review", "benchmark", "calibrate"], profile.task_type))
            native_variant_options = ["fast", "base", "lite"]
            native_variant = st.selectbox("Native Variant", native_variant_options, index=_selectbox_index(native_variant_options, profile.native_variant))
            model_id = st.text_input("Hugging Face Model ID or Alias", value=profile.model_id or "")
            local_path = st.text_input("Local Checkpoint / Snapshot Path", value=profile.local_path or "")
            model_revision = st.text_input("Model Revision", value=profile.model_revision or "")
            runtime_engine = st.selectbox("Runtime Engine", ["pytorch", "onnxruntime", "tensorrt"], index=_selectbox_index(["pytorch", "onnxruntime", "tensorrt"], profile.runtime_engine))
            cache_policy = st.selectbox("Cache Policy", ["online_and_cache", "offline_cache", "local_only"], index=_selectbox_index(["online_and_cache", "offline_cache", "local_only"], profile.cache_policy))
            latency_target_ms = st.number_input("Latency Target (ms)", min_value=0.1, value=float(profile.latency_target_ms), step=0.5)
            input_mode = st.selectbox("Input Mode", ["grayscale", "rgb_replicated"], index=_selectbox_index(["grayscale", "rgb_replicated"], profile.input_mode))
            output_mode = st.selectbox("Output Mode", ["hierarchical", "classification", "detection", "segmentation"], index=_selectbox_index(["hierarchical", "classification", "detection", "segmentation"], profile.output_mode))
            label_mapping_text = st.text_area("Label Mapping JSON", value=json.dumps(dict(profile.label_mapping), indent=2, sort_keys=True), height=120)
            family_mapping_text = st.text_area("Defect Family Mapping JSON", value=json.dumps(dict(profile.defect_family_mapping), indent=2, sort_keys=True), height=120)
            notes = st.text_area("Notes", value=profile.notes, height=100)
            submit = st.form_submit_button("Save Profile")
        if submit:
            try:
                label_mapping = json.loads(label_mapping_text or "{}")
                family_mapping = json.loads(family_mapping_text or "{}")
            except Exception as exc:
                st.error("Invalid JSON: %s" % exc)
            else:
                updated = ModelProfile(
                    profile_id=profile_id,
                    display_name=display_name,
                    backend_family=backend_family,
                    task_type=task_type,
                    model_id=model_id or None,
                    local_path=local_path or None,
                    model_revision=model_revision or None,
                    native_variant=native_variant,
                    runtime_engine=runtime_engine,
                    cache_policy=cache_policy,
                    latency_target_ms=float(latency_target_ms),
                    input_mode=input_mode,
                    output_mode=output_mode,
                    label_mapping=label_mapping,
                    defect_family_mapping=family_mapping,
                    notes=notes,
                    created_at=profile.created_at or utc_timestamp(),
                    updated_at=utc_timestamp(),
                )
                workspace = upsert_model_profile(workspace, updated)
                workspace.active_model_profile = updated.profile_id
                _persist_workspace(workspace, workspace_path)
                st.success("Profile saved.")
    buttons = st.columns(3)
    if buttons[0].button("Set Active", key="gm_profile_set_active"):
        workspace.active_model_profile = selected_id
        _persist_workspace(workspace, workspace_path)
        st.success("Active profile updated.")
    if buttons[1].button("Duplicate", key="gm_profile_duplicate") and selected_id in workspace.model_profiles:
        source = workspace.model_profiles[selected_id]
        duplicate = ModelProfile(profile_id="%s_copy" % source.profile_id, display_name="%s Copy" % (source.display_name or source.profile_id), backend_family=source.backend_family, task_type=source.task_type, model_id=source.model_id, local_path=source.local_path, model_revision=source.model_revision, native_variant=source.native_variant, runtime_engine=source.runtime_engine, cache_policy=source.cache_policy, latency_target_ms=source.latency_target_ms, input_mode=source.input_mode, output_mode=source.output_mode, label_mapping=source.label_mapping, defect_family_mapping=source.defect_family_mapping, notes=source.notes)
        workspace = upsert_model_profile(workspace, duplicate)
        _persist_workspace(workspace, workspace_path)
        st.success("Profile duplicated.")
    if buttons[2].button("Delete", key="gm_profile_delete") and selected_id in workspace.model_profiles:
        workspace = delete_model_profile(workspace, selected_id)
        _persist_workspace(workspace, workspace_path)
        st.warning("Profile deleted.")
    dataset_indexes = _known_dataset_indexes(workspace, data_root)
    if dataset_indexes:
        benchmark_choice = st.selectbox("Benchmark dataset", dataset_indexes, format_func=lambda path: str(path.relative_to(data_root)))
        if st.button("Run Latency Benchmark"):
            index = load_dataset_index(benchmark_choice)
            records = load_dataset_manifest(index.manifest_path)
            if records:
                station_configs = load_station_configs_from_index(index)
                station_config = station_config_for_record(records[0], station_configs)
                report = benchmark_profile_runtime(workspace.model_profiles[selected_id], records, station_config, cache_root=workspace.cache_root, local_files_only=workspace.model_profiles[selected_id].cache_policy != "online_and_cache", defect_families=tuple(read_json(Path(index.ontology_path)).get("defect_tags", ())), max_samples=8)
                report_path = save_benchmark_result(_review_root(Path(workspace.run_root)) / "latency_reports" / selected_id, report)
                st.success("Benchmark saved to %s" % report_path)
                st.json(report.__dict__)


def _render_train(st, repo_root: Path, run_root: Path, workspace: WorkspaceConfig, execution_defaults: UIExecutionDefaults, workspace_path: Path | None) -> None:
    st.header("Train")
    if _simple_mode(workspace):
        _render_autofit_launcher(
            st,
            repo_root=repo_root,
            run_root=run_root,
            workspace=workspace,
            execution_defaults=execution_defaults,
            title="Automated Training",
            description="Use the simplified training flow for labeled production data. Advanced stage-by-stage controls stay available when you switch the UI mode to `advanced`.",
        )
        if st.button("Save Workspace Defaults"):
            workspace.default_execution_backend = str(execution_defaults.execution_backend)
            workspace.slurm_cpus = int(execution_defaults.slurm_cpus)
            workspace.slurm_mem = str(execution_defaults.slurm_mem)
            workspace.slurm_gres = str(execution_defaults.slurm_gres)
            workspace.slurm_partition = str(execution_defaults.slurm_partition)
            workspace.slurm_queue = str(execution_defaults.slurm_queue)
            workspace.slurm_nproc_per_node = int(execution_defaults.slurm_nproc_per_node)
            _persist_workspace(workspace, workspace_path)
            st.success("Workspace defaults saved.")
        return
    profile = _active_profile(workspace)
    if not profile.is_native:
        st.info("Training jobs from the UI currently target the native GreyModel pipeline. Hugging Face profiles remain available for review and benchmark workflows.")
    stage_names = list(STAGE_LABELS.keys())
    stage = st.selectbox("Workflow step", stage_names, format_func=lambda value: STAGE_LABELS.get(value, value))
    st.caption(STAGE_HELP.get(stage, ""))
    manifest = st.text_input("Dataset Manifest", workspace.active_manifest or str(repo_root / "data" / "production" / "manifest.jsonl"))
    index = st.text_input("Dataset Index", workspace.active_dataset_index or str(repo_root / "data" / "production" / "dataset_index.json"))
    variant = st.selectbox("Variant", ["base", "lite"], index=_selectbox_index(["base", "lite"], profile.native_variant if profile.is_native else "base"))
    if profile.is_native and str(profile.native_variant).lower() == "fast":
        st.info("The production fast profile is optimized for inference and benchmark workflows. Training jobs still use the review backbones (`base` or `lite`).")
    epochs = st.number_input("Epochs", min_value=1, value=1)
    batch_size = st.number_input("Images Per Step", min_value=1, value=2)
    run_root_value = st.text_input("Run Root", workspace.run_root)
    execution_settings = _render_execution_settings(st, key_prefix="train", execution_defaults=execution_defaults, show_nproc=(stage != "calibrate"), nproc_default=execution_defaults.slurm_nproc_per_node if stage != "calibrate" else 1)
    task_tokens = ["train", stage, "--manifest", manifest, "--index", index, "--variant", variant, "--run-root", run_root_value]
    if stage != "calibrate":
        task_tokens.extend(["--epochs", str(int(epochs)), "--batch-size", str(int(batch_size))])
    metadata = {"profile_id": profile.profile_id, "profile_backend_family": profile.backend_family}
    inner_command, preview_command = _preview_job_commands(task_tokens, repo_root=repo_root, run_root=Path(run_root_value), kind="train", execution_backend=str(execution_settings["execution_backend"]), nproc_per_node=int(execution_settings["nproc_per_node"]), execution_defaults=execution_settings["execution_defaults"])
    if not _simple_mode(workspace):
        st.code(format_shell_command(preview_command))
    if st.button("Start Training Job"):
        payload = _launch_managed_job(task_tokens, cwd=repo_root, run_root=Path(run_root_value), kind="train", execution_backend=str(execution_settings["execution_backend"]), nproc_per_node=int(execution_settings["nproc_per_node"]), execution_defaults=execution_settings["execution_defaults"], metadata=metadata)
        st.success(("Submitted Slurm job %s" % payload.get("job_id")) if payload.get("backend") == "slurm" else ("Started PID %d" % int(payload["pid"])))
    _render_job_history(st, Path(run_root_value), "train")
    if st.button("Save Workspace Defaults"):
        workspace.default_execution_backend = str(execution_settings["execution_backend"])
        workspace.slurm_cpus = int(execution_settings["execution_defaults"].slurm_cpus)
        workspace.slurm_mem = str(execution_settings["execution_defaults"].slurm_mem)
        workspace.slurm_gres = str(execution_settings["execution_defaults"].slurm_gres)
        workspace.slurm_partition = str(execution_settings["execution_defaults"].slurm_partition)
        workspace.slurm_queue = str(execution_settings["execution_defaults"].slurm_queue)
        workspace.slurm_nproc_per_node = int(execution_settings["execution_defaults"].slurm_nproc_per_node)
        _persist_workspace(workspace, workspace_path)
        st.success("Workspace defaults saved.")


def _render_runs(st, run_root: Path) -> None:
    st.header("Runs")
    rows = list_run_statuses(run_root)
    if not rows:
        st.info("No run sessions found yet.")
        return
    stage_filter = st.selectbox("Stage filter", ["all"] + sorted({row.stage for row in rows}))
    status_filter = st.selectbox("Status filter", ["all"] + sorted({row.status for row in rows}))
    filtered = [row for row in rows if (stage_filter == "all" or row.stage == stage_filter) and (status_filter == "all" or row.status == status_filter)]
    st.dataframe([{"stage": row.stage, "variant": row.variant, "status": row.status, "updated_at": row.updated_at, "epoch": row.epoch, "global_step": row.global_step, "run_dir": row.run_dir, "checkpoint": row.latest_usable_checkpoint_path} for row in filtered[:40]], use_container_width=True)
    report_paths = sorted(run_root.rglob("*report.json"))
    if report_paths:
        left = st.selectbox("Left Report", report_paths, format_func=str)
        right = st.selectbox("Right Report", report_paths, index=min(1, len(report_paths) - 1), format_func=str)
        st.json(read_json(left))
        if left != right:
            st.json(compare_run_reports(left, right))


def _render_predict_review(st, repo_root: Path, run_root: Path, data_root: Path, workspace: WorkspaceConfig, workspace_path: Path | None) -> None:
    st.header("Predict & Review")
    if _simple_mode(workspace):
        st.caption("Run a quick review batch to see good/bad decisions and the strongest defect signal.")
    dataset_indexes = _known_dataset_indexes(workspace, data_root)
    if not dataset_indexes:
        st.info("No dataset bundles available.")
        return
    dataset_choice = st.selectbox("Dataset", dataset_indexes, format_func=lambda path: str(path.relative_to(data_root)))
    index = load_dataset_index(dataset_choice)
    records = load_dataset_manifest(index.manifest_path)
    station_configs = load_station_configs_from_index(index)
    if not records:
        st.info("Dataset is empty.")
        return
    profile_id = st.selectbox("Profile", _profile_ids(workspace), index=_selectbox_index(_profile_ids(workspace), workspace.active_model_profile or "prod_fast_native"))
    profile = workspace.model_profiles[profile_id]
    max_preview = int(st.number_input("Samples To Review", min_value=1, max_value=max(1, len(records)), value=min(8, len(records))))
    if st.button("Run Preview Batch"):
        defect_families = tuple(read_json(Path(index.ontology_path)).get("defect_tags", ())) if Path(index.ontology_path).exists() else ()
        predictions = []
        for record in records[:max_preview]:
            prediction = predict_record_with_profile(record, profile, station_config_for_record(record, station_configs), cache_root=workspace.cache_root, local_files_only=profile.cache_policy != "online_and_cache", defect_families=defect_families)
            predictions.append(prediction)
        review_dir = ensure_dir(_review_root(Path(workspace.run_root)) / ("%s-%s" % (profile.profile_id, utc_timestamp().replace(":", "").replace("T", "-"))))
        predictions_path = write_jsonl(review_dir / "predictions.jsonl", [asdict(prediction) for prediction in predictions])
        from greymodel.evaluation import evaluate_predictions

        report = evaluate_predictions(records[:max_preview], predictions)
        report_path = write_json(review_dir / "report.json", report)
        workspace = set_recent_run_dir(workspace, str(review_dir))
        _persist_workspace(workspace, workspace_path)
        st.success("Preview saved to %s" % review_dir)
        st.write({"predictions_path": str(predictions_path), "report_path": str(report_path)})
        st.dataframe(
            [
                {
                    "sample_id": prediction.sample_id,
                    "decision": prediction.primary_label,
                    "score": round(float(prediction.primary_score or prediction.reject_score), 4),
                    "top_defect": prediction.top_defect_family,
                }
                for prediction in predictions
            ],
            use_container_width=True,
        )
        st.json(report)
    st.markdown("### Sample Preview")
    sample = records[0]
    sample_cols = st.columns([1, 1.2])
    sample_cols[0].image(load_uint8_grayscale(Path(sample.image_path)), clamp=True, caption=sample.sample_id)
    sample_cols[1].write({"sample_id": sample.sample_id, "station_id": sample.station_id, "profile": profile.profile_id, "backend_family": profile.backend_family})


def _render_explain(st, run_root: Path, data_root: Path, workspace: WorkspaceConfig, workspace_path: Path | None) -> None:
    st.header("Explain")
    if _simple_mode(workspace):
        st.caption("Create a visual explanation bundle for one sample and inspect the saved images directly in the UI.")
    dataset_indexes = _known_dataset_indexes(workspace, data_root)
    if not dataset_indexes:
        st.info("No dataset bundles available.")
        return
    selected_index = st.selectbox("Dataset", dataset_indexes, format_func=lambda path: str(path.relative_to(data_root)))
    index = load_dataset_index(selected_index)
    records = load_dataset_manifest(index.manifest_path)
    station_configs = load_station_configs_from_index(index)
    if not records:
        st.info("Dataset is empty.")
        return
    profile_id = st.selectbox("Profile", _profile_ids(workspace), index=_selectbox_index(_profile_ids(workspace), workspace.active_model_profile or "prod_fast_native"))
    profile = workspace.model_profiles[profile_id]
    sample_id = st.selectbox("Sample", [record.sample_id for record in records], index=0)
    mode = st.selectbox("Mode", ["sample", "audit"])
    if st.button("Generate Explanation"):
        record = next((row for row in records if row.sample_id == sample_id), records[0])
        station_config = station_config_for_record(record, station_configs)
        output_dir_name = "%s-%s" % (profile.profile_id, record.sample_id.replace("/", "_"))
        output_dir = ensure_dir(Path(run_root) / "explanations" / output_dir_name)
        if profile.is_native:
            runtime = build_runtime_for_profile(profile, defect_families=tuple(read_json(Path(index.ontology_path)).get("defect_tags", ())) if Path(index.ontology_path).exists() else (), cache_root=workspace.cache_root, local_files_only=True)
            from greymodel.types import ModelInput

            bundle = build_explanation_bundle(runtime.model, ModelInput(image_uint8=load_uint8_grayscale(Path(record.image_path)), station_id=record.station_id, geometry_mode=record.geometry_mode, metadata=record.capture_metadata), station_config, output_dir)
        else:
            prediction = predict_record_with_profile(record, profile, station_config, cache_root=workspace.cache_root, local_files_only=profile.cache_policy != "online_and_cache", defect_families=())
            bundle = {
                "bundle_path": write_json(output_dir / "bundle.json", {"sample_id": record.sample_id, "profile_id": profile.profile_id, "prediction": asdict(prediction), "mode": mode}),
                "prediction_path": write_json(output_dir / "prediction.json", asdict(prediction)),
            }
        report_path = write_json(output_dir / "explain_report.json", {"mode": mode, "profile_id": profile.profile_id, "sample_id": record.sample_id, "bundle": {key: str(value) for key, value in bundle.items()}})
        st.success("Explanation saved to %s" % output_dir)
        st.json(read_json(report_path))
    bundle_paths = sorted(Path(run_root).rglob("bundle.json"))
    if bundle_paths:
        selected_bundle = st.selectbox("Existing Bundle", bundle_paths, format_func=str)
        bundle_payload = read_json(selected_bundle)
        st.json(bundle_payload)
        preview_paths = _bundle_preview_paths(bundle_payload)
        if preview_paths:
            st.image([str(path) for path in preview_paths], caption=[path.stem for path in preview_paths], clamp=True)


def _render_failures(st, run_root: Path) -> None:
    st.header("Failures")
    failures = list_failure_records(run_root)
    if not failures:
        st.info("No failure bundles found.")
        return
    selected = st.selectbox("Failure", failures, format_func=lambda row: "%s | %s | %s" % (row.timestamp, row.stage, row.error_type))
    st.write({"stage": selected.stage, "error": selected.error_message, "variant": selected.variant, "time": selected.timestamp})
    st.json(selected.__dict__)
    traceback_path = Path(selected.traceback_path)
    if traceback_path.exists():
        st.code(traceback_path.read_text(encoding="utf-8"))


def _render_settings(st, workspace: WorkspaceConfig, execution_defaults: UIExecutionDefaults, workspace_path: Path | None) -> tuple[WorkspaceConfig, UIExecutionDefaults]:
    st.header("Settings")
    with st.form("workspace_settings"):
        workspace.workspace_name = st.text_input("Workspace Name", workspace.workspace_name)
        workspace.run_root = st.text_input("Run Root", workspace.run_root)
        workspace.data_root = st.text_input("Data Root", workspace.data_root)
        workspace.cache_root = st.text_input("Cache Root", workspace.cache_root)
        workspace.ui_mode = st.selectbox("UI Mode", ["vision_engineer", "advanced"], index=_selectbox_index(["vision_engineer", "advanced"], workspace.ui_mode))
        workspace.proxy_mode = st.selectbox("Proxy Mode", ["auto", "off", "jupyter_port", "jupyter_service"], index=_selectbox_index(["auto", "off", "jupyter_port", "jupyter_service"], workspace.proxy_mode))
        defaults = _render_execution_settings(st, key_prefix="settings", execution_defaults=execution_defaults, show_nproc=True, nproc_default=execution_defaults.slurm_nproc_per_node)
        workspace.default_execution_backend = str(defaults["execution_backend"])
        workspace.slurm_cpus = int(defaults["execution_defaults"].slurm_cpus)
        workspace.slurm_mem = str(defaults["execution_defaults"].slurm_mem)
        workspace.slurm_gres = str(defaults["execution_defaults"].slurm_gres)
        workspace.slurm_partition = str(defaults["execution_defaults"].slurm_partition)
        workspace.slurm_queue = str(defaults["execution_defaults"].slurm_queue)
        workspace.slurm_nproc_per_node = int(defaults["execution_defaults"].slurm_nproc_per_node)
        workspace.ui_theme = st.selectbox("Theme", ["clean", "light", "compact"], index=_selectbox_index(["clean", "light", "compact"], workspace.ui_theme))
        workspace.notes = st.text_area("Notes", workspace.notes, height=100)
        submit = st.form_submit_button("Save Workspace")
    if submit:
        _persist_workspace(workspace, workspace_path)
        st.success("Workspace saved.")
    return workspace, defaults["execution_defaults"]


def render_app(
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    execution_defaults: UIExecutionDefaults | None = None,
    workspace_path: Path | str | None = None,
) -> None:
    import streamlit as st
    try:
        run_root = Path(run_root)
        data_root = Path(data_root)
        workspace_path = Path(workspace_path) if workspace_path is not None else workspace_path_for(run_root)
        workspace = load_workspace(run_root=run_root, data_root=data_root, workspace_path=workspace_path)
        execution_defaults = execution_defaults or _execution_defaults_from_workspace(workspace)
        st.set_page_config(page_title="GreyModel", layout="wide", initial_sidebar_state="expanded")
        _inject_theme(st)
        st.sidebar.title("GreyModel")
        st.sidebar.caption("Inspection workspace")
        st.sidebar.write({"workspace": workspace.workspace_name, "mode": workspace.ui_mode, "active_profile": workspace.active_model_profile, "run_root": workspace.run_root, "data_root": workspace.data_root})
        page = st.sidebar.radio("Page", PAGE_ORDER, index=0, key="gm_page")
        repo_root = Path(__file__).resolve().parents[2]
        if page == "Home":
            _render_home(st, repo_root, workspace, run_root, data_root, execution_defaults)
        elif page == "Datasets":
            _render_datasets(st, workspace, data_root, workspace_path)
        elif page == "Models":
            _render_models(st, workspace, data_root, workspace_path)
        elif page == "Train":
            _render_train(st, repo_root=repo_root, run_root=run_root, workspace=workspace, execution_defaults=execution_defaults, workspace_path=workspace_path)
        elif page == "Runs":
            _render_runs(st, run_root)
        elif page == "Predict & Review":
            _render_predict_review(st, repo_root=repo_root, run_root=run_root, data_root=data_root, workspace=workspace, workspace_path=workspace_path)
        elif page == "Explain":
            _render_explain(st, run_root=run_root, data_root=data_root, workspace=workspace, workspace_path=workspace_path)
        elif page == "Failures":
            _render_failures(st, run_root)
        else:
            workspace, execution_defaults = _render_settings(st, workspace, execution_defaults, workspace_path)
            _persist_workspace(workspace, workspace_path)
    except Exception as exc:
        try:
            st.set_page_config(page_title="GreyModel", layout="wide", initial_sidebar_state="expanded")
        except Exception:
            pass
        st.error("GreyModel UI failed during startup or page render.")
        st.exception(exc)


def main(argv: list[str] | None = None) -> None:
    args, _unknown = _parse_args(argv)
    render_app(
        run_root=args.run_root,
        data_root=args.data_root,
        execution_defaults=_execution_defaults_from_args(args),
        workspace_path=args.workspace_path,
    )


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        except Exception:
            return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" or _running_inside_streamlit():
    main(sys.argv[1:])
