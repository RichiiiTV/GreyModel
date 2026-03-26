from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from greymodel.data import load_dataset_index, load_dataset_manifest
from greymodel.registry import compare_run_reports, list_failure_records, list_run_statuses
from greymodel.ui import UIExecutionDefaults, build_greymodel_job_command, build_slurm_submission_command, format_shell_command
from greymodel.utils import ensure_dir, load_uint8_grayscale, read_json, utc_timestamp, write_json


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", default="artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--default-execution-backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument("--slurm-mem", default="50G")
    parser.add_argument("--slurm-gres", default="gpu:8")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-queue", default="")
    parser.add_argument("--slurm-nproc-per-node", type=int, default=8)
    parser.add_argument("--slurm-python", default=sys.executable)
    return parser.parse_known_args(argv)


def _find_dataset_indexes(data_root: Path) -> list[Path]:
    return sorted(data_root.rglob("dataset_index.json"))


def _job_root(run_root: Path) -> Path:
    return ensure_dir(run_root / "ui_jobs")


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


def _create_ui_job_paths(run_root: Path, kind: str) -> tuple[Path, Path]:
    job_dir = _job_root(run_root)
    stamp = utc_timestamp().replace(":", "").replace("-", "").replace("T", "-").replace("Z", "")
    log_path = job_dir / ("%s-%s.log" % (kind, stamp))
    metadata_path = job_dir / ("%s-%s.json" % (kind, stamp))
    return log_path, metadata_path


def _launch_local_job(command: list[str], cwd: Path, run_root: Path, kind: str) -> dict[str, object]:
    log_path, metadata_path = _create_ui_job_paths(run_root, kind)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=handle, stderr=handle)
    payload = {
        "backend": "local",
        "pid": int(process.pid),
        "kind": kind,
        "command": command,
        "log_path": str(log_path),
        "metadata_path": str(metadata_path),
        "created_at": utc_timestamp(),
    }
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
        raise RuntimeError(
            "Slurm submission failed with exit code %d: %s"
            % (int(result.returncode), (result.stderr or result.stdout).strip())
        )
    stdout = (result.stdout or "").strip()
    payload = {
        "backend": "slurm",
        "job_id": stdout.split(";", 1)[0] if stdout else "",
        "kind": kind,
        "command": command,
        "submit_command": submit_command,
        "log_path": str(log_path),
        "metadata_path": str(metadata_path),
        "created_at": utc_timestamp(),
        "submit_stdout": stdout,
        "submit_stderr": (result.stderr or "").strip(),
    }
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
) -> dict[str, object]:
    python_executable = execution_defaults.slurm_python if execution_backend == "slurm" else sys.executable
    command = build_greymodel_job_command(
        task_tokens,
        python_executable=python_executable,
        nproc_per_node=max(int(nproc_per_node), 1),
    )
    if execution_backend == "slurm":
        return _launch_slurm_job(
            command,
            cwd,
            run_root,
            kind,
            cpus=execution_defaults.slurm_cpus,
            mem=execution_defaults.slurm_mem,
            gres=execution_defaults.slurm_gres,
            partition=execution_defaults.slurm_partition,
            queue=execution_defaults.slurm_queue,
        )
    return _launch_local_job(command, cwd, run_root, kind)


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
        nproc_per_node = int(
            st.number_input(
                "Processes / GPUs",
                min_value=1,
                value=max(int(nproc_default), 1),
                key="%s_nproc" % key_prefix,
            )
        )
    if execution_backend != "slurm":
        return {
            "execution_backend": execution_backend,
            "nproc_per_node": int(nproc_per_node),
            "execution_defaults": execution_defaults,
        }
    slurm_defaults = UIExecutionDefaults(
        execution_backend="slurm",
        slurm_cpus=int(
            st.number_input(
                "Slurm CPUs (-c)",
                min_value=1,
                value=max(int(execution_defaults.slurm_cpus), 1),
                key="%s_slurm_cpus" % key_prefix,
            )
        ),
        slurm_mem=st.text_input("Slurm Memory (--mem)", execution_defaults.slurm_mem, key="%s_slurm_mem" % key_prefix),
        slurm_gres=st.text_input("Slurm GRES (--gres)", execution_defaults.slurm_gres, key="%s_slurm_gres" % key_prefix),
        slurm_partition=st.text_input(
            "Slurm Partition (-p)",
            execution_defaults.slurm_partition,
            key="%s_slurm_partition" % key_prefix,
        ),
        slurm_queue=st.text_input("Slurm Queue (-q)", execution_defaults.slurm_queue, key="%s_slurm_queue" % key_prefix),
        slurm_nproc_per_node=max(int(nproc_per_node), 1),
        slurm_python=st.text_input(
            "Slurm Python",
            execution_defaults.slurm_python,
            key="%s_slurm_python" % key_prefix,
        ),
    )
    return {
        "execution_backend": execution_backend,
        "nproc_per_node": int(nproc_per_node) if show_nproc else int(slurm_defaults.slurm_nproc_per_node),
        "execution_defaults": slurm_defaults,
    }


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
    inner_command = build_greymodel_job_command(
        task_tokens,
        python_executable=python_executable,
        nproc_per_node=max(int(nproc_per_node), 1),
    )
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


def collect_ui_state(run_root: Path | str = "artifacts", data_root: Path | str = "data") -> dict[str, object]:
    run_root = Path(run_root)
    data_root = Path(data_root)
    return {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "runs": [row.__dict__ for row in list_run_statuses(run_root)],
        "failures": [row.__dict__ for row in list_failure_records(run_root)],
        "datasets": [str(path) for path in _find_dataset_indexes(data_root)] if data_root.exists() else [],
        "jobs": _ui_jobs(run_root),
    }


def _render_overview(st, run_root: Path) -> None:
    st.header("Overview")
    run_rows = list_run_statuses(run_root)
    if not run_rows:
        st.info("No run sessions found yet.")
        return
    st.subheader("Recent Runs")
    st.dataframe(
        [
            {
                "stage": row.stage,
                "variant": row.variant,
                "status": row.status,
                "updated_at": row.updated_at,
                "epoch": row.epoch,
                "global_step": row.global_step,
                "run_dir": row.run_dir,
                "checkpoint": row.latest_usable_checkpoint_path,
            }
            for row in run_rows[:20]
        ],
        use_container_width=True,
    )
    failures = list_failure_records(run_root)
    st.subheader("Recent Failures")
    st.dataframe(
        [
            {
                "stage": row.stage,
                "variant": row.variant,
                "timestamp": row.timestamp,
                "error_type": row.error_type,
                "sample_ids": ", ".join(row.offending_sample_ids[:3]),
                "failure_dir": row.failure_dir,
            }
            for row in failures[:10]
        ],
        use_container_width=True,
    )


def _render_datasets(st, data_root: Path) -> None:
    st.header("Datasets")
    dataset_indexes = _find_dataset_indexes(data_root)
    if not dataset_indexes:
        st.info("No dataset bundles found under %s." % data_root)
        return
    selected = st.selectbox("Dataset Index", dataset_indexes, format_func=lambda path: str(path.relative_to(data_root)))
    index = load_dataset_index(selected)
    records = load_dataset_manifest(index.manifest_path)
    st.json(
        {
            "root_dir": index.root_dir,
            "manifest_path": index.manifest_path,
            "ontology_path": index.ontology_path,
            "num_records": len(records),
            "num_station_configs": len(index.station_configs),
            "grouping_keys": list(index.grouping_keys),
            "metadata": dict(index.metadata),
        }
    )
    if records:
        sample_record = records[0]
        st.subheader("Sample Preview")
        st.write({"sample_id": sample_record.sample_id, "station_id": sample_record.station_id, "split": sample_record.split})
        image = load_uint8_grayscale(Path(sample_record.image_path))
        st.image(image, clamp=True)


def _render_train(st, repo_root: Path, run_root: Path, execution_defaults: UIExecutionDefaults) -> None:
    st.header("Train")
    stage = st.selectbox("Stage", ["pretrain", "domain-adapt", "finetune", "calibrate"])
    manifest = st.text_input("Manifest", str(repo_root / "data" / "production" / "manifest.jsonl"))
    index = st.text_input("Index", str(repo_root / "data" / "production" / "dataset_index.json"))
    variant = st.selectbox("Variant", ["base", "lite"])
    epochs = st.number_input("Epochs", min_value=1, value=1)
    batch_size = st.number_input("Batch Size", min_value=1, value=2)
    run_root_value = st.text_input("Run Root", str(run_root))
    execution_settings = _render_execution_settings(
        st,
        key_prefix="train",
        execution_defaults=execution_defaults,
        show_nproc=(stage != "calibrate"),
        nproc_default=(
            execution_defaults.slurm_nproc_per_node
            if stage != "calibrate" and execution_defaults.execution_backend == "slurm"
            else 1
        ),
    )
    task_tokens = [
        "train",
        stage,
        "--manifest",
        manifest,
        "--index",
        index,
        "--variant",
        variant,
        "--run-root",
        run_root_value,
    ]
    if stage != "calibrate":
        task_tokens.extend(["--epochs", str(int(epochs)), "--batch-size", str(int(batch_size))])
    inner_command, preview_command = _preview_job_commands(
        task_tokens,
        repo_root=repo_root,
        run_root=Path(run_root_value),
        kind="train",
        execution_backend=str(execution_settings["execution_backend"]),
        nproc_per_node=int(execution_settings["nproc_per_node"]),
        execution_defaults=execution_settings["execution_defaults"],
    )
    st.code(format_shell_command(preview_command))
    if str(execution_settings["execution_backend"]) == "slurm":
        st.code(format_shell_command(inner_command))
    if st.button("Launch Job"):
        payload = _launch_managed_job(
            task_tokens,
            cwd=repo_root,
            run_root=Path(run_root_value),
            kind="train",
            execution_backend=str(execution_settings["execution_backend"]),
            nproc_per_node=int(execution_settings["nproc_per_node"]),
            execution_defaults=execution_settings["execution_defaults"],
        )
        if payload.get("backend") == "slurm":
            st.success("Submitted Slurm job %s" % str(payload.get("job_id") or "unknown"))
        else:
            st.success("Started PID %d" % int(payload["pid"]))
    _render_job_history(st, Path(run_root_value), "train")


def _report_choices(run_root: Path) -> list[Path]:
    return sorted(run_root.rglob("*report.json"))


def _render_evaluate(st, run_root: Path) -> None:
    st.header("Evaluate")
    report_paths = _report_choices(run_root)
    if not report_paths:
        st.info("No reports found under %s." % run_root)
        return
    left = st.selectbox("Left Report", report_paths, format_func=str)
    right = st.selectbox("Right Report", report_paths, index=min(1, len(report_paths) - 1), format_func=str)
    st.subheader("Selected Report")
    st.json(read_json(left))
    if left != right:
        st.subheader("Comparison")
        st.json(compare_run_reports(left, right))


def _render_predict(st, repo_root: Path, run_root: Path, execution_defaults: UIExecutionDefaults) -> None:
    st.header("Predict")
    manifest = st.text_input("Manifest", str(repo_root / "data" / "production" / "manifest.jsonl"), key="predict_manifest")
    index = st.text_input("Index", str(repo_root / "data" / "production" / "dataset_index.json"), key="predict_index")
    variant = st.selectbox("Variant", ["base", "lite"], key="predict_variant")
    evidence_policy = st.selectbox("Evidence Policy", ["bad", "all", "none"], key="predict_evidence")
    run_root_value = st.text_input("Run Root", str(run_root), key="predict_run_root")
    execution_settings = _render_execution_settings(
        st,
        key_prefix="predict",
        execution_defaults=execution_defaults,
        show_nproc=False,
    )
    task_tokens = [
        "predict",
        "--manifest",
        manifest,
        "--index",
        index,
        "--variant",
        variant,
        "--run-root",
        run_root_value,
        "--evidence-policy",
        evidence_policy,
    ]
    inner_command, preview_command = _preview_job_commands(
        task_tokens,
        repo_root=repo_root,
        run_root=Path(run_root_value),
        kind="predict",
        execution_backend=str(execution_settings["execution_backend"]),
        nproc_per_node=1,
        execution_defaults=execution_settings["execution_defaults"],
    )
    st.code(format_shell_command(preview_command))
    if str(execution_settings["execution_backend"]) == "slurm":
        st.code(format_shell_command(inner_command))
    if st.button("Launch Prediction"):
        payload = _launch_managed_job(
            task_tokens,
            cwd=repo_root,
            run_root=Path(run_root_value),
            kind="predict",
            execution_backend=str(execution_settings["execution_backend"]),
            nproc_per_node=1,
            execution_defaults=execution_settings["execution_defaults"],
        )
        if payload.get("backend") == "slurm":
            st.success("Submitted Slurm job %s" % str(payload.get("job_id") or "unknown"))
        else:
            st.success("Started PID %d" % int(payload["pid"]))
    _render_job_history(st, Path(run_root_value), "predict")


def _render_explain(st, repo_root: Path, run_root: Path, execution_defaults: UIExecutionDefaults) -> None:
    st.header("Explain")
    manifest = st.text_input("Manifest", str(repo_root / "data" / "production" / "manifest.jsonl"), key="explain_manifest")
    explain_mode = st.selectbox("Mode", ["sample", "audit"], key="explain_mode")
    sample_id = st.text_input("Sample ID", "", key="explain_sample_id")
    limit = st.number_input("Audit Limit", min_value=1, value=5, key="explain_limit")
    variant = st.selectbox("Variant", ["base", "lite"], key="explain_variant")
    run_root_value = st.text_input("Output Dir", str(run_root), key="explain_run_root")
    execution_settings = _render_execution_settings(
        st,
        key_prefix="explain",
        execution_defaults=execution_defaults,
        show_nproc=False,
    )
    task_tokens = [
        "explain",
        explain_mode,
        "--manifest",
        manifest,
        "--variant",
        variant,
        "--output-dir",
        run_root_value,
    ]
    if explain_mode == "sample":
        if sample_id.strip():
            task_tokens.extend(["--sample-id", sample_id.strip()])
    else:
        task_tokens.extend(["--limit", str(int(limit))])
    inner_command, preview_command = _preview_job_commands(
        task_tokens,
        repo_root=repo_root,
        run_root=Path(run_root_value),
        kind="explain",
        execution_backend=str(execution_settings["execution_backend"]),
        nproc_per_node=1,
        execution_defaults=execution_settings["execution_defaults"],
    )
    st.code(format_shell_command(preview_command))
    if str(execution_settings["execution_backend"]) == "slurm":
        st.code(format_shell_command(inner_command))
    if st.button("Generate Explanation"):
        payload = _launch_managed_job(
            task_tokens,
            cwd=repo_root,
            run_root=Path(run_root_value),
            kind="explain",
            execution_backend=str(execution_settings["execution_backend"]),
            nproc_per_node=1,
            execution_defaults=execution_settings["execution_defaults"],
        )
        if payload.get("backend") == "slurm":
            st.success("Submitted Slurm job %s" % str(payload.get("job_id") or "unknown"))
        else:
            st.success("Started PID %d" % int(payload["pid"]))
    _render_job_history(st, Path(run_root_value), "explain")
    bundle_paths = sorted(Path(run_root_value).rglob("bundle.json"))
    if bundle_paths:
        selected_bundle = st.selectbox("Existing Bundle", bundle_paths, format_func=str)
        bundle = read_json(selected_bundle)
        st.json(bundle)


def _render_failures(st, run_root: Path) -> None:
    st.header("Failures")
    failures = list_failure_records(run_root)
    if not failures:
        st.info("No failure bundles found.")
        return
    selected = st.selectbox("Failure", failures, format_func=lambda row: "%s | %s | %s" % (row.timestamp, row.stage, row.error_type))
    st.json(selected.__dict__)
    traceback_path = Path(selected.traceback_path)
    if traceback_path.exists():
        st.code(traceback_path.read_text(encoding="utf-8"))


def render_app(
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    execution_defaults: UIExecutionDefaults | None = None,
) -> None:
    import streamlit as st

    run_root = Path(run_root)
    data_root = Path(data_root)
    repo_root = Path(__file__).resolve().parents[2]
    execution_defaults = execution_defaults or UIExecutionDefaults()
    st.set_page_config(page_title="GreyModel", layout="wide")
    st.sidebar.title("GreyModel UI")
    page = st.sidebar.radio("Page", ["Overview", "Datasets", "Train", "Predict", "Evaluate", "Explain", "Failures"])
    st.sidebar.write({"run_root": str(run_root), "data_root": str(data_root)})
    if page == "Overview":
        _render_overview(st, run_root)
    elif page == "Datasets":
        _render_datasets(st, data_root)
    elif page == "Train":
        _render_train(st, repo_root, run_root, execution_defaults)
    elif page == "Predict":
        _render_predict(st, repo_root, run_root, execution_defaults)
    elif page == "Evaluate":
        _render_evaluate(st, run_root)
    elif page == "Explain":
        _render_explain(st, repo_root, run_root, execution_defaults)
    else:
        _render_failures(st, run_root)


def main(argv: list[str] | None = None) -> None:
    args, _unknown = _parse_args(argv)
    render_app(
        run_root=args.run_root,
        data_root=args.data_root,
        execution_defaults=_execution_defaults_from_args(args),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
