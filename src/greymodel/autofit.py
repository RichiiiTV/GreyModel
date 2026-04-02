from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

from .data import (
    build_dataset_manifest,
    build_dataset_splits,
    load_dataset_index,
    load_dataset_manifest,
    save_dataset_index,
    save_dataset_manifest,
    validate_dataset_manifest,
)
from .evaluation import benchmark_manifest
from .model_profiles import ModelProfile, load_model_profile
from .recovery import write_failure_bundle
from .runners import run_calibration_stage, run_finetune_stage
from .settings import ensure_settings
from .tracking import create_run_context, snapshot_run_config, update_run_status, write_summary
from .types import DatasetIndex, DatasetRecord
from .ui import UIExecutionDefaults, build_greymodel_job_command, build_slurm_submission_command
from .utils import ensure_dir, stable_int_hash, utc_timestamp, write_json
from .version import __version__


@dataclass(frozen=True)
class AutoFitPlan:
    status: str
    execution_backend: str
    run_root: str
    run_dir: str
    stage_root: str
    data_path: str
    data_kind: str
    prepared_bundle_dir: Optional[str]
    manifest_path: str
    index_path: str
    split_policy: str
    variant: str
    model_reference: str
    resolved_profile: Mapping[str, Any]
    warm_start: Optional[str]
    stages: Sequence[str]
    validation_report: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AutoFitResult:
    status: str
    execution_backend: str
    run_root: str
    run_dir: str
    log_path: Optional[str] = None
    summary_path: Optional[str] = None
    markdown_summary_path: Optional[str] = None
    report_path: Optional[str] = None
    calibration_report_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    latest_usable_checkpoint_path: Optional[str] = None
    job_id: Optional[str] = None
    metrics: Mapping[str, Any] = None
    resolved_plan: Mapping[str, Any] = None
    resolved_profile: Mapping[str, Any] = None
    submitted_command: Sequence[str] = ()
    submission_command: Sequence[str] = ()


class _AutoFitLogger:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = path
        self.echo = bool(echo)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = "[%s] %s" % (utc_timestamp(), str(message))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        if self.echo:
            print(line)


def _profile_alias(profile_reference: str | Path | ModelProfile | Mapping[str, Any] | None):
    if profile_reference in (None, ""):
        return "prod_fast_native"
    if isinstance(profile_reference, (Path, ModelProfile, Mapping)):
        return profile_reference
    aliases = {
        "fast": "prod_fast_native",
        "prod_fast": "prod_fast_native",
        "base": "review_native_base",
        "lite": "review_native_lite",
    }
    normalized = str(profile_reference).strip().lower()
    return aliases.get(normalized, profile_reference)


def _prepare_training_profile(
    model: str | Path | ModelProfile | Mapping[str, Any] | None,
    *,
    registry_root: Path | str,
) -> tuple[ModelProfile, str, list[str]]:
    messages: list[str] = []
    profile = load_model_profile(_profile_alias(model), registry_root=registry_root)
    if profile.is_native:
        variant = str(profile.native_variant).lower()
        if variant == "fast":
            messages.append(
                "Profile %s is an inference-optimized fast path. AutoFit will train the `base` review backbone instead."
                % profile.profile_id
            )
            resolved = load_model_profile("review_native_base", registry_root=registry_root)
            return resolved, "base", messages
        if variant not in {"base", "lite"}:
            raise ValueError("Unsupported native training variant %r for AutoFit." % variant)
        return profile, variant, messages
    if profile.backend_family == "huggingface" and profile.task_type == "classification":
        raise NotImplementedError(
            "AutoFit training currently targets native GreyModel profiles only. Hugging Face classification profiles remain available for predict/eval/review."
        )
    raise ValueError(
        "AutoFit training supports native GreyModel profiles only. Profile %s uses backend_family=%s, task_type=%s."
        % (profile.profile_id, profile.backend_family, profile.task_type)
    )


def _dataset_bundle_dir(run_root: Path, source_path: Path) -> Path:
    slug = source_path.stem if source_path.suffix else source_path.name
    slug = slug or "dataset"
    suffix = stable_int_hash(str(source_path.resolve()))
    return run_root / "autofit_datasets" / ("%s-%08x" % (slug, suffix))


def _manifest_has_train_and_val(records: Sequence[DatasetRecord]) -> bool:
    splits = {str(record.split).strip().lower() for record in records}
    return "train" in splits and "val" in splits


def _force_minimum_train_val_split(records: Sequence[DatasetRecord]) -> list[DatasetRecord]:
    if len(records) < 2:
        raise ValueError("AutoFit requires at least two labeled samples to create train/val splits.")
    rewritten: list[DatasetRecord] = []
    val_index = 0
    for index, record in enumerate(records):
        split = "val" if index == val_index else "train"
        rewritten.append(
            DatasetRecord(
                sample_id=record.sample_id,
                image_path=record.image_path,
                station_id=record.station_id,
                product_family=record.product_family,
                geometry_mode=record.geometry_mode,
                accept_reject=record.accept_reject,
                defect_tags=record.defect_tags,
                boxes=record.boxes,
                mask_path=record.mask_path,
                split=split,
                capture_metadata=record.capture_metadata,
                source_dataset=record.source_dataset,
                review_state=record.review_state,
            )
        )
    return rewritten


def _update_index_split_assignments(index_path: Path, manifest_path: Path) -> Path:
    index = load_dataset_index(index_path)
    records = load_dataset_manifest(manifest_path)
    updated = DatasetIndex(
        manifest_version=index.manifest_version,
        ontology_version=index.ontology_version,
        root_dir=index.root_dir,
        manifest_path=index.manifest_path,
        splits_path=index.splits_path,
        ontology_path=index.ontology_path,
        hard_negatives_path=index.hard_negatives_path,
        index_path=index.index_path or str(index_path.resolve()),
        split_seed=index.split_seed,
        grouping_keys=index.grouping_keys,
        split_assignments={record.sample_id: record.split for record in records},
        hard_negative_ids=index.hard_negative_ids,
        review_subset_ids=index.review_subset_ids,
        station_configs=index.station_configs,
        metadata=index.metadata,
    )
    save_dataset_index(updated, index_path)
    return index_path


def _ensure_train_val_split(
    manifest_path: Path,
    *,
    index_path: Path,
    split_policy: str,
) -> list[DatasetRecord]:
    records = load_dataset_manifest(manifest_path)
    mode = str(split_policy).lower()
    if mode == "rebuild" or (mode == "auto" and not _manifest_has_train_and_val(records)):
        build_dataset_splits(manifest_path, output_path=manifest_path.with_name("splits.json"))
        records = load_dataset_manifest(manifest_path)
    if not _manifest_has_train_and_val(records):
        records = _force_minimum_train_val_split(records)
        save_dataset_manifest(records, manifest_path)
    _update_index_split_assignments(index_path, manifest_path)
    return records


def _build_split_filtered_manifest(
    manifest_path: Path,
    split: str,
    *,
    output_dir: Path,
) -> Path:
    records = [record for record in load_dataset_manifest(manifest_path) if str(record.split).lower() == str(split).lower()]
    if not records:
        raise ValueError("AutoFit requires at least one `%s` sample in %s." % (split, manifest_path))
    output_path = output_dir / ("%s_manifest.jsonl" % str(split).lower())
    save_dataset_manifest(records, output_path)
    return output_path


def _resolved_execution_backend(
    execution: str,
    *,
    workspace_defaults: UIExecutionDefaults | None = None,
) -> str:
    normalized = str(execution or "auto").strip().lower()
    if normalized == "auto":
        if workspace_defaults is not None and str(workspace_defaults.execution_backend).lower() == "slurm":
            return "slurm"
        return "local"
    if normalized not in {"local", "slurm"}:
        raise ValueError("Unsupported AutoFit execution backend %r." % execution)
    return normalized


def _autofit_summary_markdown(
    *,
    result: AutoFitResult,
    summary_payload: Mapping[str, Any],
) -> str:
    metrics = dict(result.metrics or {})
    overall = dict(metrics.get("overall", {}))
    defect_family = dict(metrics.get("defect_family_bad_only", {}))
    lines = [
        "# AutoFit Summary",
        "",
        "Status: `%s`" % result.status,
        "Execution: `%s`" % result.execution_backend,
        "Run dir: `%s`" % result.run_dir,
        "",
        "## Outputs",
        "",
        "- Best checkpoint: `%s`" % (result.best_checkpoint_path or "n/a"),
        "- Latest checkpoint: `%s`" % (result.latest_checkpoint_path or "n/a"),
        "- Calibration report: `%s`" % (result.calibration_report_path or "n/a"),
        "- Benchmark report: `%s`" % (result.report_path or "n/a"),
        "",
        "## Metrics",
        "",
        "- Accuracy: `%s`" % overall.get("accuracy"),
        "- AUROC: `%s`" % overall.get("auroc"),
        "- FAR: `%s`" % overall.get("far"),
        "- FRR: `%s`" % overall.get("frr"),
        "- Calibrated threshold: `%s`" % summary_payload.get("recommended_threshold"),
        "- Bad-only defect-family top1: `%s`" % defect_family.get("top1_accuracy"),
        "",
    ]
    return "\n".join(lines)


def _write_autofit_summary(
    run_dir: Path,
    payload: Mapping[str, Any],
    *,
    result: AutoFitResult,
) -> tuple[Path, Path]:
    summary_json_path = write_json(run_dir / "reports" / "autofit_summary.json", payload)
    markdown_path = run_dir / "reports" / "autofit_summary.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_autofit_summary_markdown(result=result, summary_payload=payload), encoding="utf-8")
    return summary_json_path, markdown_path


def _build_benchmark_profile(profile: ModelProfile, checkpoint_path: Path) -> ModelProfile:
    return replace(profile, local_path=str(checkpoint_path))


def _materialize_runtime_profile(profile: ModelProfile, output_path: Path) -> Path:
    return write_json(output_path, profile.to_dict())


def _resolve_data_bundle(
    data: Path | str,
    *,
    run_dir: Path,
    split_policy: str,
    source_dataset: str = "autofit_folder",
) -> tuple[Path, Path, Optional[Path], Mapping[str, Any], Mapping[str, Any]]:
    data_path = Path(data)
    if data_path.is_file():
        manifest_path = data_path
        index_path = data_path.with_name("dataset_index.json")
        if not index_path.exists():
            raise FileNotFoundError("No dataset_index.json found next to %s." % manifest_path)
        validation = validate_dataset_manifest(manifest_path)
        if validation["num_errors"] > 0:
            raise ValueError("Dataset validation failed: %s" % validation["errors"][0])
        records = _ensure_train_val_split(manifest_path, index_path=index_path, split_policy=split_policy)
        if not _manifest_has_train_and_val(records):
            raise ValueError("AutoFit requires both `train` and `val` samples in the manifest.")
        return manifest_path, index_path, None, validation, {
            "data_kind": "manifest",
            "bundle_dir": None,
            "num_records": len(records),
        }

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError("AutoFit data path %s does not exist." % data_path)

    existing_manifest = data_path / "manifest.jsonl"
    existing_index = data_path / "dataset_index.json"
    if existing_manifest.exists() and existing_index.exists():
        return _resolve_data_bundle(existing_manifest, run_dir=run_dir, split_policy=split_policy, source_dataset=source_dataset)

    bundle_dir = _dataset_bundle_dir(run_dir, data_path)
    index = build_dataset_manifest(data_path, output_dir=bundle_dir, source_dataset=source_dataset)
    manifest_path = Path(index.manifest_path)
    index_path = Path(index.index_path or (manifest_path.with_name("dataset_index.json")))
    validation = validate_dataset_manifest(manifest_path)
    if validation["num_errors"] > 0:
        raise ValueError("Dataset validation failed: %s" % validation["errors"][0])
    _ensure_train_val_split(manifest_path, index_path=index_path, split_policy=split_policy)
    return manifest_path, index_path, bundle_dir, validation, {
        "data_kind": "folder",
        "bundle_dir": str(bundle_dir),
        "num_records": int(validation["num_records"]),
    }


def resolve_autofit_plan(
    *,
    data: Path | str,
    model: str | Path | ModelProfile | Mapping[str, Any] | None = None,
    run_root: Path | str | None = None,
    split_policy: str = "auto",
    warm_start: Path | str | None = None,
    execution: str = "auto",
    registry_root: Path | str | None = None,
    workspace_defaults: UIExecutionDefaults | None = None,
) -> AutoFitPlan:
    settings = ensure_settings()
    resolved_run_root = Path(run_root) if run_root is not None else Path(settings.run_root)
    resolved_registry_root = Path(registry_root) if registry_root is not None else Path(settings.registry_root)
    profile, variant, messages = _prepare_training_profile(model, registry_root=resolved_registry_root)
    context = create_run_context(resolved_run_root, stage="autofit", variant=variant)
    manifest_path, index_path, bundle_dir, validation, data_metadata = _resolve_data_bundle(
        data,
        run_dir=context.run_dir,
        split_policy=split_policy,
    )
    execution_backend = _resolved_execution_backend(execution, workspace_defaults=workspace_defaults)
    metadata = {
        **dict(data_metadata),
        "messages": list(messages),
        "model_version": __version__,
    }
    return AutoFitPlan(
        status="planned",
        execution_backend=execution_backend,
        run_root=str(resolved_run_root),
        run_dir=str(context.run_dir),
        stage_root=str(context.run_dir / "stages"),
        data_path=str(Path(data)),
        data_kind=str(data_metadata["data_kind"]),
        prepared_bundle_dir=str(bundle_dir) if bundle_dir is not None else None,
        manifest_path=str(manifest_path),
        index_path=str(index_path),
        split_policy=str(split_policy),
        variant=variant,
        model_reference=profile.profile_id,
        resolved_profile=profile.to_dict(),
        warm_start=str(warm_start) if warm_start is not None else None,
        stages=("finetune", "calibrate", "benchmark"),
        validation_report=dict(validation),
        metadata=metadata,
    )


def _training_config_from_overrides(
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    model_profile: str,
    model_registry_root: str,
) -> "TrainingConfig":
    from .training import TrainingConfig

    payload = dict(overrides or {})
    if "batch_size" in payload and "per_device_batch_size" not in payload:
        payload["per_device_batch_size"] = int(payload.pop("batch_size"))
    payload.setdefault("model_profile", model_profile)
    payload.setdefault("model_registry_root", model_registry_root)
    return TrainingConfig(**payload)


def _autofit_task_tokens(
    *,
    data: Path | str,
    model: str,
    run_root: Path | str,
    split_policy: str,
    warm_start: Path | str | None,
    overrides: Mapping[str, Any] | None,
) -> list[str]:
    tokens = [
        "auto",
        "fit",
        "--data",
        str(data),
        "--model",
        str(model),
        "--run-root",
        str(run_root),
        "--execution",
        "local",
        "--split-policy",
        str(split_policy),
    ]
    if warm_start is not None:
        tokens.extend(["--warm-start", str(warm_start)])
    payload = dict(overrides or {})
    if "epochs" in payload:
        tokens.extend(["--epochs", str(int(payload["epochs"]))])
    if "batch_size" in payload:
        tokens.extend(["--batch-size", str(int(payload["batch_size"]))])
    if "learning_rate" in payload:
        tokens.extend(["--learning-rate", str(float(payload["learning_rate"]))])
    if "num_workers" in payload:
        tokens.extend(["--num-workers", str(int(payload["num_workers"]))])
    if "precision" in payload:
        tokens.extend(["--precision", str(payload["precision"])])
    return tokens


def submit_autofit_job(
    *,
    data: Path | str,
    model: str,
    run_root: Path | str,
    split_policy: str,
    warm_start: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    execution_defaults: UIExecutionDefaults,
    repo_root: Path | str,
) -> AutoFitResult:
    log_dir = ensure_dir(Path(run_root) / "ui_jobs")
    task_tokens = _autofit_task_tokens(
        data=data,
        model=model,
        run_root=run_root,
        split_policy=split_policy,
        warm_start=warm_start,
        overrides=overrides,
    )
    inner_command = build_greymodel_job_command(
        task_tokens,
        python_executable=execution_defaults.slurm_python,
        nproc_per_node=max(int(execution_defaults.slurm_nproc_per_node), 1),
    )
    submit_command = build_slurm_submission_command(
        inner_command=inner_command,
        repo_root=repo_root,
        cpus=execution_defaults.slurm_cpus,
        mem=execution_defaults.slurm_mem,
        gres=execution_defaults.slurm_gres,
        partition=execution_defaults.slurm_partition or None,
        queue=execution_defaults.slurm_queue or None,
        job_name="greymodel-autofit",
        log_path=log_dir / "autofit-slurm.log",
    )
    result = subprocess.run(submit_command, capture_output=True, text=True, check=False, cwd=str(repo_root))
    if result.returncode != 0:
        raise RuntimeError("Slurm submission failed with exit code %d: %s" % (int(result.returncode), (result.stderr or result.stdout).strip()))
    job_id = (result.stdout or "").strip().split(";", 1)[0]
    return AutoFitResult(
        status="submitted",
        execution_backend="slurm",
        run_root=str(run_root),
        run_dir="",
        job_id=job_id or None,
        resolved_plan={},
        resolved_profile={},
        submitted_command=tuple(inner_command),
        submission_command=tuple(submit_command),
    )


def run_autofit(
    *,
    data: Path | str,
    model: str | Path | ModelProfile | Mapping[str, Any] | None = None,
    execution: str = "auto",
    run_root: Path | str | None = None,
    overrides: Optional[Mapping[str, Any]] = None,
    warm_start: Path | str | None = None,
    split_policy: str = "auto",
    registry_root: Path | str | None = None,
    workspace_defaults: UIExecutionDefaults | None = None,
    repo_root: Path | str | None = None,
) -> AutoFitResult:
    plan = resolve_autofit_plan(
        data=data,
        model=model,
        run_root=run_root,
        split_policy=split_policy,
        warm_start=warm_start,
        execution=execution,
        registry_root=registry_root,
        workspace_defaults=workspace_defaults,
    )
    if plan.execution_backend == "slurm":
        if workspace_defaults is None:
            raise ValueError("AutoFit slurm execution requires UIExecutionDefaults.")
        return submit_autofit_job(
            data=data,
            model=plan.model_reference,
            run_root=plan.run_root,
            split_policy=split_policy,
            warm_start=warm_start,
            overrides=overrides,
            execution_defaults=workspace_defaults,
            repo_root=repo_root or Path(__file__).resolve().parents[2],
        )

    run_dir = Path(plan.run_dir)
    stage_root = ensure_dir(Path(plan.stage_root))
    context = create_run_context(Path(plan.run_root), stage="autofit", variant=plan.variant)
    log_path = context.run_dir / "reports" / "autofit.log"
    logger = _AutoFitLogger(log_path)
    plan_dict = asdict(plan)
    snapshot_run_config(context, {"autofit_plan": plan_dict, "overrides": dict(overrides or {})})
    update_run_status(
        context,
        {
            "status": "running",
            "manifest_path": plan.manifest_path,
            "index_path": plan.index_path,
            "distributed_strategy": "autofit",
            "metadata": {"autofit": True, "messages": list(plan.metadata.get("messages", ()))},
        },
    )
    logger.log("Resolved dataset: %s (%s)" % (plan.manifest_path, plan.data_kind))
    logger.log(
        "Validation summary: %d records, %d warnings, %d errors"
        % (
            int(plan.validation_report.get("num_records", 0)),
            int(plan.validation_report.get("num_warnings", 0)),
            int(plan.validation_report.get("num_errors", 0)),
        )
    )
    for message in plan.metadata.get("messages", ()):
        logger.log(str(message))
    logger.log("Using model profile %s on variant %s" % (plan.model_reference, plan.variant))
    training_config = _training_config_from_overrides(
        overrides,
        model_profile=plan.model_reference,
        model_registry_root=str(Path(registry_root) if registry_root is not None else ensure_settings().registry_root),
    )
    try:
        records = load_dataset_manifest(plan.manifest_path)
        if not _manifest_has_train_and_val(records):
            raise ValueError("AutoFit requires both train and val splits after planning.")
        val_manifest_path = _build_split_filtered_manifest(Path(plan.manifest_path), "val", output_dir=context.run_dir / "prepared")
        logger.log("Starting finetune stage")
        finetune_result = run_finetune_stage(
            manifest_path=plan.manifest_path,
            index_path=plan.index_path,
            variant=plan.variant,
            run_root=stage_root,
            split="train",
            batch_size=training_config.per_device_batch_size,
            training_config=training_config,
            checkpoint_path=warm_start,
            resume_from=warm_start,
        )
        logger.log(
            "Finetune complete: epoch=%d step=%d best=%s"
            % (
                int(finetune_result.epoch),
                int(finetune_result.global_step),
                str(finetune_result.best_checkpoint_path or finetune_result.latest_checkpoint_path),
            )
        )
        checkpoint_path = Path(finetune_result.best_checkpoint_path or finetune_result.latest_checkpoint_path or "")
        if not checkpoint_path.exists():
            raise FileNotFoundError("AutoFit could not find a usable finetune checkpoint.")
        benchmark_profile = _build_benchmark_profile(load_model_profile(plan.model_reference, registry_root=registry_root or ensure_settings().registry_root), checkpoint_path)
        benchmark_profile_path = _materialize_runtime_profile(benchmark_profile, context.run_dir / "reports" / "autofit_model_profile.json")
        logger.log("Starting calibration stage")
        calibration_result = run_calibration_stage(
            manifest_path=val_manifest_path,
            index_path=plan.index_path,
            variant=plan.variant,
            run_root=stage_root,
            model_profile=str(benchmark_profile_path),
            model_registry_root=registry_root or ensure_settings().registry_root,
        )
        calibration_report = json.loads(Path(calibration_result.report_path).read_text(encoding="utf-8"))
        stations = dict(calibration_report.get("stations", {}))
        recommended_threshold = None
        if stations:
            first_station = next(iter(stations.values()))
            recommended_threshold = first_station.get("recommended_reject_threshold")
        logger.log("Calibration complete: recommended threshold=%s" % recommended_threshold)
        logger.log("Starting validation benchmark")
        benchmark_report = benchmark_manifest(
            val_manifest_path,
            index_path=plan.index_path,
            variant=plan.variant,
            output_path=context.run_dir / "reports" / "benchmark_report.json",
            model_profile=str(benchmark_profile_path),
            model_registry_root=registry_root or ensure_settings().registry_root,
        )
        overall = dict(benchmark_report.get("overall", {}))
        logger.log(
            "Validation metrics: accuracy=%s auroc=%s far=%s frr=%s"
            % (overall.get("accuracy"), overall.get("auroc"), overall.get("far"), overall.get("frr"))
        )
        result = AutoFitResult(
            status="completed",
            execution_backend="local",
            run_root=plan.run_root,
            run_dir=str(context.run_dir),
            log_path=str(log_path),
            summary_path=str(context.run_dir / "reports" / "autofit_summary.json"),
            markdown_summary_path=str(context.run_dir / "reports" / "autofit_summary.md"),
            report_path=str(context.run_dir / "reports" / "benchmark_report.json"),
            calibration_report_path=str(calibration_result.report_path),
            best_checkpoint_path=str(checkpoint_path),
            latest_checkpoint_path=str(finetune_result.latest_checkpoint_path or checkpoint_path),
            latest_usable_checkpoint_path=str(checkpoint_path),
            metrics=benchmark_report,
            resolved_plan=plan_dict,
            resolved_profile=benchmark_profile.to_dict(),
        )
        summary_payload = {
            "status": result.status,
            "stage": "autofit",
            "variant": plan.variant,
            "execution_backend": result.execution_backend,
            "run_dir": result.run_dir,
            "log_path": result.log_path,
            "report_path": result.report_path,
            "calibration_report_path": result.calibration_report_path,
            "best_checkpoint_path": result.best_checkpoint_path,
            "latest_checkpoint_path": result.latest_checkpoint_path,
            "latest_usable_checkpoint_path": result.latest_usable_checkpoint_path,
            "recommended_threshold": recommended_threshold,
            "overall": benchmark_report.get("overall", {}),
            "defect_family_bad_only": benchmark_report.get("defect_family_bad_only", {}),
            "resolved_plan": plan_dict,
            "resolved_profile": benchmark_profile.to_dict(),
            "resolved_profile_path": str(benchmark_profile_path),
            "child_runs": {
                "finetune_run_dir": str(finetune_result.run_dir),
                "calibration_run_dir": str(calibration_result.run_dir),
            },
        }
        summary_path, markdown_path = _write_autofit_summary(context.run_dir, summary_payload, result=result)
        write_summary(
            context,
            {
                "stage": "autofit",
                "variant": plan.variant,
                "report_path": str(result.report_path),
                "summary_path": str(summary_path),
                "latest_checkpoint_path": result.latest_checkpoint_path,
                "best_checkpoint_path": result.best_checkpoint_path,
                "epoch": int(finetune_result.epoch),
                "global_step": int(finetune_result.global_step),
            },
        )
        update_run_status(
            context,
            {
                "status": "completed",
                "completed_at": utc_timestamp(),
                "report_path": result.report_path,
                "summary_path": str(summary_path),
                "latest_checkpoint_path": result.latest_checkpoint_path,
                "best_checkpoint_path": result.best_checkpoint_path,
                "latest_usable_checkpoint_path": result.latest_usable_checkpoint_path,
                "epoch": int(finetune_result.epoch),
                "global_step": int(finetune_result.global_step),
                "metadata": {"autofit": True, "markdown_summary_path": str(markdown_path)},
            },
        )
        return replace(result, summary_path=str(summary_path), markdown_summary_path=str(markdown_path))
    except BaseException as exc:
        write_failure_bundle(
            context,
            stage="autofit",
            variant=plan.variant,
            exc=exc,
            manifest_path=plan.manifest_path,
            index_path=plan.index_path,
            metrics_path=context.metrics_path,
            metadata={"resolved_plan": plan_dict},
        )
        raise


def resume_autofit(
    run_dir: Path | str,
    *,
    execution: str = "local",
    workspace_defaults: UIExecutionDefaults | None = None,
    repo_root: Path | str | None = None,
) -> AutoFitResult:
    run_dir = Path(run_dir)
    summary_path = run_dir / "reports" / "autofit_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("AutoFit summary not found at %s." % summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    plan = dict(payload.get("resolved_plan", {}))
    if not plan:
        raise ValueError("AutoFit summary did not contain a resolved plan.")
    return run_autofit(
        data=plan["manifest_path"],
        model=str(plan["model_reference"]),
        execution=execution,
        run_root=Path(run_dir).parent,
        warm_start=payload.get("latest_usable_checkpoint_path") or payload.get("best_checkpoint_path"),
        split_policy=str(plan.get("split_policy", "auto")),
        workspace_defaults=workspace_defaults,
        repo_root=repo_root,
    )
