from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .api import BaseModel, LiteModel
from .data import ManifestInspectionDataset, collate_batch, load_dataset_index, load_dataset_manifest
from .evaluation import build_calibration_report, evaluate_predictions, predict_dataset, save_predictions
from .tracking import RunContext, create_run_context, log_metrics, snapshot_manifest, snapshot_run_config
from .training import (
    TrainingBatch,
    TrainingConfig,
    run_domain_adaptation_step,
    run_masked_pretrain_step,
    run_supervised_step,
)
from .utils import write_json


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for staged training runners.") from exc
    return torch


def _wrapper_for_variant(variant: str, num_defect_families: int, defect_families: Sequence[str]):
    if variant == "lite":
        return LiteModel(num_defect_families=num_defect_families, defect_families=defect_families)
    return BaseModel(num_defect_families=num_defect_families, defect_families=defect_families)


def _load_defect_families(index_path: Optional[Path | str]) -> tuple[str, ...]:
    if index_path is None:
        return ()
    index_path = Path(index_path)
    if not index_path.exists():
        return ()
    index = load_dataset_index(index_path)
    ontology_path = Path(index.ontology_path)
    if not ontology_path.exists():
        return ()
    import json

    return tuple(json.loads(ontology_path.read_text(encoding="utf-8")).get("defect_tags", ()))


def _select_items(dataset: ManifestInspectionDataset, batch_size: int) -> list[Mapping[str, Any]]:
    if len(dataset) == 0:
        raise ValueError("The dataset split is empty.")
    ordered_indices = dataset.station_balanced_indices()
    if not ordered_indices:
        ordered_indices = list(range(len(dataset)))
    return [dataset[index] for index in ordered_indices[: max(1, min(batch_size, len(dataset)))]]


def _training_defect_families(index_path: Optional[Path | str]) -> tuple[str, ...]:
    defect_families = _load_defect_families(index_path)
    if defect_families:
        return defect_families
    return ("unknown",)


def _supervised_targets(items: Sequence[Mapping[str, Any]], defect_families: Sequence[str]):
    torch = _require_torch()
    reject_targets = torch.as_tensor(
        [int(item["sample"].accept_reject) for item in items],
        dtype=torch.float32,
    )
    family_index = {name: offset for offset, name in enumerate(defect_families)}
    defect_targets = torch.zeros((len(items), len(defect_families)), dtype=torch.float32)
    for row_index, item in enumerate(items):
        for defect_tag in item["sample"].defect_tags:
            if defect_tag in family_index:
                defect_targets[row_index, family_index[defect_tag]] = 1.0
    return reject_targets, defect_targets


def _save_checkpoint(torch_model, context: RunContext, filename: str) -> Optional[Path]:
    if torch_model is None:
        return None
    torch = _require_torch()
    checkpoint_path = context.checkpoints_dir / filename
    torch.save(torch_model.state_dict(), checkpoint_path)
    return checkpoint_path


def _normalize_batch_for_model(batch, torch_model):
    batch.station_id = batch.station_id % max(int(torch_model.config.num_stations), 1)
    return batch


@dataclass(frozen=True)
class StageResult:
    stage: str
    variant: str
    run_dir: Path
    metrics_path: Path
    report_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    extra_paths: Mapping[str, str] | None = None


def _prepare_run(
    manifest_path: Path | str,
    index_path: Optional[Path | str],
    stage: str,
    variant: str,
    run_root: Path | str,
    payload: Mapping[str, Any],
) -> RunContext:
    context = create_run_context(run_root=run_root, stage=stage, variant=variant)
    snapshot_run_config(context, payload)
    snapshot_manifest(
        context,
        {
            "manifest_path": str(Path(manifest_path)),
            "index_path": str(index_path) if index_path is not None else None,
            "stage": stage,
            "variant": variant,
        },
    )
    return context


def run_pretraining_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
) -> StageResult:
    torch = _require_torch()
    training_config = training_config or TrainingConfig()
    dataset = ManifestInspectionDataset(manifest_path, index_path=index_path, split=split)
    items = _select_items(dataset, batch_size=batch_size)
    batch = collate_batch(items, as_torch=True)["model_input"]
    defect_families = _training_defect_families(index_path or Path(manifest_path).with_name("dataset_index.json"))
    wrapper = _wrapper_for_variant(variant, max(len(defect_families), 1), defect_families)
    if not hasattr(wrapper.backend, "model"):
        raise RuntimeError("The selected runtime backend does not support staged training.")
    torch_model = wrapper.backend.model
    batch = _normalize_batch_for_model(batch, torch_model)
    torch_model.train()
    reconstruction_head = torch.nn.Conv2d(torch_model.config.global_hidden_dim, 1, kernel_size=1)
    optimizer = torch.optim.Adam(list(torch_model.parameters()) + list(reconstruction_head.parameters()), lr=1e-3)
    metrics = run_masked_pretrain_step(torch_model, reconstruction_head, optimizer, batch, training_config)
    context = _prepare_run(
        manifest_path,
        index_path,
        stage="pretrain",
        variant=variant,
        run_root=run_root,
        payload={"split": split, "batch_size": batch_size, "training_config": asdict(training_config)},
    )
    log_metrics(context, metrics)
    report_path = write_json(context.reports_dir / "pretrain_report.json", metrics)
    checkpoint_path = _save_checkpoint(torch_model, context, "pretrain_model.pt")
    _save_checkpoint(reconstruction_head, context, "pretrain_reconstruction_head.pt")
    return StageResult("pretrain", variant, context.run_dir, context.metrics_path, report_path, checkpoint_path)


def run_domain_adaptation_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
) -> StageResult:
    torch = _require_torch()
    training_config = training_config or TrainingConfig()
    dataset = ManifestInspectionDataset(manifest_path, index_path=index_path, split=split)
    items = _select_items(dataset, batch_size=batch_size)
    batch = collate_batch(items, as_torch=True)["model_input"]
    defect_families = _training_defect_families(index_path or Path(manifest_path).with_name("dataset_index.json"))
    wrapper = _wrapper_for_variant(variant, max(len(defect_families), 1), defect_families)
    if not hasattr(wrapper.backend, "model"):
        raise RuntimeError("The selected runtime backend does not support staged training.")
    torch_model = wrapper.backend.model
    batch = _normalize_batch_for_model(batch, torch_model)
    torch_model.train()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    metrics = run_domain_adaptation_step(torch_model, optimizer, batch, training_config)
    context = _prepare_run(
        manifest_path,
        index_path,
        stage="domain_adapt",
        variant=variant,
        run_root=run_root,
        payload={"split": split, "batch_size": batch_size, "training_config": asdict(training_config)},
    )
    log_metrics(context, metrics)
    report_path = write_json(context.reports_dir / "domain_adaptation_report.json", metrics)
    checkpoint_path = _save_checkpoint(torch_model, context, "domain_adapt_model.pt")
    return StageResult("domain_adapt", variant, context.run_dir, context.metrics_path, report_path, checkpoint_path)


def run_finetune_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[Path | str] = None,
) -> StageResult:
    torch = _require_torch()
    training_config = training_config or TrainingConfig()
    dataset = ManifestInspectionDataset(manifest_path, index_path=index_path, split=split)
    items = _select_items(dataset, batch_size=batch_size)
    batch = collate_batch(items, as_torch=True)["model_input"]
    defect_families = _training_defect_families(index_path or Path(manifest_path).with_name("dataset_index.json"))
    wrapper = _wrapper_for_variant(variant, max(len(defect_families), 1), defect_families)
    if not hasattr(wrapper.backend, "model"):
        raise RuntimeError("The selected runtime backend does not support staged training.")
    torch_model = wrapper.backend.model
    batch = _normalize_batch_for_model(batch, torch_model)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
        torch_model.load_state_dict(state_dict, strict=False)
    torch_model.train()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    reject_targets, defect_targets = _supervised_targets(items, defect_families)
    metrics = run_supervised_step(
        torch_model,
        optimizer,
        TrainingBatch(model_input=batch, reject_targets=reject_targets, defect_targets=defect_targets),
        training_config,
    )
    context = _prepare_run(
        manifest_path,
        index_path,
        stage="finetune",
        variant=variant,
        run_root=run_root,
        payload={
            "split": split,
            "batch_size": batch_size,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "training_config": asdict(training_config),
        },
    )
    log_metrics(context, metrics)
    report_path = write_json(context.reports_dir / "finetune_report.json", metrics)
    checkpoint_path = _save_checkpoint(torch_model, context, "finetune_model.pt")
    return StageResult("finetune", variant, context.run_dir, context.metrics_path, report_path, checkpoint_path)


def run_resume_stage(
    manifest_path: Path | str,
    checkpoint_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
) -> StageResult:
    return run_finetune_stage(
        manifest_path=manifest_path,
        index_path=index_path,
        variant=variant,
        run_root=run_root,
        split=split,
        batch_size=batch_size,
        training_config=training_config,
        checkpoint_path=checkpoint_path,
    )


def run_benchmark_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
) -> StageResult:
    context = _prepare_run(
        manifest_path,
        index_path,
        stage="benchmark",
        variant=variant,
        run_root=run_root,
        payload={},
    )
    report = evaluate_predictions(
        load_dataset_manifest(manifest_path),
        predict_dataset(manifest_path, index_path=index_path, variant=variant),
    )
    report_path = write_json(context.reports_dir / "benchmark_report.json", report)
    log_metrics(
        context,
        {
            "accuracy": report["overall"]["accuracy"],
            "far": report["overall"]["far"],
            "frr": report["overall"]["frr"],
            "auroc": report["overall"].get("auroc"),
        },
    )
    return StageResult("benchmark", variant, context.run_dir, context.metrics_path, report_path)


def run_calibration_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
) -> StageResult:
    context = _prepare_run(
        manifest_path,
        index_path,
        stage="calibrate",
        variant=variant,
        run_root=run_root,
        payload={},
    )
    predictions = predict_dataset(manifest_path, index_path=index_path, variant=variant)
    predictions_path = save_predictions(predictions, context.reports_dir / "predictions.jsonl")
    report = build_calibration_report(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        index_path=index_path,
        output_path=context.reports_dir / "calibration_report.json",
    )
    log_metrics(
        context,
        {
            "num_records": int(report["num_records"]),
            "num_stations": int(len(report["stations"])),
        },
    )
    return StageResult(
        "calibrate",
        variant,
        context.run_dir,
        context.metrics_path,
        context.reports_dir / "calibration_report.json",
        extra_paths={"predictions_path": str(predictions_path)},
    )
