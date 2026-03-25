from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .data import (
    DistributedShardedSampler,
    ManifestInspectionDataset,
    StationBalancedManifestSampler,
    load_dataset_index,
    load_dataset_manifest,
)
from .evaluation import build_calibration_report, evaluate_predictions, predict_dataset, save_predictions
from .models import build_base_model, build_lite_model
from .tracking import (
    RunContext,
    create_run_context,
    log_epoch_metrics,
    log_metrics,
    log_step_metrics,
    snapshot_manifest,
    snapshot_run_config,
    write_summary,
)
from .training import (
    TrainingBatch,
    TrainingConfig,
    build_autocast_context,
    build_grad_scaler,
    build_scheduler,
    enforce_memory_guardrails,
    compute_domain_adaptation_objective,
    compute_masked_pretrain_objective,
    compute_supervised_objective,
    sample_pretrain_crops,
    seed_everything,
)
from .types import TensorBatch
from .utils import write_json


def _require_torch():
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for staged training runners.") from exc
    return torch, dist, DDP, DataLoader


def _require_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


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


def _training_defect_families(index_path: Optional[Path | str]) -> tuple[str, ...]:
    defect_families = _load_defect_families(index_path)
    if defect_families:
        return defect_families
    return ("unknown",)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: Any
    backend: str
    strategy: str
    created_process_group: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class StageResult:
    stage: str
    variant: str
    run_dir: Path
    metrics_path: Path
    report_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    best_checkpoint_path: Optional[Path] = None
    latest_checkpoint_path: Optional[Path] = None
    epoch: int = 0
    global_step: int = 0
    extra_paths: Mapping[str, str] | None = None


def _builder_for_variant(variant: str):
    return build_lite_model if variant == "lite" else build_base_model


def _require_fsdp():
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError as exc:
        raise ImportError("PyTorch FSDP support is required for the fsdp training strategy.") from exc
    return FSDP, ShardingStrategy, size_based_auto_wrap_policy, checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing


def _resolve_strategy(training_config: TrainingConfig, enabled: bool, device) -> str:
    requested = str(training_config.distributed_strategy).lower()
    if not enabled:
        return "single"
    if requested == "auto":
        requested = "fsdp"
    if requested == "fsdp" and getattr(device, "type", "cpu") != "cuda":
        return "ddp"
    if requested not in {"fsdp", "ddp"}:
        return "ddp"
    return requested


def _resolve_backend(distributed_backend: str, device) -> str:
    backend = distributed_backend.lower()
    if backend != "auto":
        return backend
    return "nccl" if getattr(device, "type", "cpu") == "cuda" else "gloo"


def _init_distributed(training_config: TrainingConfig) -> DistributedContext:
    torch, dist, _, _ = _require_torch()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        device = torch.device("cuda", max(local_rank, 0))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    created_process_group = False
    enabled = world_size > 1
    backend = _resolve_backend(training_config.distributed_backend, device)
    strategy = _resolve_strategy(training_config, enabled, device)
    if enabled and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        created_process_group = True
    return DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
        strategy=strategy,
        created_process_group=created_process_group,
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    _, dist, _, _ = _require_torch()
    if context.enabled and context.created_process_group and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_barrier(context: DistributedContext) -> None:
    _, dist, _, _ = _require_torch()
    if context.enabled and dist.is_initialized():
        dist.barrier()


def _reduce_scalar(value: float, context: DistributedContext) -> float:
    torch, dist, _, _ = _require_torch()
    tensor = torch.as_tensor(float(value), dtype=torch.float32, device=context.device)
    if context.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / float(context.world_size)
    return float(tensor.detach().cpu().item())


def _reduce_totals(total: float, count: float, context: DistributedContext) -> Tuple[float, float]:
    torch, dist, _, _ = _require_torch()
    packed = torch.as_tensor([float(total), float(count)], dtype=torch.float32, device=context.device)
    if context.enabled and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return float(packed[0].detach().cpu().item()), float(packed[1].detach().cpu().item())


def _prepare_run(
    manifest_path: Path | str,
    index_path: Optional[Path | str],
    stage: str,
    variant: str,
    run_root: Path | str,
    payload: Mapping[str, Any],
    context: DistributedContext,
) -> RunContext:
    run_context = create_run_context(run_root=run_root, stage=stage, variant=variant)
    if context.is_main_process:
        snapshot_run_config(run_context, payload)
        snapshot_manifest(
            run_context,
            {
                "manifest_path": str(Path(manifest_path)),
                "index_path": str(index_path) if index_path is not None else None,
                "stage": stage,
                "variant": variant,
                "rank": context.rank,
                "world_size": context.world_size,
                "backend": context.backend,
                "strategy": context.strategy,
            },
        )
    _maybe_barrier(context)
    return run_context


def _collate_for_training(items, defect_families: Sequence[str]):
    torch, _, _, _ = _require_torch()
    from .data import collate_batch

    batch = collate_batch(items, as_torch=True)
    model_input = batch["model_input"]
    family_index = {name: offset for offset, name in enumerate(defect_families)}
    defect_targets = torch.zeros((len(items), len(defect_families)), dtype=torch.float32)
    for row_index, sample in enumerate(batch["samples"]):
        for defect_tag in sample.defect_tags:
            if defect_tag in family_index:
                defect_targets[row_index, family_index[defect_tag]] = 1.0
    reject_targets = torch.as_tensor(batch["accept_reject"], dtype=torch.float32)
    return {
        "model_input": model_input,
        "reject_targets": reject_targets,
        "defect_targets": defect_targets,
        "samples": batch["samples"],
        "records": batch["records"],
    }


def _move_training_batch_to_device(batch_payload: Mapping[str, Any], device, channels_last: bool = False) -> Mapping[str, Any]:
    torch, _, _, _ = _require_torch()
    if getattr(device, "type", "cpu") == "cpu":
        if channels_last:
            model_input = batch_payload["model_input"]
            return {
                **batch_payload,
                "model_input": TensorBatch(
                    image=model_input.image.contiguous(memory_format=torch.channels_last),
                    valid_mask=model_input.valid_mask.contiguous(memory_format=torch.channels_last),
                    station_id=model_input.station_id,
                    geometry_id=model_input.geometry_id,
                    metadata=model_input.metadata,
                ),
            }
        return batch_payload
    model_input = batch_payload["model_input"]
    image = model_input.image.to(device, non_blocking=True)
    valid_mask = model_input.valid_mask.to(device, non_blocking=True)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
        valid_mask = valid_mask.contiguous(memory_format=torch.channels_last)
    return {
        **batch_payload,
        "model_input": TensorBatch(
            image=image,
            valid_mask=valid_mask,
            station_id=model_input.station_id.to(device, non_blocking=True),
            geometry_id=model_input.geometry_id.to(device, non_blocking=True),
            metadata=model_input.metadata,
        ),
        "reject_targets": batch_payload["reject_targets"].to(device, non_blocking=True),
        "defect_targets": batch_payload["defect_targets"].to(device, non_blocking=True),
    }


def _create_progress_bar(
    training_config: TrainingConfig,
    context: DistributedContext,
    total: int,
    desc: str,
):
    if not training_config.show_progress or not context.is_main_process or total <= 0:
        return None
    tqdm = _require_tqdm()
    if tqdm is None:
        return None
    return tqdm(total=int(total), desc=desc, dynamic_ncols=True, leave=True)


def _build_runtime_dataloaders(
    manifest_path: Path | str,
    index_path: Optional[Path | str],
    training_config: TrainingConfig,
    context: DistributedContext,
    defect_families: Sequence[str],
    stage: str,
    train_split: str,
):
    _, _, _, DataLoader = _require_torch()
    train_dataset = ManifestInspectionDataset(manifest_path, index_path=index_path, split=train_split)
    val_dataset = ManifestInspectionDataset(manifest_path, index_path=index_path, split="val")
    if len(train_dataset) == 0:
        raise ValueError("The dataset train split is empty.")

    station_balanced = stage == "finetune" and training_config.station_balanced_sampling
    train_sampler = (
        StationBalancedManifestSampler(
            train_dataset.records,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            seed=training_config.seed,
        )
        if station_balanced
        else DistributedShardedSampler(
            list(range(len(train_dataset))),
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=training_config.seed,
        )
    )
    val_sampler = DistributedShardedSampler(
        list(range(len(val_dataset))),
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=False,
        seed=training_config.seed,
    )

    common_kwargs = {
        "num_workers": max(0, int(training_config.num_workers)),
        "persistent_workers": bool(training_config.persistent_workers and training_config.num_workers > 0),
        "pin_memory": getattr(context.device, "type", "cpu") == "cuda",
        "batch_size": max(1, int(training_config.per_device_batch_size)),
        "collate_fn": partial(_collate_for_training, defect_families=defect_families),
    }
    if training_config.num_workers > 0:
        common_kwargs["prefetch_factor"] = max(2, int(training_config.prefetch_factor))

    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, **common_kwargs)
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **common_kwargs)
    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler


def _normalize_model_input(model_input: TensorBatch, num_stations: int) -> TensorBatch:
    return TensorBatch(
        image=model_input.image,
        valid_mask=model_input.valid_mask,
        station_id=model_input.station_id % max(int(num_stations), 1),
        geometry_id=model_input.geometry_id,
        metadata=model_input.metadata,
    )


def _build_modules(stage: str, variant: str, defect_families: Sequence[str]):
    torch, _, _, _ = _require_torch()
    builder = _builder_for_variant(variant)
    return builder, torch


def _instantiate_modules(stage: str, variant: str, defect_families: Sequence[str], training_config: TrainingConfig):
    builder, torch = _build_modules(stage, variant, defect_families)
    backbone = builder(
        num_defect_families=max(len(defect_families), 1),
        defect_families=defect_families,
        activation_checkpointing=bool(training_config.activation_checkpointing),
        max_global_feature_grid=max(4, int(training_config.max_global_feature_grid)),
    )
    auxiliary_modules: Dict[str, Any] = {}
    if stage == "pretrain":
        auxiliary_modules["reconstruction_head"] = torch.nn.Conv2d(backbone.config.global_hidden_dim, 1, kernel_size=1)
    return backbone, auxiliary_modules


def _move_modules_to_device(backbone, auxiliary_modules: Mapping[str, Any], device, training_config: TrainingConfig):
    backbone = backbone.to(device)
    moved_auxiliary = {name: module.to(device) for name, module in auxiliary_modules.items()}
    if bool(training_config.channels_last):
        torch, _, _, _ = _require_torch()
        backbone = backbone.to(memory_format=torch.channels_last)
        moved_auxiliary = {name: module.to(memory_format=torch.channels_last) for name, module in moved_auxiliary.items()}
    return backbone, moved_auxiliary


def _stage_uses_partial_backbone(stage: str) -> bool:
    return stage in {"pretrain", "domain_adapt"}


def _configure_activation_checkpointing(backbone, training_config: TrainingConfig):
    if not bool(training_config.activation_checkpointing):
        return backbone
    try:
        from .models.grayinspect import BiFPNBlock, CoarseContextBlock, ConvNeXtBlock
    except Exception:
        return backbone
    FSDP, ShardingStrategy, size_based_auto_wrap_policy, checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing = _require_fsdp()

    def _checkpointable(module):
        return isinstance(module, (ConvNeXtBlock, BiFPNBlock, CoarseContextBlock))

    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(backbone, checkpoint_wrapper_fn=wrapper, check_fn=_checkpointable)
    return backbone


def _wrap_modules_for_ddp(backbone, auxiliary_modules: Mapping[str, Any], context: DistributedContext, stage: str):
    _, _, DDP, _ = _require_torch()
    if not context.enabled:
        return backbone, dict(auxiliary_modules)
    ddp_kwargs = {"find_unused_parameters": _stage_uses_partial_backbone(stage)}
    if getattr(context.device, "type", "cpu") == "cuda":
        backbone = DDP(backbone, device_ids=[context.local_rank], output_device=context.local_rank, **ddp_kwargs)
        wrapped_auxiliary = {
            name: DDP(module, device_ids=[context.local_rank], output_device=context.local_rank, **ddp_kwargs)
            for name, module in auxiliary_modules.items()
        }
    else:
        backbone = DDP(backbone, **ddp_kwargs)
        wrapped_auxiliary = {name: DDP(module, **ddp_kwargs) for name, module in auxiliary_modules.items()}
    return backbone, wrapped_auxiliary


def _wrap_modules_for_fsdp(backbone, auxiliary_modules: Mapping[str, Any], context: DistributedContext, training_config: TrainingConfig):
    if not context.enabled:
        return backbone, dict(auxiliary_modules)
    FSDP, ShardingStrategy, size_based_auto_wrap_policy, _, _, _ = _require_fsdp()
    auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=50000)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap,
        "device_id": context.local_rank if getattr(context.device, "type", "cpu") == "cuda" else None,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "use_orig_params": True,
        "limit_all_gathers": True,
    }
    backbone = FSDP(backbone, **fsdp_kwargs)
    wrapped_auxiliary = {name: FSDP(module, **fsdp_kwargs) for name, module in auxiliary_modules.items()}
    return backbone, wrapped_auxiliary


def _wrap_modules_for_strategy(
    backbone,
    auxiliary_modules: Mapping[str, Any],
    context: DistributedContext,
    stage: str,
    training_config: TrainingConfig,
):
    backbone = _configure_activation_checkpointing(backbone, training_config)
    if context.strategy == "fsdp":
        return _wrap_modules_for_fsdp(backbone, auxiliary_modules, context, training_config)
    return _wrap_modules_for_ddp(backbone, auxiliary_modules, context, stage=stage)


def _unwrap_module(module):
    if hasattr(module, "module"):
        return module.module
    if module.__class__.__name__ == "FullyShardedDataParallel" and hasattr(module, "_fsdp_wrapped_module"):
        wrapped = module._fsdp_wrapped_module
        return getattr(wrapped, "module", wrapped)
    return module


def _maybe_compile_modules(backbone, auxiliary_modules: Mapping[str, Any], training_config: TrainingConfig, context: DistributedContext):
    torch, _, _, _ = _require_torch()
    if not bool(training_config.compile_model):
        return backbone, dict(auxiliary_modules)
    if not hasattr(torch, "compile") or context.strategy != "single":
        return backbone, dict(auxiliary_modules)
    backbone = torch.compile(backbone)
    return backbone, dict(auxiliary_modules)


def _optimizer_and_scheduler(backbone, auxiliary_modules: Mapping[str, Any], training_config: TrainingConfig, total_optimizer_steps: int):
    torch, _, _, _ = _require_torch()
    parameters = list(backbone.parameters())
    for module in auxiliary_modules.values():
        parameters.extend(list(module.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=total_optimizer_steps, warmup_steps=training_config.warmup_steps)
    return optimizer, scheduler


def _checkpoint_payload(
    stage: str,
    variant: str,
    manifest_path: Path | str,
    index_path: Optional[Path | str],
    training_config: TrainingConfig,
    defect_families: Sequence[str],
    epoch: int,
    global_step: int,
    best_val_metric: Optional[float],
    backbone,
    auxiliary_modules: Mapping[str, Any],
    optimizer,
    scheduler,
    scaler,
    ema_state: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    return {
        "stage": stage,
        "variant": variant,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_metric": best_val_metric,
        "manifest_path": str(Path(manifest_path)),
        "index_path": str(index_path) if index_path is not None else None,
        "training_config": asdict(training_config),
        "defect_families": list(defect_families),
        "model_state": _unwrap_module(backbone).state_dict(),
        "auxiliary_state": {name: _unwrap_module(module).state_dict() for name, module in auxiliary_modules.items()},
        "ema_state": dict(ema_state or {}),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }


def _load_checkpoint(path: Path | str):
    torch, _, _, _ = _require_torch()
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _restore_from_checkpoint(
    checkpoint_path: Optional[Path | str],
    backbone,
    auxiliary_modules: Mapping[str, Any],
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_optimizer: bool = False,
):
    start_epoch = 0
    global_step = 0
    best_val_metric = None
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return start_epoch, global_step, best_val_metric
    checkpoint = _load_checkpoint(checkpoint_path)
    if "model_state" in checkpoint:
        _unwrap_module(backbone).load_state_dict(checkpoint["model_state"], strict=False)
        for name, module in auxiliary_modules.items():
            state = checkpoint.get("auxiliary_state", {}).get(name)
            if state is not None:
                _unwrap_module(module).load_state_dict(state, strict=False)
        if restore_optimizer and optimizer is not None and checkpoint.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception:
                restore_optimizer = False
        if restore_optimizer and scheduler is not None and checkpoint.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception:
                restore_optimizer = False
        if restore_optimizer and scaler is not None and checkpoint.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception:
                restore_optimizer = False
        if not restore_optimizer:
            start_epoch = 0
            global_step = 0
            best_val_metric = None
            return start_epoch, global_step, best_val_metric
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_val_metric = checkpoint.get("best_val_metric")
        return start_epoch, global_step, best_val_metric
    _unwrap_module(backbone).load_state_dict(checkpoint, strict=False)
    return start_epoch, global_step, best_val_metric


def _prune_periodic_checkpoints(context: RunContext, prefix: str, keep_last_k: int) -> None:
    keep_last_k = max(int(keep_last_k), 0)
    checkpoint_paths = sorted(context.checkpoints_dir.glob("%s-*.pt" % prefix))
    if keep_last_k == 0:
        for checkpoint_path in checkpoint_paths:
            checkpoint_path.unlink(missing_ok=True)
        return
    for checkpoint_path in checkpoint_paths[:-keep_last_k]:
        checkpoint_path.unlink(missing_ok=True)


def _save_checkpoint(
    context: RunContext,
    context_dist: DistributedContext,
    payload: Mapping[str, Any],
    filename: str,
    best: bool = False,
    periodic_prefix: Optional[str] = None,
    keep_last_k: int = 2,
) -> Optional[Path]:
    torch, _, _, _ = _require_torch()
    if not context_dist.is_main_process:
        _maybe_barrier(context_dist)
        return None
    checkpoint_path = context.checkpoints_dir / filename
    torch.save(dict(payload), checkpoint_path)
    torch.save(dict(payload), context.latest_checkpoint_path)
    if best:
        torch.save(dict(payload), context.best_checkpoint_path)
    if periodic_prefix is not None:
        _prune_periodic_checkpoints(context, periodic_prefix, keep_last_k)
    _maybe_barrier(context_dist)
    return checkpoint_path


def _write_stage_auxiliary_artifacts(stage: str, context: RunContext, context_dist: DistributedContext, auxiliary_modules: Mapping[str, Any]) -> None:
    torch, _, _, _ = _require_torch()
    if not context_dist.is_main_process:
        _maybe_barrier(context_dist)
        return
    if stage == "pretrain" and "reconstruction_head" in auxiliary_modules:
        torch.save(
            _unwrap_module(auxiliary_modules["reconstruction_head"]).state_dict(),
            context.checkpoints_dir / "pretrain_reconstruction_head.pt",
        )
    _maybe_barrier(context_dist)


def _samples_in_batch(batch_payload: Mapping[str, Any]) -> int:
    return int(batch_payload["model_input"].image.shape[0])


def _metric_value(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _memory_metrics(context: DistributedContext) -> Dict[str, float]:
    torch, dist, _, _ = _require_torch()
    if getattr(context.device, "type", "cpu") != "cuda":
        return {}
    stats = {
        "memory_allocated_mb": float(torch.cuda.memory_allocated(context.device) / (1024.0 * 1024.0)),
        "memory_reserved_mb": float(torch.cuda.memory_reserved(context.device) / (1024.0 * 1024.0)),
        "memory_peak_allocated_mb": float(torch.cuda.max_memory_allocated(context.device) / (1024.0 * 1024.0)),
        "memory_peak_reserved_mb": float(torch.cuda.max_memory_reserved(context.device) / (1024.0 * 1024.0)),
    }
    if context.enabled and dist.is_initialized():
        reduced = {}
        for key, value in stats.items():
            tensor = torch.as_tensor(value, dtype=torch.float32, device=context.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            reduced["max_%s" % key] = float(tensor.detach().cpu().item())
        stats.update(reduced)
    return stats


def _rank_memory_snapshot(context: DistributedContext) -> Optional[Dict[str, float]]:
    torch, dist, _, _ = _require_torch()
    if getattr(context.device, "type", "cpu") != "cuda":
        return None
    payload = {
        "rank": int(context.rank),
        "allocated_mb": float(torch.cuda.memory_allocated(context.device) / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved(context.device) / (1024.0 * 1024.0)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(context.device) / (1024.0 * 1024.0)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(context.device) / (1024.0 * 1024.0)),
    }
    if not context.enabled or not dist.is_initialized():
        return payload
    gathered = [None for _ in range(context.world_size)] if context.is_main_process else None
    dist.gather_object(payload, gathered, dst=0)
    return {"ranks": gathered} if context.is_main_process else None


def _reset_peak_memory(context: DistributedContext) -> None:
    torch, _, _, _ = _require_torch()
    if getattr(context.device, "type", "cpu") == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)


class ExponentialMovingAverage:
    def __init__(self, modules: Mapping[str, Any], decay: float) -> None:
        self.decay = float(decay)
        self.state = {
            name: {key: value.detach().cpu().clone() for key, value in _unwrap_module(module).state_dict().items()}
            for name, module in modules.items()
        }

    def update(self, modules: Mapping[str, Any]) -> None:
        if self.decay <= 0.0:
            return
        for name, module in modules.items():
            current_state = _unwrap_module(module).state_dict()
            shadow = self.state.setdefault(name, {})
            for key, value in current_state.items():
                detached = value.detach().cpu()
                if key not in shadow:
                    shadow[key] = detached.clone()
                    continue
                shadow[key].mul_(self.decay).add_(detached, alpha=1.0 - self.decay)


def _validation_epoch(
    stage: str,
    backbone,
    auxiliary_modules: Mapping[str, Any],
    loader,
    training_config: TrainingConfig,
    context: DistributedContext,
) -> Dict[str, float]:
    if loader is None:
        return {}
    for module in auxiliary_modules.values():
        module.eval()
    backbone.eval()
    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    import torch

    with torch.no_grad():
        for batch_payload in loader:
            batch_payload = _move_training_batch_to_device(
                batch_payload,
                context.device,
                channels_last=bool(training_config.channels_last),
            )
            model_input = _normalize_model_input(batch_payload["model_input"], _unwrap_module(backbone).config.num_stations)
            if stage == "pretrain":
                model_input = sample_pretrain_crops(model_input, training_config)
            enforce_memory_guardrails(model_input, training_config, stage)
            with build_autocast_context(training_config, context.device):
                if stage == "pretrain":
                    loss, metrics, _ = compute_masked_pretrain_objective(
                        backbone,
                        auxiliary_modules["reconstruction_head"],
                        model_input,
                        training_config,
                    )
                elif stage == "domain_adapt":
                    loss, metrics, _ = compute_domain_adaptation_objective(backbone, model_input, training_config)
                else:
                    loss, metrics, _ = compute_supervised_objective(
                        backbone,
                        TrainingBatch(
                            model_input=model_input,
                            reject_targets=batch_payload["reject_targets"],
                            defect_targets=batch_payload["defect_targets"],
                        ),
                        training_config,
                    )
            sample_count = _samples_in_batch(batch_payload)
            totals["loss"] = totals.get("loss", 0.0) + _metric_value(loss) * sample_count
            counts["loss"] = counts.get("loss", 0.0) + sample_count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + _metric_value(value) * sample_count
                counts[key] = counts.get(key, 0.0) + sample_count
    reduced = {}
    for key, total in totals.items():
        total_value, count_value = _reduce_totals(total, counts.get(key, 0.0), context)
        reduced[key] = total_value / max(count_value, 1.0)
    return {"val_%s" % key: value for key, value in reduced.items()}


def _run_training_loop(
    stage: str,
    variant: str,
    manifest_path: Path | str,
    index_path: Optional[Path | str],
    run_root: Path | str,
    training_config: TrainingConfig,
    split: str = "train",
    checkpoint_path: Optional[Path | str] = None,
    resume_from: Optional[Path | str] = None,
) -> StageResult:
    distributed_context = _init_distributed(training_config)
    seed_everything(training_config.seed + distributed_context.rank)
    try:
        defect_families = _training_defect_families(index_path or Path(manifest_path).with_name("dataset_index.json"))
        _, _, train_loader, val_loader, train_sampler, _ = _build_runtime_dataloaders(
            manifest_path,
            index_path,
            training_config,
            distributed_context,
            defect_families,
            stage,
            split,
        )
        resolved_grad_accum_steps = training_config.resolved_grad_accum_steps(distributed_context.world_size)
        total_micro_steps = min(len(train_loader), training_config.steps_per_epoch or len(train_loader))
        total_optimizer_steps_per_epoch = max(1, math.ceil(total_micro_steps / float(resolved_grad_accum_steps)))
        total_optimizer_steps = max(1, training_config.epochs * total_optimizer_steps_per_epoch)
        backbone, auxiliary_modules = _instantiate_modules(stage, variant, defect_families, training_config)
        backbone, auxiliary_modules = _move_modules_to_device(
            backbone,
            auxiliary_modules,
            distributed_context.device,
            training_config,
        )
        backbone, auxiliary_modules = _wrap_modules_for_strategy(
            backbone,
            auxiliary_modules,
            distributed_context,
            stage=stage,
            training_config=training_config,
        )
        backbone, auxiliary_modules = _maybe_compile_modules(backbone, auxiliary_modules, training_config, distributed_context)
        optimizer, scheduler = _optimizer_and_scheduler(backbone, auxiliary_modules, training_config, total_optimizer_steps)
        scaler = build_grad_scaler(training_config, distributed_context.device)
        ema_tracker = None
        if float(training_config.ema_decay) > 0.0 and distributed_context.strategy == "single":
            ema_tracker = ExponentialMovingAverage({"backbone": backbone, **auxiliary_modules}, decay=training_config.ema_decay)

        full_resume_path = resume_from or training_config.resume_from
        if full_resume_path is not None and Path(full_resume_path).exists():
            checkpoint_metadata = _load_checkpoint(full_resume_path)
            if checkpoint_metadata.get("stage") != stage:
                checkpoint_path = checkpoint_path or full_resume_path
                full_resume_path = None
        start_epoch, global_step, best_val_metric = _restore_from_checkpoint(
            full_resume_path,
            backbone,
            auxiliary_modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_optimizer=True,
        )
        if full_resume_path is None:
            _restore_from_checkpoint(checkpoint_path, backbone, auxiliary_modules, restore_optimizer=False)

        run_context = _prepare_run(
            manifest_path,
            index_path,
            stage,
            variant,
            run_root,
            {
                "training_config": asdict(training_config),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "resume_from": str(full_resume_path) if full_resume_path is not None else None,
                "effective_global_batch_size": training_config.effective_global_batch_size(distributed_context.world_size),
                "distributed_strategy": distributed_context.strategy,
            },
            distributed_context,
        )

        best_checkpoint_path: Optional[Path] = None
        latest_checkpoint_path: Optional[Path] = None
        latest_report_path: Optional[Path] = None
        total_samples_seen = 0
        stage_start_time = time.perf_counter()

        for epoch in range(start_epoch, training_config.epochs):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            backbone.train()
            for module in auxiliary_modules.values():
                module.train()
            optimizer.zero_grad(set_to_none=True)
            _reset_peak_memory(distributed_context)
            micro_loss_total = 0.0
            micro_metrics_total: Dict[str, float] = {}
            micro_sample_total = 0
            micro_batches_since_step = 0
            epoch_totals: Dict[str, float] = {}
            epoch_counts: Dict[str, float] = {}
            optimizer_steps_this_epoch = 0
            effective_micro_steps = min(len(train_loader), training_config.steps_per_epoch or len(train_loader))
            progress_bar = _create_progress_bar(
                training_config,
                distributed_context,
                total=effective_micro_steps,
                desc="%s epoch %d/%d" % (stage, epoch + 1, training_config.epochs),
            )
            try:
                for micro_step_index, batch_payload in enumerate(train_loader, start=1):
                    if micro_step_index > effective_micro_steps:
                        break
                    batch_payload = _move_training_batch_to_device(
                        batch_payload,
                        distributed_context.device,
                        channels_last=bool(training_config.channels_last),
                    )
                    model_input = _normalize_model_input(batch_payload["model_input"], _unwrap_module(backbone).config.num_stations)
                    if stage == "pretrain":
                        model_input = sample_pretrain_crops(model_input, training_config)
                    enforce_memory_guardrails(model_input, training_config, stage)
                    with build_autocast_context(training_config, distributed_context.device):
                        if stage == "pretrain":
                            loss, metrics, _ = compute_masked_pretrain_objective(
                                backbone,
                                auxiliary_modules["reconstruction_head"],
                                model_input,
                                training_config,
                            )
                        elif stage == "domain_adapt":
                            loss, metrics, _ = compute_domain_adaptation_objective(backbone, model_input, training_config)
                        else:
                            loss, metrics, _ = compute_supervised_objective(
                                backbone,
                                TrainingBatch(
                                    model_input=model_input,
                                    reject_targets=batch_payload["reject_targets"],
                                    defect_targets=batch_payload["defect_targets"],
                                ),
                                training_config,
                            )
                        scaled_loss = loss / float(resolved_grad_accum_steps)

                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    batch_samples = _samples_in_batch(batch_payload)
                    total_samples_seen += batch_samples * distributed_context.world_size
                    micro_loss_total += _metric_value(loss) * batch_samples
                    micro_sample_total += batch_samples
                    micro_batches_since_step += 1
                    epoch_totals["loss"] = epoch_totals.get("loss", 0.0) + _metric_value(loss) * batch_samples
                    epoch_counts["loss"] = epoch_counts.get("loss", 0.0) + batch_samples
                    for key, value in metrics.items():
                        metric_value = _metric_value(value)
                        micro_metrics_total[key] = micro_metrics_total.get(key, 0.0) + metric_value * batch_samples
                        epoch_totals[key] = epoch_totals.get(key, 0.0) + metric_value * batch_samples
                        epoch_counts[key] = epoch_counts.get(key, 0.0) + batch_samples

                    should_step = micro_batches_since_step >= resolved_grad_accum_steps or micro_step_index == effective_micro_steps
                    if not should_step:
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue

                    import torch

                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    parameters = list(backbone.parameters())
                    for module in auxiliary_modules.values():
                        parameters.extend(list(module.parameters()))
                    torch.nn.utils.clip_grad_norm_(parameters, training_config.gradient_clip_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema_tracker is not None:
                        ema_tracker.update({"backbone": backbone, **auxiliary_modules})
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    train_metrics = {}
                    loss_total, sample_total = _reduce_totals(micro_loss_total, micro_sample_total, distributed_context)
                    train_metrics["loss"] = loss_total / max(sample_total, 1.0)
                    for key, total in micro_metrics_total.items():
                        total_value, count_value = _reduce_totals(total, micro_sample_total, distributed_context)
                        train_metrics[key] = total_value / max(count_value, 1.0)
                    train_metrics.update(
                        {
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "samples_per_second": float(total_samples_seen) / max(time.perf_counter() - stage_start_time, 1e-6),
                            "effective_global_batch_size": training_config.effective_global_batch_size(distributed_context.world_size),
                            "distributed_strategy": distributed_context.strategy,
                        }
                    )
                    if training_config.memory_report:
                        train_metrics.update(_memory_metrics(distributed_context))
                    if distributed_context.is_main_process and (
                        global_step == 1 or global_step % max(training_config.log_every_n_steps, 1) == 0
                    ):
                        log_step_metrics(run_context, train_metrics)

                    if progress_bar is not None:
                        progress_bar.update(micro_batches_since_step)
                        progress_bar.set_postfix(
                            loss="%.4f" % float(train_metrics["loss"]),
                            lr="%.2e" % float(train_metrics["learning_rate"]),
                            step=int(global_step),
                        )

                    if val_loader is not None and training_config.val_every_n_steps > 0 and global_step % training_config.val_every_n_steps == 0:
                        val_metrics = _validation_epoch(stage, backbone, auxiliary_modules, val_loader, training_config, distributed_context)
                        if distributed_context.is_main_process and val_metrics:
                            log_step_metrics(run_context, {"epoch": epoch + 1, "global_step": global_step, **val_metrics, "event": "validation"})
                        candidate_metric = val_metrics.get("val_loss")
                        if candidate_metric is not None and (best_val_metric is None or candidate_metric < best_val_metric):
                            best_val_metric = float(candidate_metric)
                            best_checkpoint_path = _save_checkpoint(
                                run_context,
                                distributed_context,
                                _checkpoint_payload(
                                    stage,
                                    variant,
                                    manifest_path,
                                    index_path,
                                    training_config,
                                    defect_families,
                                    epoch + 1,
                                    global_step,
                                    best_val_metric,
                                    backbone,
                                    auxiliary_modules,
                                    optimizer,
                                    scheduler,
                                    scaler,
                                    ema_state=ema_tracker.state if ema_tracker is not None else None,
                                ),
                                "best.pt",
                                best=True,
                            )

                    if training_config.checkpoint_every_n_steps > 0 and global_step % training_config.checkpoint_every_n_steps == 0:
                        latest_checkpoint_path = _save_checkpoint(
                            run_context,
                            distributed_context,
                            _checkpoint_payload(
                                stage,
                                variant,
                                manifest_path,
                                index_path,
                                training_config,
                                defect_families,
                                epoch + 1,
                                global_step,
                                best_val_metric,
                                backbone,
                                auxiliary_modules,
                                optimizer,
                                scheduler,
                                scaler,
                                ema_state=ema_tracker.state if ema_tracker is not None else None,
                            ),
                            "%s-step-%06d.pt" % (stage, global_step),
                            periodic_prefix="%s-step" % stage,
                            keep_last_k=training_config.keep_last_k_checkpoints,
                        )

                    micro_loss_total = 0.0
                    micro_metrics_total = {}
                    micro_sample_total = 0
                    micro_batches_since_step = 0
                if progress_bar is not None and progress_bar.n < effective_micro_steps:
                    progress_bar.update(effective_micro_steps - progress_bar.n)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            val_metrics = _validation_epoch(stage, backbone, auxiliary_modules, val_loader, training_config, distributed_context)
            reduced_epoch_metrics = {}
            for key, total in epoch_totals.items():
                total_value, count_value = _reduce_totals(total, epoch_counts.get(key, 0.0), distributed_context)
                reduced_epoch_metrics[key] = total_value / max(count_value, 1.0)
            reduced_epoch_metrics = {"train_%s" % key: value for key, value in reduced_epoch_metrics.items()}
            reduced_epoch_metrics.update(val_metrics)
            reduced_epoch_metrics.update(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "optimizer_steps": optimizer_steps_this_epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "distributed_strategy": distributed_context.strategy,
                }
            )
            if training_config.memory_report:
                reduced_epoch_metrics.update(_memory_metrics(distributed_context))
            if distributed_context.is_main_process:
                log_epoch_metrics(run_context, reduced_epoch_metrics)

            candidate_metric = val_metrics.get("val_loss")
            if candidate_metric is not None and (best_val_metric is None or candidate_metric < best_val_metric):
                best_val_metric = float(candidate_metric)
                best_checkpoint_path = _save_checkpoint(
                    run_context,
                    distributed_context,
                    _checkpoint_payload(
                        stage,
                        variant,
                        manifest_path,
                        index_path,
                        training_config,
                        defect_families,
                        epoch + 1,
                        global_step,
                        best_val_metric,
                        backbone,
                        auxiliary_modules,
                        optimizer,
                        scheduler,
                        scaler,
                        ema_state=ema_tracker.state if ema_tracker is not None else None,
                    ),
                    "best.pt",
                    best=True,
                )

            latest_checkpoint_path = _save_checkpoint(
                run_context,
                distributed_context,
                _checkpoint_payload(
                    stage,
                    variant,
                    manifest_path,
                    index_path,
                    training_config,
                    defect_families,
                    epoch + 1,
                    global_step,
                    best_val_metric,
                    backbone,
                    auxiliary_modules,
                    optimizer,
                    scheduler,
                    scaler,
                    ema_state=ema_tracker.state if ema_tracker is not None else None,
                ),
                "%s-epoch-%03d.pt" % (stage, epoch + 1),
                periodic_prefix="%s-epoch" % stage,
                keep_last_k=training_config.keep_last_k_checkpoints,
            )
            _write_stage_auxiliary_artifacts(stage, run_context, distributed_context, auxiliary_modules)

            summary_payload = {
                "stage": stage,
                "variant": variant,
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_metric": best_val_metric,
                "latest_checkpoint_path": str(run_context.latest_checkpoint_path),
                "best_checkpoint_path": str(run_context.best_checkpoint_path if best_val_metric is not None else run_context.latest_checkpoint_path),
                "effective_global_batch_size": training_config.effective_global_batch_size(distributed_context.world_size),
                "world_size": distributed_context.world_size,
                "samples_per_second": float(total_samples_seen) / max(time.perf_counter() - stage_start_time, 1e-6),
                "distributed_strategy": distributed_context.strategy,
            }
            if training_config.memory_report:
                summary_payload.update(_memory_metrics(distributed_context))
                rank_memory = _rank_memory_snapshot(distributed_context)
                if rank_memory is not None:
                    summary_payload["rank_memory"] = rank_memory.get("ranks", rank_memory)
            if distributed_context.is_main_process:
                latest_report_path = write_json(run_context.reports_dir / ("%s_report.json" % stage), {**summary_payload, **reduced_epoch_metrics})
                write_summary(run_context, {**summary_payload, **reduced_epoch_metrics})

        if distributed_context.is_main_process:
            log_metrics(
                run_context,
                {
                    "stage": stage,
                    "epoch": training_config.epochs,
                    "global_step": global_step,
                    "best_val_metric": best_val_metric,
                    "distributed_strategy": distributed_context.strategy,
                },
            )
        _maybe_barrier(distributed_context)
        checkpoint_path_out = latest_checkpoint_path or (run_context.latest_checkpoint_path if run_context.latest_checkpoint_path.exists() else None)
        best_path_out = best_checkpoint_path or (run_context.best_checkpoint_path if run_context.best_checkpoint_path.exists() else checkpoint_path_out)
        return StageResult(
            stage=stage,
            variant=variant,
            run_dir=run_context.run_dir,
            metrics_path=run_context.metrics_path,
            report_path=latest_report_path,
            checkpoint_path=checkpoint_path_out,
            best_checkpoint_path=best_path_out,
            latest_checkpoint_path=checkpoint_path_out,
            epoch=training_config.epochs,
            global_step=global_step,
            extra_paths={
                "epoch_metrics_path": str(run_context.epoch_metrics_path),
                "summary_path": str(run_context.summary_path),
            },
        )
    finally:
        _cleanup_distributed(distributed_context)


def run_pretraining_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[Path | str] = None,
    resume_from: Optional[Path | str] = None,
) -> StageResult:
    training_config = training_config or TrainingConfig()
    if batch_size != training_config.per_device_batch_size:
        training_config = TrainingConfig(**{**asdict(training_config), "per_device_batch_size": int(batch_size)})
    return _run_training_loop(
        stage="pretrain",
        variant=variant,
        manifest_path=manifest_path,
        index_path=index_path,
        run_root=run_root,
        training_config=training_config,
        split=split,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )


def run_domain_adaptation_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[Path | str] = None,
    resume_from: Optional[Path | str] = None,
) -> StageResult:
    training_config = training_config or TrainingConfig()
    if batch_size != training_config.per_device_batch_size:
        training_config = TrainingConfig(**{**asdict(training_config), "per_device_batch_size": int(batch_size)})
    return _run_training_loop(
        stage="domain_adapt",
        variant=variant,
        manifest_path=manifest_path,
        index_path=index_path,
        run_root=run_root,
        training_config=training_config,
        split=split,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )


def run_finetune_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
    split: str = "train",
    batch_size: int = 4,
    training_config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[Path | str] = None,
    resume_from: Optional[Path | str] = None,
) -> StageResult:
    training_config = training_config or TrainingConfig()
    if batch_size != training_config.per_device_batch_size:
        training_config = TrainingConfig(**{**asdict(training_config), "per_device_batch_size": int(batch_size)})
    return _run_training_loop(
        stage="finetune",
        variant=variant,
        manifest_path=manifest_path,
        index_path=index_path,
        run_root=run_root,
        training_config=training_config,
        split=split,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )


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
    training_config = training_config or TrainingConfig()
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
    distributed_context = DistributedContext(False, 0, 1, 0, "cpu", "gloo", "single", False)
    run_context = _prepare_run(
        manifest_path,
        index_path,
        stage="benchmark",
        variant=variant,
        run_root=run_root,
        payload={},
        context=distributed_context,
    )
    report = evaluate_predictions(
        load_dataset_manifest(manifest_path),
        predict_dataset(manifest_path, index_path=index_path, variant=variant),
    )
    report_path = write_json(run_context.reports_dir / "benchmark_report.json", report)
    log_metrics(
        run_context,
        {
            "accuracy": report["overall"]["accuracy"],
            "far": report["overall"]["far"],
            "frr": report["overall"]["frr"],
            "auroc": report["overall"].get("auroc"),
        },
    )
    return StageResult("benchmark", variant, run_context.run_dir, run_context.metrics_path, report_path=report_path)


def run_calibration_stage(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    run_root: Path | str = "artifacts",
) -> StageResult:
    distributed_context = DistributedContext(False, 0, 1, 0, "cpu", "gloo", "single", False)
    run_context = _prepare_run(
        manifest_path,
        index_path,
        stage="calibrate",
        variant=variant,
        run_root=run_root,
        payload={},
        context=distributed_context,
    )
    predictions = predict_dataset(manifest_path, index_path=index_path, variant=variant)
    predictions_path = save_predictions(predictions, run_context.reports_dir / "predictions.jsonl")
    report = build_calibration_report(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        index_path=index_path,
        output_path=run_context.reports_dir / "calibration_report.json",
    )
    log_metrics(
        run_context,
        {
            "num_records": int(report["num_records"]),
            "num_stations": int(len(report["stations"])),
        },
    )
    return StageResult(
        "calibrate",
        variant,
        run_context.run_dir,
        run_context.metrics_path,
        report_path=run_context.reports_dir / "calibration_report.json",
        extra_paths={"predictions_path": str(predictions_path)},
    )
