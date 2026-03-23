from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .api import BaseModel, LiteModel
from .data import (
    build_dataset_manifest,
    build_dataset_splits,
    build_huggingface_dataset_manifest,
    build_hard_negative_subset,
    load_dataset_index,
    load_dataset_manifest,
    load_station_configs_from_index,
    register_synthetic_recipe,
    station_config_for_record,
    validate_dataset_manifest,
)
from .evaluation import benchmark_manifest, build_calibration_report
from .explainability import build_audit_report, build_explanation_bundle
from .graphing import export_model_graph
from .runners import (
    run_benchmark_stage,
    run_calibration_stage,
    run_domain_adaptation_stage,
    run_finetune_stage,
    run_pretraining_stage,
    run_resume_stage,
)
from .training import TrainingConfig
from .types import ModelInput
from .utils import ensure_dir, load_uint8_grayscale, write_json


def _variant_model(variant: str, num_defect_families: int):
    if variant == "lite":
        return LiteModel(num_defect_families=num_defect_families)
    return BaseModel(num_defect_families=num_defect_families)


def _add_training_arguments(parser: argparse.ArgumentParser, include_checkpoint: bool = False) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", default=None)
    parser.add_argument("--variant", choices=("base", "lite"), default="base")
    parser.add_argument("--run-root", default="artifacts")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--distributed-backend", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--val-every-n-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=0)
    parser.add_argument("--keep-last-k-checkpoints", type=int, default=2)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--defect-positive-weight", type=float, default=1.0)
    parser.add_argument("--reject-positive-weight", type=float, default=1.0)
    parser.add_argument("--station-balanced-sampling", action="store_true")
    parser.add_argument("--no-station-balanced-sampling", action="store_true")
    if include_checkpoint:
        parser.add_argument("--checkpoint", required=True)


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    station_balanced_sampling = True
    if getattr(args, "no_station_balanced_sampling", False):
        station_balanced_sampling = False
    elif getattr(args, "station_balanced_sampling", False):
        station_balanced_sampling = True
    return TrainingConfig(
        defect_positive_weight=args.defect_positive_weight,
        reject_positive_weight=args.reject_positive_weight,
        focal_gamma=args.focal_gamma,
        mask_ratio=args.mask_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        global_batch_size=args.global_batch_size,
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        precision=args.precision,
        distributed_backend=args.distributed_backend,
        seed=args.seed,
        log_every_n_steps=args.log_every_n_steps,
        val_every_n_steps=args.val_every_n_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        keep_last_k_checkpoints=args.keep_last_k_checkpoints,
        resume_from=args.resume_from,
        station_balanced_sampling=station_balanced_sampling,
        show_progress=not bool(getattr(args, "no_progress", False)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="greymodel", description="GreyModel finetuning framework CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="Dataset ingest, validation, splits, and hard negatives.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_sub.add_parser("build", help="Scan a folder tree and build a canonical manifest bundle.")
    dataset_build.add_argument("root_dir")
    dataset_build.add_argument("--output-dir", default=None)
    dataset_build.add_argument("--source-dataset", default="folder_import")
    dataset_build.add_argument("--seed", type=int, default=17)
    dataset_build.set_defaults(func=_cmd_dataset_build)

    dataset_build_hf = dataset_sub.add_parser(
        "build-hf",
        help="Materialize a public Hugging Face image dataset into a local grayscale manifest bundle.",
    )
    dataset_build_hf.add_argument("--dataset-name", required=True)
    dataset_build_hf.add_argument("--output-dir", required=True)
    dataset_build_hf.add_argument("--config-name", default=None)
    dataset_build_hf.add_argument("--data-dir", default=None)
    dataset_build_hf.add_argument("--split", action="append", dest="splits", default=None)
    dataset_build_hf.add_argument("--image-column", default="image")
    dataset_build_hf.add_argument("--station-id", default="hf-public")
    dataset_build_hf.add_argument("--product-family", default="unknown")
    dataset_build_hf.add_argument("--geometry-mode", choices=("auto", "rect", "square"), default="auto")
    dataset_build_hf.add_argument("--source-dataset", default=None)
    dataset_build_hf.add_argument("--cache-dir", default=None)
    dataset_build_hf.add_argument("--max-records", type=int, default=None)
    dataset_build_hf.add_argument("--accept-reject-column", default=None)
    dataset_build_hf.add_argument("--defect-tags-column", default=None)
    dataset_build_hf.add_argument("--station-id-column", default=None)
    dataset_build_hf.add_argument("--product-family-column", default=None)
    dataset_build_hf.add_argument("--geometry-mode-column", default=None)
    dataset_build_hf.add_argument("--metadata-column", action="append", dest="metadata_columns", default=None)
    dataset_build_hf.add_argument("--allow-rgb-conversion", action="store_true")
    dataset_build_hf.add_argument("--token", default=None)
    dataset_build_hf.add_argument("--local-files-only", action="store_true")
    dataset_build_hf.add_argument("--max-retries", type=int, default=4)
    dataset_build_hf.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    dataset_build_hf.set_defaults(func=_cmd_dataset_build_hf)

    dataset_validate = dataset_sub.add_parser("validate", help="Validate a manifest.")
    dataset_validate.add_argument("manifest")
    dataset_validate.set_defaults(func=_cmd_dataset_validate)

    dataset_split = dataset_sub.add_parser("split", help="Regenerate leakage-safe splits.")
    dataset_split.add_argument("manifest")
    dataset_split.add_argument("--output-path", default=None)
    dataset_split.add_argument("--seed", type=int, default=17)
    dataset_split.set_defaults(func=_cmd_dataset_split)

    dataset_hn = dataset_sub.add_parser("hard-negatives", help="Build a reusable hard-negative subset.")
    dataset_hn.add_argument("manifest")
    dataset_hn.add_argument("--output-path", default=None)
    dataset_hn.add_argument("--predictions-path", default=None)
    dataset_hn.add_argument("--score-threshold", type=float, default=0.5)
    dataset_hn.set_defaults(func=_cmd_dataset_hard_negatives)

    dataset_recipe = dataset_sub.add_parser("register-recipe", help="Register a synthetic-data recipe in the dataset index.")
    dataset_recipe.add_argument("index")
    dataset_recipe.add_argument("recipe_name")
    dataset_recipe.add_argument("--payload-json", default="{}")
    dataset_recipe.set_defaults(func=_cmd_dataset_register_recipe)

    train = subparsers.add_parser("train", help="Staged training and calibration.")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    for name, handler in (
        ("pretrain", _cmd_train_pretrain),
        ("domain-adapt", _cmd_train_domain_adapt),
        ("finetune", _cmd_train_finetune),
        ("resume", _cmd_train_resume),
        ("calibrate", _cmd_train_calibrate),
    ):
        sub = train_sub.add_parser(name)
        _add_training_arguments(sub, include_checkpoint=(name == "resume"))
        sub.set_defaults(func=handler)

    evaluate = subparsers.add_parser("eval", help="Benchmark and calibration reports.")
    eval_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_benchmark = eval_sub.add_parser("benchmark")
    eval_benchmark.add_argument("--manifest", required=True)
    eval_benchmark.add_argument("--index", default=None)
    eval_benchmark.add_argument("--variant", choices=("base", "lite"), default="base")
    eval_benchmark.add_argument("--output-path", default=None)
    eval_benchmark.set_defaults(func=_cmd_eval_benchmark)

    eval_threshold = eval_sub.add_parser("threshold-sweep")
    eval_threshold.add_argument("--manifest", required=True)
    eval_threshold.add_argument("--index", default=None)
    eval_threshold.add_argument("--variant", choices=("base", "lite"), default="base")
    eval_threshold.add_argument("--output-path", default=None)
    eval_threshold.set_defaults(func=_cmd_eval_threshold_sweep)

    eval_calibration = eval_sub.add_parser("calibration")
    eval_calibration.add_argument("--manifest", required=True)
    eval_calibration.add_argument("--index", default=None)
    eval_calibration.add_argument("--predictions-path", default=None)
    eval_calibration.add_argument("--output-path", default=None)
    eval_calibration.set_defaults(func=_cmd_eval_calibration)

    explain = subparsers.add_parser("explain", help="Architecture graph and sample audit bundles.")
    explain_sub = explain.add_subparsers(dest="explain_command", required=True)

    explain_graph = explain_sub.add_parser("graph")
    explain_graph.add_argument("--variant", choices=("base", "lite"), default="base")
    explain_graph.add_argument("--output-dir", default=str(Path("docs") / "graphs"))
    explain_graph.add_argument("--height", type=int, default=256)
    explain_graph.add_argument("--width", type=int, default=256)
    explain_graph.add_argument("--num-defect-families", type=int, default=4)
    explain_graph.set_defaults(func=_cmd_explain_graph)

    explain_sample = explain_sub.add_parser("sample")
    explain_sample.add_argument("--manifest", required=True)
    explain_sample.add_argument("--index", default=None)
    explain_sample.add_argument("--sample-id", default=None)
    explain_sample.add_argument("--variant", choices=("base", "lite"), default="base")
    explain_sample.add_argument("--output-dir", default=str(Path("artifacts") / "sample_explanations"))
    explain_sample.set_defaults(func=_cmd_explain_sample)

    explain_audit = explain_sub.add_parser("audit")
    explain_audit.add_argument("--manifest", required=True)
    explain_audit.add_argument("--variant", choices=("base", "lite"), default="base")
    explain_audit.add_argument("--output-dir", default=str(Path("artifacts") / "audit"))
    explain_audit.add_argument("--limit", type=int, default=5)
    explain_audit.set_defaults(func=_cmd_explain_audit)
    return parser


def _cmd_dataset_build(args: argparse.Namespace):
    return build_dataset_manifest(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        source_dataset=args.source_dataset,
        seed=args.seed,
    )


def _cmd_dataset_build_hf(args: argparse.Namespace):
    return build_huggingface_dataset_manifest(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        config_name=args.config_name,
        split_names=args.splits,
        data_dir=args.data_dir,
        image_column=args.image_column,
        station_id=args.station_id,
        product_family=args.product_family,
        geometry_mode=args.geometry_mode,
        source_dataset=args.source_dataset,
        cache_dir=args.cache_dir,
        max_records=args.max_records,
        accept_reject_column=args.accept_reject_column,
        defect_tags_column=args.defect_tags_column,
        station_id_column=args.station_id_column,
        product_family_column=args.product_family_column,
        geometry_mode_column=args.geometry_mode_column,
        metadata_columns=tuple(args.metadata_columns or ()),
        strict_grayscale=not bool(args.allow_rgb_conversion),
        token=args.token,
        local_files_only=bool(args.local_files_only),
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


def _cmd_dataset_validate(args: argparse.Namespace):
    return validate_dataset_manifest(args.manifest)


def _cmd_dataset_split(args: argparse.Namespace):
    return build_dataset_splits(args.manifest, output_path=args.output_path, seed=args.seed)


def _cmd_dataset_hard_negatives(args: argparse.Namespace):
    return build_hard_negative_subset(
        args.manifest,
        output_path=args.output_path,
        predictions_path=args.predictions_path,
        score_threshold=args.score_threshold,
    )


def _cmd_dataset_register_recipe(args: argparse.Namespace):
    import json

    payload = json.loads(args.payload_json)
    return register_synthetic_recipe(args.index, args.recipe_name, payload)


def _cmd_train_pretrain(args: argparse.Namespace):
    return run_pretraining_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.run_root,
        split=args.split,
        batch_size=args.batch_size,
        training_config=_training_config_from_args(args),
        checkpoint_path=getattr(args, "checkpoint", None),
        resume_from=args.resume_from,
    )


def _cmd_train_domain_adapt(args: argparse.Namespace):
    return run_domain_adaptation_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.run_root,
        split=args.split,
        batch_size=args.batch_size,
        training_config=_training_config_from_args(args),
        checkpoint_path=getattr(args, "checkpoint", None),
        resume_from=args.resume_from,
    )


def _cmd_train_finetune(args: argparse.Namespace):
    return run_finetune_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.run_root,
        split=args.split,
        batch_size=args.batch_size,
        training_config=_training_config_from_args(args),
        checkpoint_path=getattr(args, "checkpoint", None),
        resume_from=args.resume_from,
    )


def _cmd_train_resume(args: argparse.Namespace):
    return run_resume_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.run_root,
        split=args.split,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        training_config=_training_config_from_args(args),
    )


def _cmd_train_calibrate(args: argparse.Namespace):
    return run_calibration_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.run_root,
    )


def _cmd_eval_benchmark(args: argparse.Namespace):
    if args.output_path:
        return benchmark_manifest(args.manifest, index_path=args.index, variant=args.variant, output_path=args.output_path)
    result = run_benchmark_stage(args.manifest, index_path=args.index, variant=args.variant)
    return {"report_path": str(result.report_path), "run_dir": str(result.run_dir)}


def _cmd_eval_threshold_sweep(args: argparse.Namespace):
    report = benchmark_manifest(args.manifest, index_path=args.index, variant=args.variant)
    payload = report["threshold_sweep"]
    if args.output_path:
        write_json(Path(args.output_path), payload)
    return payload


def _cmd_eval_calibration(args: argparse.Namespace):
    output_path = args.output_path or str(Path("reports") / "calibration_report.json")
    return build_calibration_report(
        manifest_path=args.manifest,
        predictions_path=args.predictions_path,
        index_path=args.index,
        output_path=output_path,
    )


def _cmd_explain_graph(args: argparse.Namespace):
    model = _variant_model(args.variant, num_defect_families=args.num_defect_families)
    backend_model = getattr(model.backend, "model", None)
    if backend_model is None:
        raise RuntimeError("Graph export requires the PyTorch backend.")
    return export_model_graph(backend_model, args.output_dir, image_shape=(args.height, args.width))


def _cmd_explain_sample(args: argparse.Namespace):
    import json

    records = load_dataset_manifest(args.manifest)
    if not records:
        raise ValueError("The manifest is empty.")
    record = next((row for row in records if row.sample_id == args.sample_id), records[0])
    index_path = args.index or str(Path(args.manifest).with_name("dataset_index.json"))
    index = load_dataset_index(index_path)
    station_configs = load_station_configs_from_index(index)
    ontology_path = Path(index.ontology_path)
    defect_count = 4
    if ontology_path.exists():
        defect_count = max(len(json.loads(ontology_path.read_text(encoding="utf-8")).get("defect_tags", ())), 1)
    image = load_uint8_grayscale(Path(record.image_path))
    model_input = ModelInput(
        image_uint8=image,
        station_id=record.station_id,
        geometry_mode=record.geometry_mode,
        metadata=record.capture_metadata,
    )
    station_config = station_config_for_record(record, station_configs)
    model = _variant_model(args.variant, num_defect_families=defect_count)
    sample_dir = ensure_dir(Path(args.output_dir) / record.sample_id.replace("/", "_"))
    return build_explanation_bundle(model, model_input, station_config, sample_dir)


def _cmd_explain_audit(args: argparse.Namespace):
    model_factory = lambda: _variant_model(args.variant, num_defect_families=4)
    return build_audit_report(model_factory, args.manifest, args.output_dir, limit=args.limit)


def cli_main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = args.func(args)
    return result
