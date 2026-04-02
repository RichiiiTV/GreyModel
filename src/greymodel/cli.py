from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .api import BaseModel, LiteModel
from .data import (
    build_dataset_manifest,
    build_dataset_splits,
    build_huggingface_dataset_manifest,
    build_hard_negative_subset,
    load_dataset_index,
    register_synthetic_recipe,
    validate_dataset_manifest,
)
from .evaluation import benchmark_manifest, build_calibration_report
from .graphing import export_model_graph
from .hf_backends import build_huggingface_model_backend
from .model_profiles import (
    ModelProfile,
    delete_model_profile,
    ensure_default_model_profiles,
    list_model_profiles,
    load_model_profile,
    model_profile_registry_dir,
    register_model_profile,
    save_model_profile,
)
from .prediction import run_batch_prediction_stage
from .pretrain_registry import get_pretrain_dataset_preset, list_pretrain_dataset_presets
from .recovery import ensure_failure_bundle
from .registry import compare_run_reports
from .runners import (
    run_benchmark_stage,
    run_calibration_stage,
    run_domain_adaptation_stage,
    run_explain_audit_stage,
    run_explain_sample_stage,
    run_finetune_stage,
    run_prediction_stage,
    run_pretraining_stage,
    run_resume_stage,
)
from .training import TrainingConfig
from .ui import launch_streamlit_ui
from .utils import read_json, write_json


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
    parser.add_argument("--distributed-strategy", choices=("auto", "fsdp", "ddp"), default="fsdp")
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
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--memory-report", action="store_true")
    parser.add_argument("--no-memory-report", action="store_true")
    parser.add_argument("--pretrain-crop-size", type=int, default=512)
    parser.add_argument("--pretrain-num-crops", type=int, default=1)
    parser.add_argument("--pretrain-crop-scales", nargs="+", type=float, default=(0.75, 1.0, 1.25))
    parser.add_argument("--pretrain-min-valid-fraction", type=float, default=0.2)
    parser.add_argument("--max-global-feature-grid", type=int, default=16)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model-registry-root", default="artifacts/model_profiles")
    if include_checkpoint:
        parser.add_argument("--checkpoint", required=True)


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    station_balanced_sampling = True
    if getattr(args, "no_station_balanced_sampling", False):
        station_balanced_sampling = False
    elif getattr(args, "station_balanced_sampling", False):
        station_balanced_sampling = True
    activation_checkpointing = True
    if getattr(args, "no_activation_checkpointing", False):
        activation_checkpointing = False
    elif getattr(args, "activation_checkpointing", False):
        activation_checkpointing = True
    memory_report = True
    if getattr(args, "no_memory_report", False):
        memory_report = False
    elif getattr(args, "memory_report", False):
        memory_report = True
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
        distributed_strategy=args.distributed_strategy,
        activation_checkpointing=activation_checkpointing,
        memory_report=memory_report,
        seed=args.seed,
        log_every_n_steps=args.log_every_n_steps,
        val_every_n_steps=args.val_every_n_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        keep_last_k_checkpoints=args.keep_last_k_checkpoints,
        resume_from=args.resume_from,
        station_balanced_sampling=station_balanced_sampling,
        show_progress=not bool(getattr(args, "no_progress", False)),
        pretrain_crop_size=args.pretrain_crop_size,
        pretrain_num_crops=args.pretrain_num_crops,
        pretrain_crop_scales=tuple(args.pretrain_crop_scales),
        pretrain_min_valid_fraction=args.pretrain_min_valid_fraction,
        max_global_feature_grid=args.max_global_feature_grid,
        channels_last=bool(args.channels_last),
        ema_decay=args.ema_decay,
        compile_model=bool(args.compile_model),
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )


def _profile_registry_root_from_args(args: argparse.Namespace) -> Path:
    return model_profile_registry_dir(getattr(args, "model_registry_root", "artifacts/model_profiles"))


def _load_profile_for_args(args: argparse.Namespace) -> ModelProfile:
    profile_reference = getattr(args, "model_profile", None)
    if profile_reference in (None, ""):
        raise ValueError("A model profile is required for this command.")
    return load_model_profile(profile_reference, registry_root=_profile_registry_root_from_args(args))


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
    dataset_build_hf.add_argument("--dataset-name", default=None)
    dataset_build_hf.add_argument("--dataset-preset", choices=tuple(sorted(list_pretrain_dataset_presets().keys())), default=None)
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
    dataset_build_hf.add_argument("--no-shape-bucketed-stations", action="store_true")
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

    dataset_ontology = dataset_sub.add_parser("ontology", help="Inspect the resolved ontology for a manifest or dataset index.")
    dataset_ontology.add_argument("--manifest", default=None)
    dataset_ontology.add_argument("--index", default=None)
    dataset_ontology.set_defaults(func=_cmd_dataset_ontology)

    models = subparsers.add_parser("models", help="Model registry and backend profiles.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="List registered model profiles.")
    models_list.add_argument("--registry-root", default="artifacts/model_profiles")
    models_list.set_defaults(func=_cmd_models_list)

    models_show = models_sub.add_parser("show", help="Inspect a model profile.")
    models_show.add_argument("profile")
    models_show.add_argument("--registry-root", default="artifacts/model_profiles")
    models_show.set_defaults(func=_cmd_models_show)

    models_register = models_sub.add_parser("register", help="Register or update a model profile.")
    models_register.add_argument("profile_id")
    models_register.add_argument("--backend-family", choices=("native", "huggingface"), default="huggingface")
    models_register.add_argument("--task-type", choices=("native", "classification", "detection", "segmentation"), default="classification")
    models_register.add_argument("--model-id", default=None)
    models_register.add_argument("--local-path", default=None)
    models_register.add_argument("--revision", default=None)
    models_register.add_argument("--cache-dir", default=None)
    models_register.add_argument("--native-variant", choices=("fast", "base", "lite"), default="base")
    models_register.add_argument("--runtime-engine", default="pytorch")
    models_register.add_argument("--latency-target-ms", type=float, default=None)
    models_register.add_argument("--reject-threshold", type=float, default=0.5)
    models_register.add_argument("--uncertainty-low", type=float, default=0.35)
    models_register.add_argument("--uncertainty-high", type=float, default=0.65)
    models_register.add_argument("--label-mapping-json", default="{}")
    models_register.add_argument("--defect-family-mapping-json", default="{}")
    models_register.add_argument("--good-labels", nargs="+", default=("good", "ok", "pass"))
    models_register.add_argument("--bad-labels", nargs="+", default=("bad", "reject", "ng"))
    models_register.add_argument("--grayscale-mode", default="replicate_rgb")
    models_register.add_argument("--evidence-policy", default="bad")
    models_register.add_argument("--metadata-json", default="{}")
    models_register.add_argument("--registry-root", default="artifacts/model_profiles")
    models_register.set_defaults(func=_cmd_models_register)

    models_delete = models_sub.add_parser("delete", help="Delete a registered model profile.")
    models_delete.add_argument("profile")
    models_delete.add_argument("--registry-root", default="artifacts/model_profiles")
    models_delete.set_defaults(func=_cmd_models_delete)

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
    eval_benchmark.add_argument("--model-profile", default=None)
    eval_benchmark.add_argument("--model-registry-root", default="artifacts/model_profiles")
    eval_benchmark.set_defaults(func=_cmd_eval_benchmark)

    eval_threshold = eval_sub.add_parser("threshold-sweep")
    eval_threshold.add_argument("--manifest", required=True)
    eval_threshold.add_argument("--index", default=None)
    eval_threshold.add_argument("--variant", choices=("base", "lite"), default="base")
    eval_threshold.add_argument("--output-path", default=None)
    eval_threshold.add_argument("--model-profile", default=None)
    eval_threshold.add_argument("--model-registry-root", default="artifacts/model_profiles")
    eval_threshold.set_defaults(func=_cmd_eval_threshold_sweep)

    eval_calibration = eval_sub.add_parser("calibration")
    eval_calibration.add_argument("--manifest", required=True)
    eval_calibration.add_argument("--index", default=None)
    eval_calibration.add_argument("--predictions-path", default=None)
    eval_calibration.add_argument("--output-path", default=None)
    eval_calibration.add_argument("--model-profile", default=None)
    eval_calibration.add_argument("--model-registry-root", default="artifacts/model_profiles")
    eval_calibration.set_defaults(func=_cmd_eval_calibration)

    eval_compare = eval_sub.add_parser("compare")
    eval_compare.add_argument("--left-report", required=True)
    eval_compare.add_argument("--right-report", required=True)
    eval_compare.add_argument("--output-path", default=None)
    eval_compare.set_defaults(func=_cmd_eval_compare)

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
    explain_sample.add_argument("--output-dir", default=str(Path("artifacts") / "sample_bundle"))
    explain_sample.add_argument("--model-profile", default=None)
    explain_sample.add_argument("--model-registry-root", default="artifacts/model_profiles")
    explain_sample.set_defaults(func=_cmd_explain_sample)

    explain_audit = explain_sub.add_parser("audit")
    explain_audit.add_argument("--manifest", required=True)
    explain_audit.add_argument("--index", default=None)
    explain_audit.add_argument("--variant", choices=("base", "lite"), default="base")
    explain_audit.add_argument("--output-dir", default=str(Path("artifacts") / "audit"))
    explain_audit.add_argument("--limit", type=int, default=5)
    explain_audit.add_argument("--model-profile", default=None)
    explain_audit.add_argument("--model-registry-root", default="artifacts/model_profiles")
    explain_audit.set_defaults(func=_cmd_explain_audit)

    predict = subparsers.add_parser("predict", help="Batch hierarchical prediction over a manifest or image folder.")
    predict_inputs = predict.add_mutually_exclusive_group(required=True)
    predict_inputs.add_argument("--manifest", default=None)
    predict_inputs.add_argument("--input-dir", default=None)
    predict.add_argument("--index", default=None)
    predict.add_argument("--station-config", default=None)
    predict.add_argument("--variant", choices=("base", "lite"), default="base")
    predict.add_argument("--run-root", default="artifacts")
    predict.add_argument("--evidence-policy", choices=("none", "bad", "all"), default="bad")
    predict.add_argument("--station-id", default="station-predict")
    predict.add_argument("--product-family", default="unknown")
    predict.add_argument("--geometry-mode", choices=("auto", "rect", "square"), default="auto")
    predict.add_argument("--defect-family", action="append", dest="defect_families", default=None)
    predict.add_argument("--model-profile", default=None)
    predict.add_argument("--model-registry-root", default="artifacts/model_profiles")
    predict.set_defaults(func=_cmd_predict)

    ui = subparsers.add_parser("ui", help="Launch the local Streamlit framework UI.")
    ui.add_argument("--run-root", default="artifacts")
    ui.add_argument("--data-root", default="data")
    ui.add_argument("--dataset-root", dest="data_root")
    ui.add_argument("--workspace-path", default=None)
    ui.add_argument("--bind-address", default=None)
    ui.add_argument("--host", dest="bind_address", default=None)
    ui.add_argument("--bind-port", type=int, default=None)
    ui.add_argument("--port", dest="bind_port", type=int, default=None)
    ui.add_argument("--proxy-mode", choices=("auto", "off", "jupyter_port", "jupyter_service"), default="auto")
    ui.add_argument("--public-base-url", default=None)
    ui.add_argument("--base-url-path", default=None)
    ui.add_argument("--browser-server-address", default=None)
    ui.add_argument("--browser-server-port", type=int, default=None)
    ui.add_argument("--print-url", action="store_true")
    ui.add_argument("--browser", action="store_true")
    ui.add_argument("--default-execution-backend", choices=("local", "slurm"), default="local")
    ui.add_argument("--slurm-cpus", type=int, default=8)
    ui.add_argument("--slurm-mem", default="50G")
    ui.add_argument("--slurm-gres", default="gpu:8")
    ui.add_argument("--slurm-partition", default="")
    ui.add_argument("--slurm-queue", default="")
    ui.add_argument("--slurm-nproc-per-node", type=int, default=8)
    ui.add_argument("--slurm-python", default=None)
    ui.add_argument("--dry-run", action="store_true")
    ui.set_defaults(func=_cmd_ui)
    return parser


def _cmd_dataset_build(args: argparse.Namespace):
    return build_dataset_manifest(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        source_dataset=args.source_dataset,
        seed=args.seed,
    )


def _cmd_dataset_build_hf(args: argparse.Namespace):
    preset = get_pretrain_dataset_preset(args.dataset_preset) if args.dataset_preset else None
    if preset is None and not args.dataset_name:
        raise ValueError("dataset build-hf requires --dataset-name or --dataset-preset.")
    allow_rgb_conversion = bool(args.allow_rgb_conversion)
    if preset is not None:
        allow_rgb_conversion = allow_rgb_conversion or bool(preset.allow_rgb_conversion)
    return build_huggingface_dataset_manifest(
        dataset_name=preset.dataset_name if preset is not None else args.dataset_name,
        output_dir=args.output_dir,
        config_name=preset.config_name if preset is not None and args.config_name is None else args.config_name,
        split_names=args.splits,
        data_dir=preset.data_dir if preset is not None and args.data_dir is None else args.data_dir,
        image_column=args.image_column,
        station_id=args.station_id,
        product_family=args.product_family,
        geometry_mode=args.geometry_mode,
        source_dataset=(preset.source_dataset if preset is not None and args.source_dataset is None else args.source_dataset),
        cache_dir=args.cache_dir,
        max_records=args.max_records,
        accept_reject_column=args.accept_reject_column,
        defect_tags_column=args.defect_tags_column,
        station_id_column=args.station_id_column,
        product_family_column=args.product_family_column,
        geometry_mode_column=args.geometry_mode_column,
        metadata_columns=tuple(args.metadata_columns or ()),
        strict_grayscale=not allow_rgb_conversion,
        token=args.token,
        local_files_only=bool(args.local_files_only),
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        shape_bucketed_stations=not bool(args.no_shape_bucketed_stations),
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
    payload = json.loads(args.payload_json)
    return register_synthetic_recipe(args.index, args.recipe_name, payload)


def _cmd_dataset_ontology(args: argparse.Namespace):
    if args.index:
        index = load_dataset_index(args.index)
    elif args.manifest:
        candidate = Path(args.manifest).with_name("dataset_index.json")
        if not candidate.exists():
            raise ValueError("No dataset_index.json was found next to %s." % args.manifest)
        index = load_dataset_index(candidate)
    else:
        raise ValueError("dataset ontology requires --manifest or --index.")
    return read_json(Path(index.ontology_path))


def _cmd_models_list(args: argparse.Namespace):
    profiles = ensure_default_model_profiles(args.registry_root)
    return [profile.to_dict() for profile in profiles]


def _cmd_models_show(args: argparse.Namespace):
    profile = load_model_profile(args.profile, registry_root=args.registry_root)
    return profile.to_dict()


def _cmd_models_register(args: argparse.Namespace):
    label_mapping = json.loads(args.label_mapping_json)
    defect_family_mapping = json.loads(args.defect_family_mapping_json)
    metadata = json.loads(args.metadata_json)
    if args.backend_family == "native":
        metadata = {**dict(metadata), "variant": args.native_variant}
    if args.latency_target_ms is not None:
        metadata = {**dict(metadata), "latency_target_ms": float(args.latency_target_ms)}
    profile = ModelProfile(
        profile_id=args.profile_id,
        backend_family=args.backend_family,
        task_type=args.task_type,
        model_id=args.model_id,
        local_path=args.local_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        runtime_engine=args.runtime_engine,
        reject_threshold=args.reject_threshold,
        uncertainty_low=args.uncertainty_low,
        uncertainty_high=args.uncertainty_high,
        label_mapping=label_mapping,
        defect_family_mapping=defect_family_mapping,
        good_labels=tuple(args.good_labels),
        bad_labels=tuple(args.bad_labels),
        grayscale_mode=args.grayscale_mode,
        evidence_policy=args.evidence_policy,
        metadata=metadata,
    )
    register_model_profile(profile, args.registry_root)
    return profile.to_dict()


def _cmd_models_delete(args: argparse.Namespace):
    deleted = delete_model_profile(args.registry_root, args.profile)
    return {"profile": args.profile, "deleted": deleted, "registry_root": str(args.registry_root)}


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
        return benchmark_manifest(
            args.manifest,
            index_path=args.index,
            variant=args.variant,
            output_path=args.output_path,
            model_profile=args.model_profile,
            model_registry_root=args.model_registry_root,
        )
    result = run_benchmark_stage(
        args.manifest,
        index_path=args.index,
        variant=args.variant,
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )
    return {"report_path": str(result.report_path), "run_dir": str(result.run_dir)}


def _cmd_eval_threshold_sweep(args: argparse.Namespace):
    report = benchmark_manifest(
        args.manifest,
        index_path=args.index,
        variant=args.variant,
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )
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
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )


def _cmd_eval_compare(args: argparse.Namespace):
    payload = compare_run_reports(args.left_report, args.right_report)
    if args.output_path:
        write_json(Path(args.output_path), payload)
    return payload


def _cmd_explain_graph(args: argparse.Namespace):
    model = _variant_model(args.variant, num_defect_families=args.num_defect_families)
    backend_model = getattr(model.backend, "model", None)
    if backend_model is None:
        raise RuntimeError("Graph export requires the PyTorch backend.")
    return export_model_graph(backend_model, args.output_dir, image_shape=(args.height, args.width))


def _cmd_explain_sample(args: argparse.Namespace):
    result = run_explain_sample_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        sample_id=args.sample_id,
        variant=args.variant,
        run_root=args.output_dir,
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )
    payload = {"report_path": str(result.report_path), "run_dir": str(result.run_dir)}
    payload.update(dict(result.extra_paths or {}))
    return payload


def _cmd_explain_audit(args: argparse.Namespace):
    result = run_explain_audit_stage(
        manifest_path=args.manifest,
        index_path=args.index,
        variant=args.variant,
        run_root=args.output_dir,
        limit=args.limit,
        model_profile=args.model_profile,
        model_registry_root=args.model_registry_root,
    )
    return {"report_path": str(result.report_path), "run_dir": str(result.run_dir)}


def _cmd_predict(args: argparse.Namespace):
    if args.input_dir:
        result = run_batch_prediction_stage(
            input_dir=args.input_dir,
            index_path=args.index,
            station_config_path=args.station_config,
            variant=args.variant,
            run_root=args.run_root,
            station_id=args.station_id,
            product_family=args.product_family,
            geometry_mode=args.geometry_mode,
            defect_families=tuple(args.defect_families or ()),
            evidence_policy=args.evidence_policy,
            model_profile=args.model_profile,
            model_registry_root=args.model_registry_root,
        )
    else:
        result = run_prediction_stage(
            manifest_path=args.manifest,
            index_path=args.index,
            variant=args.variant,
            run_root=args.run_root,
            evidence_policy=args.evidence_policy,
            model_profile=args.model_profile,
            model_registry_root=args.model_registry_root,
        )
    payload = {"run_dir": str(result.run_dir)}
    if result.report_path is not None:
        payload["report_path"] = str(result.report_path)
    payload.update(dict(result.extra_paths or {}))
    return payload


def _cmd_ui(args: argparse.Namespace):
    return launch_streamlit_ui(
        run_root=args.run_root,
        data_root=args.data_root,
        workspace_path=args.workspace_path,
        bind_address=args.bind_address,
        bind_port=args.bind_port,
        headless=not bool(args.browser),
        dry_run=bool(args.dry_run),
        proxy_mode=args.proxy_mode,
        public_base_url=args.public_base_url,
        base_url_path=args.base_url_path,
        browser_server_address=args.browser_server_address,
        browser_server_port=args.browser_server_port,
        print_url=bool(args.print_url),
        default_execution_backend=args.default_execution_backend,
        slurm_cpus=args.slurm_cpus,
        slurm_mem=args.slurm_mem,
        slurm_gres=args.slurm_gres,
        slurm_partition=args.slurm_partition,
        slurm_queue=args.slurm_queue,
        slurm_nproc_per_node=args.slurm_nproc_per_node,
        slurm_python=args.slurm_python,
    )


def _failure_context_from_args(args: argparse.Namespace) -> Mapping[str, object] | None:
    command = getattr(args, "command", None)
    if command is None:
        return None
    if command == "train":
        return {
            "run_root": getattr(args, "run_root", "artifacts"),
            "stage": str(getattr(args, "train_command", "train")).replace("-", "_"),
            "variant": getattr(args, "variant", "base"),
            "manifest_path": getattr(args, "manifest", None),
            "index_path": getattr(args, "index", None),
        }
    if command == "eval":
        return {
            "run_root": "artifacts",
            "stage": "eval_%s" % str(getattr(args, "eval_command", "eval")).replace("-", "_"),
            "variant": getattr(args, "variant", "base"),
            "manifest_path": getattr(args, "manifest", None),
            "index_path": getattr(args, "index", None),
        }
    if command == "explain":
        return {
            "run_root": getattr(args, "output_dir", "artifacts"),
            "stage": "explain_%s" % str(getattr(args, "explain_command", "explain")).replace("-", "_"),
            "variant": getattr(args, "variant", "base"),
            "manifest_path": getattr(args, "manifest", None),
            "index_path": getattr(args, "index", None),
        }
    if command == "predict":
        return {
            "run_root": getattr(args, "run_root", "artifacts"),
            "stage": "predict",
            "variant": getattr(args, "variant", "base"),
            "manifest_path": getattr(args, "manifest", None),
            "index_path": getattr(args, "index", None),
        }
    if command == "ui":
        return {
            "run_root": getattr(args, "run_root", "artifacts"),
            "stage": "ui",
            "variant": "local",
            "manifest_path": None,
            "index_path": None,
        }
    return {
        "run_root": "artifacts",
        "stage": "%s_%s" % (command, getattr(args, "%s_command" % command, command)),
        "variant": "local",
        "manifest_path": getattr(args, "manifest", None),
        "index_path": getattr(args, "index", None),
    }


def cli_main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except BaseException as exc:
        if not getattr(exc, "_greymodel_failure_written", False):
            failure_context = _failure_context_from_args(args)
            if failure_context is not None:
                try:
                    ensure_failure_bundle(
                        run_root=failure_context["run_root"],
                        stage=str(failure_context["stage"]),
                        variant=str(failure_context["variant"]),
                        exc=exc,
                        manifest_path=failure_context.get("manifest_path"),
                        index_path=failure_context.get("index_path"),
                        metadata={"command": getattr(args, "command", "unknown")},
                    )
                except Exception:
                    pass
        raise
