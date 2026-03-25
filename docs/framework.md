# GreyModel Framework Guide

## Goal

The repo is a finetuning framework around the grayscale inspection backbone, not just a model definition. It covers:

- dataset ingestion
- staged training
- evaluation
- calibration
- explainability

## Dataset Model

### Folder-First Import

External production data still enters as folders. The framework immediately normalizes it into:

- `manifest.jsonl`
- `splits.json`
- `ontology.json`
- `hard_negatives.jsonl`
- `dataset_index.json`

This keeps the workflow reproducible while matching how inspection images are usually delivered.

### Public Hugging Face Import

Public pretraining data uses the same internal contract.

`greymodel dataset build-hf` now supports curated presets through `--dataset-preset`, for example:

- `ds_dagm`
- `defect_spectrum_full`
- `mvtec_ad_gray`

Key behavior:

- grayscale is enforced by default
- optional RGB-to-grayscale conversion is explicit
- mixed resolutions are shape-bucketed into pseudo-stations
- imported records are stored locally and trained through the same manifest interface as production data

## Training Workflow

### Smoke Runs Versus Real Runs

Smoke runs are still useful for verifying manifests, startup, and graph export. Real jobs are:

- epoch-based
- checkpointed
- resumable
- metrics-driven

### Distributed Strategy

The framework is now `FSDP`-first for large pretraining jobs.

- `--distributed-strategy fsdp` is the default multi-GPU path
- `--distributed-strategy ddp` remains available for finetune/debug use
- activation checkpointing, channels-last, and memory telemetry are exposed in the public CLI

### Patch-Based Public Pretraining

Public pretraining no longer runs full imported public images through the backbone.

Instead:

1. import the dataset into a manifest bundle
2. sample bounded square crops from the imported full images at train time
3. run masked reconstruction on those crops

This is the main safeguard against the oversized public-image VRAM failures that the older flow triggered.

Important knobs:

- `--pretrain-crop-size`
- `--pretrain-num-crops`
- `--pretrain-crop-scales`
- `--max-global-feature-grid`
- `--memory-report`

### Domain Adaptation

Domain adaptation still runs on unlabeled production frames and keeps the shared manifest/index workflow.

### Finetuning

Finetuning still uses the full production station canvas and the existing reject/defect/heatmap contract. Station-balanced sampling remains available.

## Run Artifacts

Training runs write:

- `metrics.jsonl`
- `epoch_metrics.jsonl`
- `config_snapshot.json`
- `manifest_snapshot.json`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `reports/<stage>_report.json`
- `reports/training_summary.json`

When `--memory-report` is enabled, step and epoch artifacts also include CUDA allocation and reserve metrics, plus per-rank summary memory when available.

## Evaluation

Evaluation remains framework-level:

- FAR
- FRR
- AUROC
- PR behavior
- per-defect-family slices
- tiny / medium / global defect slices
- per-station behavior

## Explainability

The explainability stack still includes:

- `torch.fx` architecture graph export
- Mermaid graph output
- sample-level attribution bundles
- heatmaps and top tiles

The graph exporter was updated to follow the current CNN-heavy model instead of the earlier transformer-specific internals.

## CLI Shape

The CLI groups remain:

- `dataset`
- `train`
- `eval`
- `explain`

The public shape did not change, but `train pretrain` now accepts the new patch-pretraining and distributed-runtime flags directly.
