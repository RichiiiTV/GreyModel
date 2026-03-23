# GreyModel Framework Guide

## Goal

This repo is more than a backbone implementation. The framework should support dataset curation, staged training, evaluation, calibration, and explainability for grayscale syringe and vial inspection.

## Dataset Model

### Folder-First Import

External data should be imported from folders, because that matches how inspection images are usually handed off in practice.

Internally, the framework should normalize folders into versioned manifests so the training and evaluation runs are reproducible.

### Internal Artifacts

The canonical framework artifacts should include:

- `manifest.jsonl`
- `splits.json`
- `ontology.json`
- `hard_negatives.jsonl`

### Required Records

Each record should carry:

- image path
- station ID
- product family
- geometry mode
- image-level label
- optional defect tags
- optional boxes or masks
- split assignment
- capture metadata

### Why This Matters

Without manifests, the framework cannot reliably do:

- leakage-safe splits by station, day, batch, and camera
- hard-negative harvesting
- review queues
- repeatable finetuning experiments

## Training Workflow

### Pretrain

Use large unlabeled grayscale imagery and masked-image pretraining to learn generic line and part structure.

### Domain Adapt

Adapt on unlabeled production frames from the target line before supervised finetuning.

### Finetune

Use image-level labels as the main supervision signal.

- Add boxes or masks only for curated hard cases.
- Keep the sampling station-aware.
- Use hard clean negatives and recent false rejects/false accepts.

### Calibrate

Fit station-specific thresholds and, if needed, lightweight adapters.

## Evaluation

Evaluation should be framework-level, not just loss reporting.

- Report FAR, FRR, AUROC, and PR curves.
- Slice metrics by tiny, medium, and global defects.
- Break metrics out by station and product family.
- Include hard clean negatives and small-particle challenge sets.

## Explainability

The explainability stack should include both architecture-level and sample-level artifacts.

- Architecture graph from `torch.fx`.
- Mermaid graph output as the baseline artifact.
- Optional SVG or DOT if Graphviz is installed.
- Per-sample saliency or integrated gradients.
- Tile overlays and heatmaps for operator review.

## CLI Shape

The framework should expose four command groups:

- `dataset`: scan, validate, split, and mine hard negatives.
- `train`: pretrain, domain adapt, finetune, resume, and calibrate.
- `eval`: benchmark and threshold sweep.
- `explain`: graph export and per-sample audit bundles.

## Practical Constraints

- Keep `Base` and `Lite` identical at the I/O level.
- Preserve `5x5` defect sensitivity.
- Avoid aspect-ratio distortion.
- Keep grayscale as first-class input, not RGB conversion.
- Keep all framework outputs easy to audit and version.
