# GreyModel Architecture Walkthrough

## Objective

`GreyModel` is designed for grayscale syringe and vial inspection where defect scale spans:

- tiny particles around `5x5`
- medium scratches or streaks
- full-image anomalies

The model keeps one shared backbone family across stations while preserving:

- grayscale `8-bit` input handling
- rectangular and square station support
- aspect-ratio-safe preprocessing
- `Base`/`Lite` output parity

## Input And Preprocessing

Input contract:

- one grayscale `uint8` image
- `station_id`
- `geometry_mode`

Preprocessing:

- preserve aspect ratio
- pad to the station canvas
- emit a valid-pixel mask
- normalize after padding

This prevents geometry distortion and keeps padding from corrupting defect evidence.

## Backbone

The current `GrayInspect-H` implementation is a CNN-heavy hybrid:

- grayscale stem
- ConvNeXt-style pyramid
- BiFPN-style fusion
- bounded coarse context block
- high-resolution local defect branch
- station and geometry conditioning
- fused reject / defect-family / weak-heatmap heads

### Grayscale Stem

The stem uses small-stride convolution so grayscale micro-contrast survives into the feature pyramid.

### Feature Pyramid

The pyramid builds multiscale features at roughly:

- `1/4`
- `1/8`
- `1/16`
- `1/32`

This supports both large structural context and localized fine detail.

### BiFPN-Style Fusion

The fusion stack keeps high-resolution detail alive while still propagating low-resolution semantic context back upward.

### Bounded Coarse Context

Global context is only applied on the coarsest feature map and bounded by `max_global_feature_grid`.

That keeps global reasoning without allowing memory to explode with public pretraining image size.

### Local Branch

The local branch runs on overlapping tiles from the padded full image. This is the main protection for `5x5` defects.

Outputs from the local branch include:

- tile logits
- top-tile evidence
- local weak heatmap

## Conditioning

`station_id` and `geometry_mode` are fed into lightweight conditioning layers so one shared model family can adapt across stations without forking into separate station-specific backbones.

## Heads And Output

The model produces:

- calibrated binary reject score / logit
- defect-family probabilities
- weak heatmap
- top tile evidence

At the framework level this is persisted as a hierarchical prediction record:

- `primary_label`
- `primary_score`
- `top_defect_family`
- `defect_family_probs`
- `evidence`

## Training Stages

### Public Pretraining

Public pretraining is patch-based by default:

1. import public images into a manifest bundle
2. sample bounded grayscale crops at train time
3. run masked reconstruction on those crops

This is the main public-data VRAM safeguard.

### Domain Adaptation

Domain adaptation runs on unlabeled production frames and keeps the same manifest/index workflow.

### Finetuning

Finetuning runs on full production station canvases and optimizes the binary primary decision plus defect-family secondary outputs.

### Calibration

Calibration remains external to the shared backbone and is stored per station.

## Distributed Runtime

The training runtime is:

- `FSDP`-first for large multi-GPU pretraining
- `DDP` fallback for simpler finetune/debug paths
- activation checkpointing
- mixed precision
- channels-last support
- explicit memory telemetry

## Explainability

Explainability includes:

- `torch.fx` graph export
- Mermaid graphs
- sample attribution maps
- weak heatmaps
- top-tile bundles

Artifacts are written under:

- `docs/graphs/`
- run `explanations/`
- run `reports/`

## Recovery Integration

Training, prediction, evaluation, and explainability stages all write run status and failure bundles. This is part of the architecture, not an afterthought, because inspection workflows need usable artifacts even on partial failure.
