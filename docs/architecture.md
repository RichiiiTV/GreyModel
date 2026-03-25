# GreyModel Architecture Walkthrough

## Overview

`GreyModel` is built for grayscale syringe and vial inspection where defect scale ranges from micro-particles to full-frame anomalies. The current `GrayInspect-H` implementation is a CNN-heavy hybrid:

- grayscale stem
- ConvNeXt-style feature pyramid
- BiFPN-style top-down and bottom-up fusion
- one bounded coarse-context block at the lowest-resolution feature map
- a separate local tile branch for tiny defects
- station and geometry conditioning
- fused reject, defect-family, and weak heatmap heads

The inference contract did not change, but the training internals were redesigned for lower VRAM usage and more robust public-data pretraining.

## Preprocessing

Input remains:

- one `uint8` grayscale image
- `station_id`
- `geometry_mode` (`rect` or `square`)

Preprocessing still:

- preserves aspect ratio
- pads to the station canvas
- emits a valid-pixel mask
- normalizes after padding

This keeps geometry intact across rectangular and square stations and prevents padding from leaking into the model.

## Backbone

### Grayscale Stem

The stem uses small-stride convolutions to retain fine grayscale contrast before the pyramid gets deeper. This is the first guardrail for `5x5`-scale defects.

### ConvNeXt-Style Pyramid

After the stem, the backbone builds feature maps at roughly:

- `1/4`
- `1/8`
- `1/16`
- `1/32`

Each stage uses ConvNeXt-style depthwise-heavy blocks, with station/geometry conditioning applied between stages.

### BiFPN-Style Fusion

The pyramid is fused with repeated BiFPN-style blocks. This keeps high-resolution detail alive while still letting low-resolution semantic context propagate back upward.

The weak heatmap head is attached to the high-resolution fused feature map, not to the coarse global map, so localization stays sharp enough for audit and relabeling workflows.

### Bounded Coarse Context

The global context path is now bounded. Only the lowest-resolution feature map passes through a coarse-context attention block, and the exported global feature map is pooled down to `max_global_feature_grid`.

That is the key difference from the older transformer-heavy design:

- context is still global
- memory is no longer allowed to scale with the full imported public image size

### Local Defect Branch

The local branch still uses overlapping tiles extracted from the padded image. It exists for one reason: tiny defects do not survive a purely coarse global path.

The local branch produces:

- tile logits
- top-tile evidence
- a local heatmap

These are fused with the coarse global signal before the final decision heads.

## Conditioning

`station_id` and `geometry_mode` are embedded and fed into lightweight affine conditioning layers. This keeps one shared backbone across stations instead of splitting the system into separate station-specific networks.

## Heads

The model exposes:

- binary accept/reject logit and score
- multi-label defect-family logits and probabilities
- weak heatmap
- top tile evidence

`Base` and `Lite` keep the same public outputs. `Lite` reduces channel counts, fusion repeats, and local-path capacity without changing the contract.

## Tiny-Defect Protection

The design preserves tiny evidence through several layers of protection:

- aspect-preserving preprocessing
- valid-mask-aware training
- overlapping local tiles
- high-resolution fused heatmap head
- delayed fusion of local and global evidence

## Training Stages

### Public Pretraining

Public-data pretraining is now patch-based by default.

The workflow is:

1. import a public Hugging Face dataset into a manifest bundle
2. keep the imported full images on disk
3. sample bounded square grayscale crops at train time
4. run masked reconstruction on those crops with `global_only` forward mode

This is the main VRAM fix for heterogeneous public datasets.

### Domain Adaptation

Domain adaptation keeps the same manifest contract but uses the production unlabeled images. The objective remains consistency between local and global evidence.

### Finetuning

Finetuning uses the full production station canvas and the shared reject/defect/heatmap heads. `DDP` remains available here as a simpler fallback.

### Calibration

Calibration is still per-station and external to the shared backbone.

## Distributed Runtime

The training runtime is now:

- `FSDP`-first for multi-GPU pretraining
- `DDP` fallback for finetune/debug flows
- optional activation checkpointing
- `bf16`/`fp16` autocast
- channels-last support
- explicit memory telemetry in metrics and reports

## Explainability

The architecture graph is still exported through `torch.fx`, but the adapter is now aligned with the CNN-heavy implementation rather than the old transformer branch.

Sample-level XAI still includes:

- attribution map
- heatmap
- top tiles
- prediction bundle

Graph artifacts remain:

- `docs/graphs/base/model_graph.mmd`
- `docs/graphs/lite/model_graph.mmd`

Mermaid is always written; DOT is still optional when Graphviz is available.
