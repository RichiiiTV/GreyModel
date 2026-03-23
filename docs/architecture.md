# GreyModel Architecture Walkthrough

## Overview

`GreyModel` is built for visual inspection of syringes and vials where defect scale varies from a few pixels to the full frame. The architecture is intentionally hybrid:

- A small-stride CNN stem keeps fine grayscale contrast alive.
- A global transformer branch captures whole-object geometry and broad defects.
- A local tiled branch preserves tiny defects and produces weak localization.
- Station and geometry conditioning let one shared model work across square and rectangular stations.

The model is not designed as a generic image classifier. It is built around inline inspection constraints, reproducible calibration, and auditability.

## Data Flow

### 1. Input

The public input is a single `uint8` grayscale image plus station metadata:

- `image_uint8`: `H x W`, values `0-255`.
- `station_id`: used for station-specific conditioning and calibration.
- `geometry_mode`: `rect` or `square`.

### 2. Preprocessing

The preprocessing path preserves the image aspect ratio.

- Resize with aspect fit.
- Pad to the station canvas.
- Emit a valid-pixel mask.
- Normalize only after padding.

This matters because the model should not learn from distorted geometry, especially when the same model has to handle both rectangular and square stations.

### 3. Shared Backbone

The backbone is `GrayInspect-H`.

- The CNN stem extracts early features without aggressive downsampling.
- The global branch works on the padded frame to model fill level, container shape, scratches, cracks, and occlusions.
- The local branch processes overlapping tiles to keep `5x5` defects visible.
- The station-conditioning path modulates features with station and geometry embeddings.

## Model Blocks

### CNN Stem

The stem is deliberately small-stride.

- It avoids destroying micro-defect evidence too early.
- It produces feature maps that still retain local intensity differences.
- It feeds both the local and global branches.

### Global Branch

The global branch is a hierarchical transformer over the padded frame.

- It captures long-range structure and full-image anomalies.
- It uses relative positional bias so the same model can handle rectangular and square canvases.
- It receives the valid-pixel mask to ignore padded regions.

### Local Branch

The local branch runs on overlapping tiles.

- It preserves small defects that would disappear in a global downsample.
- It produces tile scores and tile embeddings.
- It supports weak localization even when training labels are image-level only.

### Fusion

Local and global features are fused before the final heads.

- A gating module balances local evidence against global context.
- This is important when a defect is tiny but should still be judged in the context of the whole part.
- The fused representation feeds the reject and defect-family heads.

### Heads

The framework exposes three main outputs.

- `accept/reject` logit and score.
- Multi-label defect family probabilities.
- A weak heatmap for audit and later relabeling.

## Tiny Defect Preservation

The architecture is specifically tuned for tiny defects.

- Use aspect-preserving preprocessing instead of rescaling into a fixed square.
- Keep a valid mask so padding does not pollute feature learning.
- Use overlapping tiles so a `5x5` defect is seen in more than one context window.
- Avoid collapsing local and global evidence into a single pooled vector too early.

## Base And Lite

### Base

- Higher-capacity model for industrial GPU inference.
- More tile overlap.
- Stronger local detail retention.
- Best for the highest inline quality target.

### Lite

- Distilled or reduced-capacity model for constrained environments.
- Lower overlap.
- Smaller transformer depth.
- Same public input and output contract as `Base`.

## Training Stages

### Pretraining

Pretrain on unlabeled grayscale industrial imagery with masked-image or self-supervised objectives.

### Domain Adaptation

Adapt on unlabeled production frames from the target line so the model learns station optics, motion, and background statistics.

### Supervised Finetuning

Finetune on image-level labels, with optional boxes or masks on curated hard cases.

### Calibration

Fit per-station thresholds and lightweight adapters without splitting into separate backbone models.

## Explainability And Audit

The architecture is meant to produce inspectable artifacts.

- Tile scores for local evidence.
- Heatmaps for weak localization.
- Architecture graphs for model documentation.
- Sample-level explanation bundles for QA and operator review.

The committed graph exports are generated from a traceable adapter around the live PyTorch model so the architecture docs stay aligned with the implementation.

- Base graph: `docs/graphs/base/model_graph.mmd`
- Lite graph: `docs/graphs/lite/model_graph.mmd`
- The graph exporter always writes Mermaid and JSON, and will add Graphviz DOT only when `dot` is available.

## References For The Framework

- Use `torch.fx` for graph tracing.
- Use a modern PyTorch-compatible attribution library such as Captum for saliency and integrated gradients.
- Treat the older CNN visualization repository as inspiration only, not as a runtime dependency.
