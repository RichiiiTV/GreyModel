# GreyModel Architecture

## System Goal

Build a grayscale-native inspection system for syringe and vial images that can:

- Detect tiny local defects as small as `5x5` pixels.
- Detect broad defects that span most or all of the image.
- Run on both rectangular and square station formats.
- Reuse one shared model family across stations.
- Support a stronger `Base` configuration and a smaller `Lite` configuration with the same I/O contract.

## Data Contract

### `Sample`

Represents one training example.

- `image_uint8`: single-channel `H x W` image in the range `0-255`.
- `station_id`: station identifier used for calibration and geometry conditioning.
- `product_family`: syringe, vial, or a more specific family if needed.
- `geometry_mode`: rectangular or square station mode.
- `accept_reject`: binary label.
- `defect_tags`: optional multi-label defect family tags.
- metadata: optional camera, lighting, batch, and capture details.

### `ModelInput`

Represents one inference request.

- One grayscale image.
- `station_id`.
- `geometry_mode`.

### `ModelOutput`

Represents one inference response.

- `reject_score`.
- `accept_reject_logit`.
- `defect_family_probs`.
- `defect_heatmap`.
- `top_tiles`.
- calibrated decision metadata.

### `StationConfig`

Defines how a specific station is processed.

- Canvas and padding rule.
- Normalization statistics.
- Tile size and overlap.
- Adapter identifier.
- Decision thresholds.

## Model Shape

Use one hybrid architecture with three parts:

- A small-stride CNN stem to preserve fine local contrast.
- A global hierarchical branch for whole-image geometry and large defects.
- A tiled local-detail branch for tiny defects and weak localization.

Fuse local and global features before the final heads. Keep `station_id` conditioning outside the backbone via lightweight adapters or FiLM-style modulation.

## Base and Lite

### `Base`

- Primary inline deployment model.
- Higher capacity.
- More tile overlap and stronger localization.
- Intended for industrial GPU inference.

### `Lite`

- Distilled from `Base`.
- Smaller transformer depth and reduced tile overlap.
- Intended for CPU fallback or constrained deployments.
- Must keep the same public input and output contract.

## Training Plan

### Stage 1: Pretraining

Train on large unlabeled grayscale industrial imagery using masked-image or self-supervised objectives.

### Stage 2: Domain Adaptation

Adapt on unlabeled syringe and vial production frames. Preserve image structure and station geometry.

### Stage 3: Supervised Finetuning

Train on image-level accept/reject labels and optional defect-family tags. Use hard clean negatives and station-balanced batches.

### Stage 4: Calibration

Fit per-station thresholds and lightweight adapters on real production data. Avoid splitting into separate station backbones.

## Weak Localization

The model should keep weak localization even if the primary label is image-level classification.

- Produce tile scores from overlapping local crops.
- Aggregate tile scores into a heatmap.
- Use the heatmap for auditability, hard-example mining, and future relabeling.

## Evaluation

Track performance separately for:

- Tiny defects.
- Medium defects.
- Large or image-spanning defects.
- Clean hard negatives.

Recommended metrics include recall, false accept rate, false reject rate, AUROC, and station-separated validation splits.

## Implementation Notes

- Preserve 8-bit grayscale values until normalization.
- Avoid aspect-ratio distortion when moving between rectangular and square stations.
- Use masks for padded pixels.
- Keep inference deterministic enough for inline QA and audit logging.
- Treat patentability as a system-level combination of geometry-aware preprocessing, dual-scale inference, weak localization, and calibration.
