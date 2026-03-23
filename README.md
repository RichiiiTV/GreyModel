# GreyModel

`GreyModel` is a grayscale inspection framework for syringe and vial defect detection. It is built around one shared model family that can run on rectangular and square stations, preserve tiny defects as small as `5x5` pixels, and keep the same public I/O contract between `Base` and `Lite`.

## What This Repo Contains

- A hybrid `GrayInspect-H` backbone for grayscale `8-bit` images.
- Geometry-aware preprocessing with pad masks instead of aspect distortion.
- A dataset layer designed for folder-first ingestion with internal manifests.
- Training, evaluation, calibration, and explainability hooks for finetuning.
- Runtime wrappers with `Base` and `Lite` parity.

## Start Here

- [Architecture Walkthrough](docs/architecture.md)
- [Framework Guide](docs/framework.md)
- [Base Graph](docs/graphs/base/model_graph.mmd)
- [Lite Graph](docs/graphs/lite/model_graph.mmd)
- [Contributor Rules](AGENTS.md)

## Public Contract

The core public types are:

- `Sample` for labeled training examples.
- `ModelInput` for single-image inference.
- `ModelOutput` for reject score, defect family probabilities, and heatmaps.
- `StationConfig` for station-specific canvas, tile, and calibration settings.

## Current Focus

This repo is organized around finetuning and inspection workflow, not only model definition.

- Folder-first dataset import with internal manifest generation.
- Station-separated splits to avoid leakage.
- Weak localization through tile scores and heatmaps.
- Architecture graph and per-sample explainability artifacts.
- `train`, `eval`, `dataset`, and `explain` CLI entrypoints planned around the same contract.

## Runtime Expectations

- `Base` is the inline production configuration for industrial GPU inference.
- `Lite` keeps the same contract with reduced overlap and smaller capacity for CPU-constrained deployment.
- Both should operate on the same grayscale image contract and station metadata.

## Notes

- Preserve 8-bit grayscale values until normalization.
- Avoid aspect-ratio distortion when moving between rectangular and square stations.
- Treat patentability as a system-level property of geometry-aware preprocessing, dual-scale inference, weak localization, and calibration.
- The framework CLI entrypoint is `greymodel`, with `dataset`, `train`, `eval`, and `explain` command groups.
