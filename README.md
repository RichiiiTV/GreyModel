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
- `train`, `eval`, `dataset`, and `explain` CLI entrypoints share the same contract.
- Production pretraining is launched with `torchrun` and native PyTorch DDP on a multi-GPU node.
- Smoke runs are still useful for quick wiring checks, but they are not the same as a real epoch-based training job.
- Public Hugging Face datasets can be materialized into the same manifest format with `greymodel dataset build-hf`.
- Training commands show `tqdm` progress bars by default; use `--no-progress` for quiet runs.

## Runtime Expectations

- `Base` is the inline production configuration for industrial GPU inference.
- `Lite` keeps the same contract with reduced overlap and smaller capacity for CPU-constrained deployment.
- Both should operate on the same grayscale image contract and station metadata.
- Training runs should emit checkpoints, metrics JSONL, and a run report under a deterministic artifact directory.

## Notes

- Preserve 8-bit grayscale values until normalization.
- Avoid aspect-ratio distortion when moving between rectangular and square stations.
- Treat patentability as a system-level property of geometry-aware preprocessing, dual-scale inference, weak localization, and calibration.
- The framework CLI entrypoint is `greymodel`, with `dataset`, `train`, `eval`, and `explain` command groups.
- Example production pretraining invocation:
```bash
torchrun --nproc_per_node=4 -m greymodel train pretrain --manifest /path/to/manifest.jsonl --index /path/to/dataset_index.json --run-root artifacts
```

## Public Pretrain Then Production Finetune

1. Materialize a public grayscale Hugging Face dataset into a local manifest bundle:
```bash
greymodel dataset build-hf --dataset-name <hf-dataset> --output-dir data/public_pretrain --split train --split validation
```
If the source is not strictly grayscale but you still want to use it, replace the last flag with `--allow-rgb-conversion`.

2. Launch multi-GPU pretraining from the imported manifest:
```bash
torchrun --nproc_per_node=4 -m greymodel train pretrain --manifest data/public_pretrain/manifest.jsonl --index data/public_pretrain/dataset_index.json --run-root artifacts
```

3. Build a separate manifest for your production images and finetune from the pretrained checkpoint:
```bash
greymodel dataset build /path/to/production_images --output-dir data/production
torchrun --nproc_per_node=4 -m greymodel train finetune --manifest data/production/manifest.jsonl --index data/production/dataset_index.json --checkpoint artifacts/pretrain-base/checkpoints/best.pt --run-root artifacts
```

## Saved 8xA100 Entry Points

For a larger public pretraining run, the repo includes executable entrypoint files that import the full `DefectSpectrum/Defect_Spectrum` dataset, convert it to `8-bit` grayscale, validate it, and launch pretraining:

- Linux: [pretrain_8xa100_defect_spectrum.sh](/c:/Users/Ricardo/Desktop/GreyModel/scripts/pretrain_8xa100_defect_spectrum.sh)
- PowerShell: [pretrain_8xa100_defect_spectrum.ps1](/c:/Users/Ricardo/Desktop/GreyModel/scripts/pretrain_8xa100_defect_spectrum.ps1)

Examples:
```bash
bash scripts/pretrain_8xa100_defect_spectrum.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pretrain_8xa100_defect_spectrum.ps1
```
