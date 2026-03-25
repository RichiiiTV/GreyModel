# GreyModel

`GreyModel` is a grayscale inspection framework for syringe and vial defect detection. It is designed around one shared model family that supports rectangular and square stations, protects defects as small as `5x5` pixels, and keeps the same public inference contract across `Base` and `Lite`.

## Start Here

- [Architecture Walkthrough](docs/architecture.md)
- [Framework Guide](docs/framework.md)
- [Base Graph](docs/graphs/base/model_graph.mmd)
- [Lite Graph](docs/graphs/lite/model_graph.mmd)
- [Contributor Rules](AGENTS.md)

## What Changed

- The runtime is now `FSDP`-first for multi-GPU pretraining, with `DDP` kept as a fallback for finetune/debug flows.
- Public-data pretraining is patch-based by default. Imported Hugging Face images are materialized into a manifest bundle, then cropped into bounded square grayscale patches at train time.
- The backbone is CNN-heavy: grayscale stem, ConvNeXt-style pyramid, BiFPN-style fusion, bounded coarse context, and a local defect branch for tiny evidence.
- Training artifacts now include strategy and memory telemetry alongside the existing metrics and checkpoints.

## Public Contract

The public types are unchanged:

- `Sample`
- `ModelInput`
- `ModelOutput`
- `StationConfig`

`Base` and `Lite` still expose the same inference I/O contract.

## Public Pretrain Then Production Finetune

1. Import a public dataset into the GreyModel manifest format.

```bash
python -m greymodel dataset build-hf \
  --dataset-preset defect_spectrum_full \
  --output-dir data/public_pretrain/defect_spectrum_full
```

Available curated presets are:

- `ds_dagm`
- `defect_spectrum_full`
- `mvtec_ad_gray`

2. Validate the imported bundle.

```bash
python -m greymodel dataset validate data/public_pretrain/defect_spectrum_full/manifest.jsonl
```

3. Launch patch-based pretraining.

```bash
torchrun --standalone --nproc_per_node=8 -m greymodel train pretrain \
  --manifest data/public_pretrain/defect_spectrum_full/manifest.jsonl \
  --index data/public_pretrain/defect_spectrum_full/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --epochs 200 \
  --batch-size 16 \
  --global-batch-size 128 \
  --num-workers 8 \
  --precision auto \
  --distributed-strategy fsdp \
  --activation-checkpointing \
  --memory-report \
  --pretrain-crop-size 512 \
  --pretrain-num-crops 2 \
  --pretrain-crop-scales 0.75 1.0 1.25 \
  --max-global-feature-grid 12 \
  --channels-last \
  --log-every-n-steps 10 \
  --checkpoint-every-n-steps 200 \
  --keep-last-k-checkpoints 5
```

4. Build a separate manifest for production data and finetune from the pretrained checkpoint.

```bash
python -m greymodel dataset build /path/to/production_images --output-dir data/production

torchrun --standalone --nproc_per_node=8 -m greymodel train finetune \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --checkpoint artifacts/pretrain-base/checkpoints/latest.pt \
  --variant base \
  --run-root artifacts \
  --epochs 20 \
  --batch-size 8 \
  --global-batch-size 64 \
  --num-workers 8 \
  --precision auto \
  --distributed-strategy ddp
```

## Key Training Flags

- `--distributed-strategy {fsdp,ddp,auto}`: `fsdp` is the default multi-GPU path.
- `--activation-checkpointing`: enables block checkpointing on the CNN-heavy backbone.
- `--memory-report`: logs allocated, reserved, and peak CUDA memory into run metrics and reports.
- `--pretrain-crop-size`: base square crop size for public pretraining.
- `--pretrain-num-crops`: number of crops sampled per imported image each step.
- `--pretrain-crop-scales`: square crop scales relative to `--pretrain-crop-size`.
- `--max-global-feature-grid`: upper bound for the coarse global feature map.
- `--channels-last`: enables channels-last tensors for convolution-heavy training.
- `--ema-decay`: optional EMA for single-device training jobs.
- `--compile-model`: optional `torch.compile` for stable single-device runtimes.

## Hugging Face Notes

- `build-hf` enforces grayscale by default.
- Use `--allow-rgb-conversion` only when a preset intentionally converts public RGB data to `8-bit` grayscale.
- Mixed-resolution public imports are still shape-bucketed into pseudo-stations so giant canvases are not inferred across the full public corpus.
- If Hugging Face returns `429`, provide `HF_TOKEN`, keep `--cache-dir` stable, and rerun with `--local-files-only` once the cache is warm.

## Saved 8xA100 Entry Points

- Linux: `scripts/pretrain_8xa100_defect_spectrum.sh`
- PowerShell: `scripts/pretrain_8xa100_defect_spectrum.ps1`
- Slurm: `scripts/pretrain_8xa100_defect_spectrum.slurm`

The saved scripts now run the patch-based `FSDP` path, not the old full-frame public pretraining flow.

### Copyable Slurm Wrap

If your cluster uses:

```bash
sbatch -c 8 --mem=50G --gres=gpu:8 -p batch_gpu -q 3h --wrap="..."
```

copy this:

```bash
sbatch -c 8 --mem=50G --gres=gpu:8 -p batch_gpu -q 3h --wrap="cd /path/to/GreyModel && source .venv/bin/activate && python -m greymodel dataset build-hf --dataset-preset defect_spectrum_full --output-dir data/public_pretrain/defect_spectrum_full && python -m greymodel dataset validate data/public_pretrain/defect_spectrum_full/manifest.jsonl && torchrun --standalone --nproc_per_node=8 -m greymodel train pretrain --manifest data/public_pretrain/defect_spectrum_full/manifest.jsonl --index data/public_pretrain/defect_spectrum_full/dataset_index.json --variant base --run-root artifacts --epochs 200 --batch-size 16 --global-batch-size 128 --num-workers 8 --precision auto --distributed-strategy fsdp --activation-checkpointing --memory-report --pretrain-crop-size 512 --pretrain-num-crops 2 --pretrain-crop-scales 0.75 1.0 1.25 --max-global-feature-grid 12 --channels-last --log-every-n-steps 10 --checkpoint-every-n-steps 200 --keep-last-k-checkpoints 5"
```

## Notes

- Preserve `8-bit` grayscale until normalization.
- Avoid aspect-ratio distortion when switching between rectangular and square stations.
- Treat patentability as a system property of geometry-aware preprocessing, bounded coarse context, dual-scale evidence, and per-station calibration rather than a single isolated layer choice.
