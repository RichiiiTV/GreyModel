# GreyModel Framework Guide

## Goal

This repo is more than a backbone implementation. The framework should support dataset curation, staged training, evaluation, calibration, and explainability for grayscale syringe and vial inspection.

## Dataset Model

### Folder-First Import

External data should be imported from folders, because that matches how inspection images are usually handed off in practice.

Internally, the framework should normalize folders into versioned manifests so the training and evaluation runs are reproducible.

### Public Hugging Face Pretraining Import

Pretraining can also start from a public Hugging Face image dataset, but the framework should still materialize it locally before training.

- Import the source dataset with `greymodel dataset build-hf`.
- Preserve split intent from Hugging Face, while normalizing `validation` to `val`.
- Enforce grayscale by default so pretraining does not silently drift into an RGB workflow.
- Materialize local `.npy` images and reuse the same `manifest.jsonl` and `dataset_index.json` contracts as folder imports.
- If the remote dataset is rate-limited, prefer a token-backed import with a stable cache directory and switch to local-cache-only replays once the cache is populated.
- Mixed-resolution public imports should be grouped into pseudo-stations by image shape so inferred canvas sizes stay bounded during pretraining.

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

### Smoke Runs Versus Real Training

The repo supports two different execution modes.

- Smoke runs are short, low-cost checks that validate wiring, manifests, and model startup.
- Real training runs are epoch-based, checkpointed jobs launched through `torchrun` with native PyTorch DDP on a multi-GPU node.

Smoke runs are useful for development, but they are not a substitute for a production pretraining or finetuning job.

### Distributed Pretraining

Pretraining should be launched with `torchrun` so the same code path can scale across GPUs without changing the model contract.

- Use manifest-backed datasets and deterministic splits.
- Shard data with DDP rather than hand-picking a single batch.
- Track `epoch`, `global_step`, and optimizer/scheduler/scaler state in checkpoints.
- Keep the run directory deterministic so a job can be resumed or audited later.

On clusters, it is valid to wrap the saved entrypoint script in the scheduler command rather than inlining the full training command. For example, on a system that uses:

```bash
sbatch -c 8 --mem=50G --gres=gpu:8 -p batch_gpu -q 3h --wrap="cd /path/to/GreyModel && bash scripts/pretrain_8xa100_defect_spectrum.sh"
```

the saved script remains the canonical place for dataset import and pretraining defaults, while the scheduler command controls resources and queue policy.

### Pretrain

Use large unlabeled grayscale imagery and masked-image pretraining to learn generic line and part structure.

Recommended workflow:

1. Import a public grayscale Hugging Face dataset into `data/public_pretrain/`.
2. Run `torchrun ... greymodel train pretrain` on that manifest.
3. Build a separate manifest for your own production images.
4. Warm-start `greymodel train finetune` from the best pretraining checkpoint.

### Domain Adapt

Adapt on unlabeled production frames from the target line before supervised finetuning.

### Finetune

Use image-level labels as the main supervision signal.

- Add boxes or masks only for curated hard cases.
- Keep the sampling station-aware.
- Use hard clean negatives and recent false rejects/false accepts.

### Calibrate

Fit station-specific thresholds and, if needed, lightweight adapters.

## Run Artifacts

Training and evaluation jobs should write a predictable artifact set.

- `metrics.jsonl` for per-step and per-epoch metrics.
- `config_snapshot.json` for the resolved training configuration.
- `manifest_snapshot.json` for the dataset/index references used by the run.
- `checkpoints/` for latest and best checkpoints.
- `reports/` for summary reports, calibration outputs, and benchmark results.

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
