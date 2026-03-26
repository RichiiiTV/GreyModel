# GreyModel Framework Guide

## Scope

`GreyModel` is a local-first framework for grayscale syringe and vial inspection. It combines:

- dataset ingestion and normalization
- public pretraining
- domain adaptation
- supervised finetuning
- calibration
- hierarchical batch prediction
- explainability
- recovery and run tracking
- a local Streamlit UI

The primary decision is always `good` vs `bad`. Defect-family probabilities are secondary evidence for bad samples.

## End-To-End Workflow

### 1. Build a dataset bundle

Folder-first import:

```bash
python -m greymodel dataset build /path/to/images --output-dir data/production
python -m greymodel dataset validate data/production/manifest.jsonl
python -m greymodel dataset ontology --manifest data/production/manifest.jsonl
```

Public Hugging Face import:

```bash
python -m greymodel dataset build-hf \
  --dataset-preset defect_spectrum_full \
  --output-dir data/public_pretrain/defect_spectrum_full
```

### 2. Train

Pretrain:

```bash
torchrun --standalone --nproc_per_node=8 -m greymodel train pretrain \
  --manifest data/public_pretrain/defect_spectrum_full/manifest.jsonl \
  --index data/public_pretrain/defect_spectrum_full/dataset_index.json \
  --variant base \
  --run-root artifacts
```

Domain adaptation:

```bash
python -m greymodel train domain-adapt \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts
```

Finetune:

```bash
python -m greymodel train finetune \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts
```

Calibration:

```bash
python -m greymodel train calibrate \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts
```

### 3. Evaluate

Benchmark:

```bash
python -m greymodel eval benchmark \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts
```

Threshold sweep:

```bash
python -m greymodel eval threshold-sweep \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base
```

Compare reports:

```bash
python -m greymodel eval compare \
  --left-report artifacts/benchmark-base/reports/benchmark_report.json \
  --right-report artifacts/predict-base/reports/predict_report.json
```

### 4. Predict

Manifest-backed prediction:

```bash
python -m greymodel predict \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --evidence-policy bad
```

Folder-backed prediction:

```bash
python -m greymodel predict \
  --input-dir /path/to/incoming_images \
  --variant lite \
  --run-root artifacts
```

### 5. Explain

Single sample:

```bash
python -m greymodel explain sample \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --sample-id sample_001 \
  --variant base \
  --run-root artifacts
```

Audit batch:

```bash
python -m greymodel explain audit \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --limit 20
```

## Artifact Layout

Every run writes under `<run_root>/<stage>-<variant>/`.

Core files:

- `run_status.json`
- `metrics.jsonl`
- `epoch_metrics.jsonl`
- `config_snapshot.json`
- `manifest_snapshot.json`
- `checkpoints/`
- `reports/`
- `predictions/`
- `explanations/`
- `failures/`
- `sessions/<session_id>/...`

The latest stage state stays in the stable stage directory. Session folders preserve point-in-time status and failure artifacts.

## Prediction Records

Prediction persistence is hierarchical:

- `primary_label`
- `primary_score`
- `top_defect_family`
- `defect_family_probs`
- `evidence`

This is the contract used by:

- batch prediction output
- evaluation
- recovery bundles
- the UI

## Recovery Model

If a run fails after initialization, GreyModel writes:

- run status with `failed` or `completed_with_failures`
- traceback bundle
- manifest and index references
- checkpoint references when available
- offending sample ids for quarantined batch failures

See [recovery.md](recovery.md) for the exact payload.

## UI

Launch:

```bash
python -m greymodel ui --run-root artifacts --data-root data
```

The UI is local-only and reads the same on-disk artifacts that the CLI writes.

If your compute environment is Slurm-backed, start the UI with cluster defaults so the `Train`, `Predict`, and `Explain` pages submit GPU work through `sbatch` instead of local subprocesses:

```bash
python -m greymodel ui \
  --run-root artifacts \
  --data-root data \
  --default-execution-backend slurm \
  --slurm-cpus 8 \
  --slurm-mem 50G \
  --slurm-gres gpu:8 \
  --slurm-partition batch_gpu \
  --slurm-queue 3h \
  --slurm-nproc-per-node 8
```

The Streamlit process itself still stays local. Only the launched GPU jobs are scheduled through Slurm.
