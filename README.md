# GreyModel

`GreyModel` is a local grayscale inspection framework for syringe and vial visual inspection. It keeps `8-bit` grayscale handling first-class, preserves rectangular and square station support, and keeps the final decision contract hierarchical:

- primary decision: `good` vs `bad`
- secondary output: defect-family probabilities for bad samples
- evidence: heatmaps, top tiles, and station decision metadata

The framework is local-first and filesystem-based. There is no Docker requirement and no backend service.

## What It Covers

- folder-first dataset ingestion normalized into manifests
- public Hugging Face pretraining import
- pretraining, domain adaptation, finetuning, and calibration
- batch prediction with persisted hierarchical prediction records
- explainability bundles and audit runs
- failure bundles and run-status tracking
- local run registry and report comparison
- a local Streamlit UI

## Install

Minimal install:

```bash
pip install -e .
```

Full framework install:

```bash
pip install -e .[framework]
```

The `framework` extra includes PyTorch, `datasets`, Captum, Pillow, tqdm, and Streamlit.

## Documentation

- [Framework Guide](docs/framework.md)
- [Architecture Walkthrough](docs/architecture.md)
- [Data Format](docs/data_format.md)
- [Recovery And Run Layout](docs/recovery.md)
- [UI Guide](docs/ui.md)
- [Contributor Rules](AGENTS.md)
- [Base Graph](docs/graphs/base/model_graph.mmd)
- [Lite Graph](docs/graphs/lite/model_graph.mmd)

## Quickstart

Build a production bundle:

This scans a folder tree of production images, normalizes it into the internal manifest format, validates the records, and shows the resolved defect ontology that later training and evaluation stages will use.

```bash
python -m greymodel dataset build /path/to/production_images --output-dir data/production
python -m greymodel dataset validate data/production/manifest.jsonl
python -m greymodel dataset ontology --manifest data/production/manifest.jsonl
```

Import a public pretraining bundle:

This pulls a public Hugging Face dataset preset, converts it into the same local bundle format, and prepares it for patch-based grayscale pretraining.

```bash
python -m greymodel dataset build-hf \
  --dataset-preset defect_spectrum_full \
  --output-dir data/public_pretrain/defect_spectrum_full
```

Pretrain:

This runs self-supervised public-data pretraining. The model learns generic grayscale defect structure before it ever sees your production labels.

```bash
torchrun --standalone --nproc_per_node=8 -m greymodel train pretrain \
  --manifest data/public_pretrain/defect_spectrum_full/manifest.jsonl \
  --index data/public_pretrain/defect_spectrum_full/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --epochs 200 \
  --batch-size 16 \
  --global-batch-size 128 \
  --distributed-strategy fsdp \
  --activation-checkpointing \
  --memory-report \
  --pretrain-crop-size 512 \
  --pretrain-num-crops 2 \
  --pretrain-crop-scales 0.75 1.0 1.25 \
  --max-global-feature-grid 12 \
  --channels-last
```

If you already have unlabeled production images, the next stage is domain adaptation. `python -m greymodel train domain-adapt ...` keeps the shared backbone but adapts it to the real production image distribution before supervised finetuning.

Finetune:

This is the main supervised training stage on your production dataset. It learns the final `good` vs `bad` decision boundary and the secondary defect-family outputs.

```bash
python -m greymodel train finetune \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --epochs 20 \
  --batch-size 8 \
  --global-batch-size 64 \
  --distributed-strategy ddp
```

Batch prediction:

This runs hierarchical inference over a manifest and writes persisted prediction records, reports, explanations, and failure bundles under the chosen run root.

```bash
python -m greymodel predict \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --variant base \
  --run-root artifacts \
  --evidence-policy bad
```

Single-sample explainability:

This generates a local audit bundle for one image, including the prediction payload, heatmap, attribution artifacts, and top-tile evidence.

```bash
python -m greymodel explain sample \
  --manifest data/production/manifest.jsonl \
  --index data/production/dataset_index.json \
  --sample-id sample_001 \
  --variant base \
  --run-root artifacts
```

Local UI:

This launches the local Streamlit operator UI for browsing datasets, runs, failures, reports, and explanations. The UI process itself stays local. From inside the UI, the `Train`, `Predict`, and `Explain` pages can either launch local subprocesses or submit GPU jobs to Slurm.

```bash
python -m greymodel ui --run-root artifacts --data-root data
```

HPC or Jupyter-proxied UI:

This keeps the UI process local to your notebook or HPC session, but resolves the externally reachable URL through either Jupyter port proxying or JupyterHub service-prefix routing. `--print-url` prints the proxied URL first so you can open the correct route instead of guessing the raw port.

Notebook-style arbitrary port proxy:

```bash
python -m greymodel ui \
  --run-root artifacts \
  --data-root data \
  --proxy-mode auto \
  --bind-port 8501 \
  --print-url
```

JupyterHub service-prefix routing:

```bash
python -m greymodel ui \
  --run-root artifacts \
  --data-root data \
  --proxy-mode jupyter_service \
  --base-url-path /services/greymodel/ \
  --print-url
```

Local UI with Slurm defaults:

This launches the same UI, but it preconfigures the execution backend so the job forms default to `sbatch` submissions instead of local Python processes. The UI still runs locally; only the GPU workloads are handed off to Slurm.

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

Dry-run the UI launch command:

This prints the exact Streamlit launch command without starting the UI, which is useful for debugging environments or copying the command into another shell. It also returns the resolved local and proxied URLs, plus any Slurm defaults that will be injected into the UI job forms.

```bash
python -m greymodel ui --run-root artifacts --data-root data --dry-run
```

## Main CLI Surface

Dataset:

- `python -m greymodel dataset build ...`: Scan a raw image folder, infer labels and metadata where possible, and materialize a canonical bundle with `manifest.jsonl`, `dataset_index.json`, `splits.json`, `ontology.json`, and `hard_negatives.jsonl`.
- `python -m greymodel dataset build-hf ...`: Import a public Hugging Face dataset into the same canonical local bundle format, converting images to grayscale when configured.
- `python -m greymodel dataset validate ...`: Check that a manifest is internally consistent, image files exist, geometry modes are valid, and the grayscale contract is preserved.
- `python -m greymodel dataset split ...`: Rebuild leakage-safe split assignments so station, day, batch, and camera metadata can stay separated.
- `python -m greymodel dataset hard-negatives ...`: Build a reusable subset of difficult clean negatives or confusing examples for later curation and retraining.
- `python -m greymodel dataset ontology ...`: Inspect the resolved defect-tag ontology that the framework will use for defect-family outputs and reports.

Training:

- `python -m greymodel train pretrain ...`: Run public-data self-supervised pretraining to teach the backbone general grayscale defect structure before production supervision.
- `python -m greymodel train domain-adapt ...`: Adapt the pretrained model to unlabeled production imagery so it better matches the real station distribution.
- `python -m greymodel train finetune ...`: Run supervised training on labeled production images to learn final reject scoring and defect-family behavior.
- `python -m greymodel train resume ...`: Continue an interrupted supervised training run from a saved checkpoint and its optimizer state.
- `python -m greymodel train calibrate ...`: Build station-specific calibration outputs such as reject thresholds after prediction or finetuning.

Evaluation:

- `python -m greymodel eval benchmark ...`: Run a full benchmark over a manifest and write binary quality metrics, per-station slices, per-scale slices, and bad-sample defect-family results.
- `python -m greymodel eval threshold-sweep ...`: Export metric behavior over a range of reject thresholds so you can choose operating points explicitly.
- `python -m greymodel eval calibration ...`: Produce calibration statistics and recommended per-station reject thresholds from a manifest or saved predictions.
- `python -m greymodel eval compare --left-report ... --right-report ...`: Compare two saved reports directly from disk to see metric deltas between runs.

Explainability:

- `python -m greymodel explain graph ...`: Export the current model architecture graph as Mermaid and JSON artifacts from the actual PyTorch model.
- `python -m greymodel explain sample ...`: Generate a full explanation bundle for one manifest sample, including heatmaps, attribution, and prediction metadata.
- `python -m greymodel explain audit ...`: Build explanation bundles for a limited set of samples so an operator or engineer can review localization quality in batch.

Framework operations:

- `python -m greymodel predict ...`: Run hierarchical batch inference over either a manifest or a raw folder and persist predictions, reports, explanations, and quarantined failures.
- `python -m greymodel ui ...`: Launch the local Streamlit UI that reads the same on-disk framework artifacts the CLI produces. It can auto-detect Jupyter/HPC proxying, print the resolved proxy URL, and still preseed Slurm defaults for UI-launched GPU jobs.

## Run Artifacts

Each stage writes a stable run directory under the chosen `run_root`:

- `<stage>-<variant>/run_status.json`
- `<stage>-<variant>/metrics.jsonl`
- `<stage>-<variant>/epoch_metrics.jsonl`
- `<stage>-<variant>/config_snapshot.json`
- `<stage>-<variant>/manifest_snapshot.json`
- `<stage>-<variant>/checkpoints/`
- `<stage>-<variant>/reports/`
- `<stage>-<variant>/predictions/`
- `<stage>-<variant>/explanations/`
- `<stage>-<variant>/failures/`
- `<stage>-<variant>/sessions/<session_id>/...`

The framework also appends status events to `run_root/run_registry.jsonl`.

## Recovery

Failures are persisted as artifacts instead of being dropped. Failure bundles can contain:

- traceback
- run status metadata
- manifest and index references
- checkpoint references
- offending sample ids
- partial artifact paths
- resume metadata

Training, explainability, evaluation, and batch prediction all use the same failure bundle pattern.

## Cluster Example

If your cluster uses the custom wall-time flag `-q 3h`, a copyable direct Python submit looks like:

```bash
sbatch -c 8 --mem=50G --gres=gpu:8 -p batch_gpu -q 3h --wrap="cd /path/to/GreyModel && source .venv/bin/activate && python -m greymodel dataset build-hf --dataset-preset defect_spectrum_full --output-dir data/public_pretrain/defect_spectrum_full && python -m greymodel dataset validate data/public_pretrain/defect_spectrum_full/manifest.jsonl && torchrun --standalone --nproc_per_node=8 -m greymodel train pretrain --manifest data/public_pretrain/defect_spectrum_full/manifest.jsonl --index data/public_pretrain/defect_spectrum_full/dataset_index.json --variant base --run-root artifacts --epochs 200 --batch-size 16 --global-batch-size 128 --num-workers 8 --precision auto --distributed-strategy fsdp --activation-checkpointing --memory-report --pretrain-crop-size 512 --pretrain-num-crops 2 --pretrain-crop-scales 0.75 1.0 1.25 --max-global-feature-grid 12 --channels-last --log-every-n-steps 10 --checkpoint-every-n-steps 200 --keep-last-k-checkpoints 5"
```

If you prefer to keep the operator workflow inside the UI, start the UI itself with the same Slurm defaults:

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

## Non-Negotiables

- preserve `8-bit` grayscale handling
- avoid aspect-ratio distortion
- keep rectangular and square station support
- keep `5x5` defect sensitivity
- preserve `Base` and `Lite` I/O parity
