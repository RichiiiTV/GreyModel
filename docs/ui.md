# GreyModel UI

## Launch

Start the local Streamlit UI:

```bash
python -m greymodel ui --run-root artifacts --data-root data
```

Start the UI in an HPC or Jupyter notebook session:

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

Start the UI with Slurm defaults for GPU jobs:

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

Preview the exact launch command without starting Streamlit:

```bash
python -m greymodel ui --run-root artifacts --data-root data --dry-run
```

In HPC mode, `--dry-run` also reports the resolved `local_url`, `proxy_url`, `proxy_mode`, and `base_url_path`.

The UI is:

- local only
- no Docker
- no auth
- file-backed

The UI process always runs locally. Slurm integration only affects the jobs launched from the compute pages.

For Jupyter or HPC use:

- `jupyter_port` mode assumes the notebook server proxies arbitrary ports like `/proxy/<port>/` and strips that prefix before forwarding
- `jupyter_service` mode assumes the UI is mounted under a stable prefix and therefore needs `server.baseUrlPath`
- `auto` prefers service-prefix routing when `JUPYTERHUB_SERVICE_URL` is present, otherwise it falls back to notebook-style port proxy detection

## Pages

### Overview

Shows:

- recent run sessions
- stage, variant, and status
- latest usable checkpoints
- recent failure summaries

### Datasets

Shows:

- discovered `dataset_index.json` bundles under `data_root`
- manifest and ontology references
- station config counts
- sample preview

### Train

Provides launch forms for:

- pretrain
- domain-adapt
- finetune
- calibrate

Each job can run in one of two backends:

- `local`: launch a background Python or `torch.distributed.run` subprocess from the machine hosting the UI
- `slurm`: submit an `sbatch --wrap="..."` job and keep the UI as the control surface

The UI never runs the trainer in-process.

### Predict

Provides a manifest-backed prediction form for hierarchical batch inference.

You can use the same local or Slurm backend selection as the training page. Prediction job metadata and logs are written under `<run_root>/ui_jobs/`.

### Evaluate

Lets you:

- browse saved report JSON files
- compare two reports from disk

### Explain

Lets you:

- launch a single-sample explain command
- launch an audit-batch explain command
- browse existing explanation bundles under the run root

Like the training and prediction pages, explain jobs can run locally or via Slurm.

### Failures

Shows:

- failure bundle metadata
- stack traces
- offending sample ids when available

## Data Source

The UI reads the same artifacts the CLI writes:

- `run_status.json`
- `run_registry.jsonl`
- reports
- predictions
- explanation bundles
- failure bundles

There is no separate state store.

## Notes

- Use the UI as a local operator console, not as a multi-user service.
- For long-running training jobs, treat the UI as a launcher and inspector around the on-disk run artifacts.
- UI-submitted jobs, whether local or Slurm-backed, write metadata and logs under `<run_root>/ui_jobs/`.
