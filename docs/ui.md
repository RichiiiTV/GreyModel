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

If your browser can load the shell but reports websocket failures on `/_stcore/stream`, rerun with:

```bash
python -m greymodel ui \
  --run-root artifacts \
  --data-root data \
  --proxy-mode jupyter_port \
  --bind-port 8501 \
  --disable-cors \
  --disable-xsrf-protection \
  --print-url
```

These flags are intentionally explicit. They are meant for notebook or HPC proxy environments where the proxy path or origin handling breaks Streamlit's websocket negotiation. In `jupyter_port` mode the UI now uses the generic `/proxy/absolute/<port>/` route so websocket and static paths stay under the proxied prefix.

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
- workspace-backed via `<run_root>/workspace.json` unless overridden with `--workspace-path`

The UI process always runs locally. Slurm integration only affects the jobs launched from the compute pages.

For Jupyter or HPC use:

- `jupyter_port` mode assumes the notebook server exposes arbitrary ports through `/proxy/absolute/<port>/` so the app keeps the full proxy prefix
- `jupyter_service` mode assumes the UI is mounted under a stable prefix and therefore needs `server.baseUrlPath`
- `auto` prefers service-prefix routing when `JUPYTERHUB_SERVICE_URL` is present, otherwise it falls back to notebook-style port proxy detection

## Pages

### Home

Shows:

- recent run sessions
- stage, variant, and status
- latest usable checkpoints
- recent failure summaries
- active workspace summary
- active model profile summary

### Datasets

Shows:

- discovered `dataset_index.json` bundles under `data_root`
- manifest and ontology references
- station config counts
- sample preview
- activation of the current dataset bundle in the workspace

### Models

Shows:

- registered model profiles
- native GreyModel and Hugging Face model profile metadata
- profile editing for backend family, task type, native variant, cache policy, and latency target
- latency benchmark preview

The built-in native profiles are:

- `prod_fast_native`: the production fast path. This is the screen-plus-MIL cascade meant for promotion after latency benchmarking.
- `review_native_base`: the higher-capacity review model.
- `review_native_lite`: the compact review model.

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

### Predict & Review

Provides a manifest-backed preview and review flow for hierarchical inference.

You can use the same local or Slurm backend selection as the training page for job launches, and the page also supports in-app preview batches using the selected workspace profile. Prediction job metadata and logs are written under `<run_root>/ui_jobs/`.

If the selected profile is `prod_fast_native`, the in-app preview uses the production cascade:

- stage A screens the full frame
- clearly good samples can exit immediately
- uncertain or suspicious samples go to the patch/MIL refiner

### Evaluate

Lets you:

- browse saved report JSON files
- compare two reports from disk

### Explain

Lets you:

- launch a single-sample explain command
- launch an audit-batch explain command
- browse existing explanation bundles under the run root
- generate review bundles from the active model profile

Like the training and prediction pages, explain jobs can run locally or via Slurm.

### Failures

Shows:

- failure bundle metadata
- stack traces
- offending sample ids when available

### Settings

Shows:

- workspace name and storage roots
- default execution backend
- Slurm defaults
- proxy mode and UI theme
- save/load of workspace-local preferences

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
