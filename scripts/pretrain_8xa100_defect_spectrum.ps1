$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = "$RepoRoot\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$PythonBin = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$TorchrunBin = if (Test-Path ".venv\Scripts\torchrun.exe") { ".venv\Scripts\torchrun.exe" } else { "torchrun" }

$DatasetDir = if ($env:DATASET_DIR) { $env:DATASET_DIR } else { "data\public_pretrain\defect_spectrum_full" }
$RunRoot = if ($env:RUN_ROOT) { $env:RUN_ROOT } else { "artifacts" }
$Variant = if ($env:VARIANT) { $env:VARIANT } else { "base" }
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "200" }
$NprocPerNode = if ($env:NPROC_PER_NODE) { $env:NPROC_PER_NODE } else { "8" }
$PerGpuBatchSize = if ($env:PER_GPU_BATCH_SIZE) { $env:PER_GPU_BATCH_SIZE } else { "16" }
$GlobalBatchSize = if ($env:GLOBAL_BATCH_SIZE) { $env:GLOBAL_BATCH_SIZE } else { "128" }
$NumWorkers = if ($env:NUM_WORKERS) { $env:NUM_WORKERS } else { "8" }
$LogEveryNSteps = if ($env:LOG_EVERY_N_STEPS) { $env:LOG_EVERY_N_STEPS } else { "10" }
$CheckpointEveryNSteps = if ($env:CHECKPOINT_EVERY_N_STEPS) { $env:CHECKPOINT_EVERY_N_STEPS } else { "200" }
$KeepLastKCheckpoints = if ($env:KEEP_LAST_K_CHECKPOINTS) { $env:KEEP_LAST_K_CHECKPOINTS } else { "5" }

$env:USE_LIBUV = "0"

Write-Host "[GreyModel] Importing DefectSpectrum and converting to 8-bit grayscale at $DatasetDir"
& $PythonBin -m greymodel dataset build-hf `
  --dataset-name "DefectSpectrum/Defect_Spectrum" `
  --output-dir $DatasetDir `
  --source-dataset "DefectSpectrum/Defect_Spectrum:full" `
  --allow-rgb-conversion

Write-Host "[GreyModel] Validating imported manifest"
& $PythonBin -m greymodel dataset validate "$DatasetDir\manifest.jsonl"

Write-Host "[GreyModel] Launching 8xA100 pretraining"
& $TorchrunBin --standalone --nproc_per_node=$NprocPerNode -m greymodel train pretrain `
  --manifest "$DatasetDir\manifest.jsonl" `
  --index "$DatasetDir\dataset_index.json" `
  --variant $Variant `
  --run-root $RunRoot `
  --epochs $Epochs `
  --batch-size $PerGpuBatchSize `
  --global-batch-size $GlobalBatchSize `
  --num-workers $NumWorkers `
  --precision auto `
  --log-every-n-steps $LogEveryNSteps `
  --checkpoint-every-n-steps $CheckpointEveryNSteps `
  --keep-last-k-checkpoints $KeepLastKCheckpoints
