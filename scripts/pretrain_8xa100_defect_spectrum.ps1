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
$PretrainCropSize = if ($env:PRETRAIN_CROP_SIZE) { $env:PRETRAIN_CROP_SIZE } else { "512" }
$PretrainNumCrops = if ($env:PRETRAIN_NUM_CROPS) { $env:PRETRAIN_NUM_CROPS } else { "2" }
$PretrainCropScales = if ($env:PRETRAIN_CROP_SCALES) { $env:PRETRAIN_CROP_SCALES } else { "0.75 1.0 1.25" }
$MaxGlobalFeatureGrid = if ($env:MAX_GLOBAL_FEATURE_GRID) { $env:MAX_GLOBAL_FEATURE_GRID } else { "12" }
$DistStrategy = if ($env:DIST_STRATEGY) { $env:DIST_STRATEGY } else { "fsdp" }
$CropScaleArgs = $PretrainCropScales -split "\s+" | Where-Object { $_ -ne "" }

$env:USE_LIBUV = "0"

Write-Host "[GreyModel] Importing DefectSpectrum and converting to 8-bit grayscale at $DatasetDir"
& $PythonBin -m greymodel dataset build-hf `
  --dataset-preset "defect_spectrum_full" `
  --output-dir $DatasetDir `
  --allow-rgb-conversion

Write-Host "[GreyModel] Validating imported manifest"
& $PythonBin -m greymodel dataset validate "$DatasetDir\manifest.jsonl"

Write-Host "[GreyModel] Launching 8xA100 patch-based pretraining with $DistStrategy"
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
  --distributed-strategy $DistStrategy `
  --activation-checkpointing `
  --memory-report `
  --pretrain-crop-size $PretrainCropSize `
  --pretrain-num-crops $PretrainNumCrops `
  --pretrain-crop-scales $CropScaleArgs `
  --max-global-feature-grid $MaxGlobalFeatureGrid `
  --channels-last `
  --log-every-n-steps $LogEveryNSteps `
  --checkpoint-every-n-steps $CheckpointEveryNSteps `
  --keep-last-k-checkpoints $KeepLastKCheckpoints
