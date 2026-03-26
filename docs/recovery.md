# Recovery And Run Layout

## Goal

GreyModel uses best-effort fail-save behavior across:

- training
- evaluation
- prediction
- explainability
- CLI entrypoints

The framework is designed to persist useful artifacts even when a run fails mid-stream.

## Run Layout

For a run root such as `artifacts/`, a stage writes:

```text
artifacts/
  <stage>-<variant>/
    run_status.json
    metrics.jsonl
    epoch_metrics.jsonl
    config_snapshot.json
    manifest_snapshot.json
    checkpoints/
    reports/
    predictions/
    explanations/
    failures/
    sessions/
      <session_id>/
        run_status.json
        reports/
        predictions/
        explanations/
        failures/
```

## Run Status

`run_status.json` tracks:

- stage
- variant
- status
- manifest/index references
- latest checkpoint path
- best checkpoint path
- latest usable checkpoint path
- report path
- summary path
- epoch / global step
- model version

Possible statuses include:

- `created`
- `running`
- `completed`
- `completed_with_failures`
- `failed`

## Failure Bundles

Failure bundles are written under session failures:

```text
.../sessions/<session_id>/failures/<failure_id>/
  failure.json
  traceback.txt
```

`failure.json` includes:

- error type
- error message
- stage / variant
- run dir
- manifest / index references
- latest and best checkpoint paths when available
- offending sample IDs when available
- partial artifacts
- resume metadata

## Run Registry

The run root also stores:

- `run_registry.jsonl`

Registry helpers can:

- list all run sessions
- list failure bundles
- return the latest run
- compare reports

## Recovery Behavior By Surface

Training:

- writes checkpoints as the run progresses
- writes failure bundle on handled crash
- records latest usable checkpoint in run status

Prediction:

- can quarantine failing samples
- writes completed status with failure count when possible

Explainability audit:

- can continue across sample-level failures when routed through the stage runner

CLI:

- top-level exception handling attempts to write a fallback failure bundle if a stage-specific handler did not already do so

## Resume Guidance

If a training run fails, inspect:

- `run_status.json`
- `failures/<failure_id>/failure.json`
- latest checkpoint under `checkpoints/`

Use the recorded checkpoint path with:

```bash
python -m greymodel train resume --manifest ... --checkpoint ...
```
