# AGENTS.md

## Purpose

This repository is a grayscale inspection framework for syringes and vials. Future contributors and agents should treat it as a production-oriented computer vision system, not as a generic image-classification demo.

## Non-Negotiables

- Preserve the public inference contract unless the user explicitly asks to change it.
- Keep support for both rectangular and square stations.
- Preserve `5x5` defect sensitivity.
- Keep `Base` and `Lite` I/O parity.
- Keep grayscale `8-bit` input handling first-class.
- Avoid aspect-ratio distortion in preprocessing.

## Working Style

- Prefer framework additions around dataset curation, evaluation, explainability, and calibration before changing the backbone.
- Use folder-first ingestion for external data, but normalize it into internal manifests for reproducibility.
- Keep changes localized and do not revert unrelated edits.
- Do not touch files outside the scope of the task unless explicitly requested.

## Documentation Expectations

- Update `README.md` when public workflow or install instructions change.
- Keep `docs/architecture.md` as the implementation walkthrough for the model and training stages.
- Add new documentation under `docs/` when introducing framework subsystems.
- Document graph/XAI artifact locations whenever explainability changes.

## Framework Expectations

- Dataset tooling should support manifest generation, split reproducibility, and hard-negative mining.
- Evaluation should report per-station and per-defect-scale metrics.
- Explainability should include architecture graphs and per-sample audit bundles.
- Training workflows should cover pretraining, domain adaptation, supervised finetuning, and calibration.

## Testing Expectations

- Add or update tests for each new framework subsystem.
- Preserve regression coverage for preprocessing, tile coverage, and Base/Lite parity.
- Do not weaken the existing public API contract without a clear request.

## Suggested Artifact Layout

- `docs/architecture.md` for the model and training walkthrough.
- `docs/framework.md` for dataset, CLI, evaluation, and XAI concepts.
- `docs/graphs/` for architecture graph exports.
- `artifacts/` or `reports/` for run outputs if a future task adds them.
