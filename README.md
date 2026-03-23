# GreyModel

`GreyModel` is a greenfield Python package for grayscale syringe and vial inspection with one shared model family across rectangular and square stations.

The repo now includes:

- A public inference contract built around `Sample`, `ModelInput`, `ModelOutput`, and `StationConfig`.
- Geometry-aware preprocessing that preserves aspect ratio, pads to a station canvas, and emits valid-pixel masks.
- Tile-grid utilities that verify `5x5`-scale defect coverage.
- A hybrid `GrayInspect-H` PyTorch architecture in `src/greymodel/models/` with:
  - a small-stride CNN stem,
  - a global transformer branch,
  - a local tiled-detail branch,
  - station and geometry conditioning,
  - binary reject, defect-family, and weak-heatmap outputs.
- `BaseModel` and `LiteModel` runtime wrappers with the same public I/O contract.
- A NumPy fallback backend so the public API remains runnable before `torch` is installed.
- Synthetic-defect helpers, calibration utilities, and training hooks for masked pretraining, domain adaptation, and supervised finetuning.

## Project Layout

```text
src/greymodel/
  api.py              Public Base/Lite model wrappers
  preprocessing.py    Aspect-safe grayscale preprocessing
  tiling.py           Tile grids and coverage checks
  synthetic.py        Synthetic defect injection helpers
  calibration.py      Per-station calibration primitives
  losses.py           Supervised and consistency losses
  training.py         Pretraining and finetuning step helpers
  models/             PyTorch GrayInspect-H implementation
tests/                Public API and preprocessing contract tests
docs/architecture.md  Architecture and training overview
```

## Install

Minimum local setup:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pytest
.venv\Scripts\python.exe -m pip install -e .
```

For the PyTorch model path instead of the NumPy fallback:

```powershell
.venv\Scripts\python.exe -m pip install torch
```

## Quick Check

Run the current public test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Example

```python
import numpy as np

from greymodel import BaseModel, ModelInput, StationConfig

station = StationConfig(
    canvas_shape=(256, 704),
    tile_size=32,
    tile_stride=16,
    adapter_id="rect-line-a",
)

request = ModelInput(
    image_uint8=np.zeros((225, 652), dtype=np.uint8),
    station_id="station-01",
    geometry_mode="rect",
)

model = BaseModel(num_defect_families=6)
output = model.forward(request, station)

print(output.reject_score)
print(output.defect_family_probs.shape)
print(output.defect_heatmap.shape)
print(output.top_tiles.shape)
```

## Notes

- `BaseModel` is the higher-capacity inline configuration.
- `LiteModel` keeps the same contract with reduced overlap and smaller capacity for CPU-oriented fallback.
- Without `torch`, the wrappers use a deterministic NumPy fallback for contract testing and scaffolding.
- Patentability still depends on prior-art review and counsel; this repo only implements the technical system.
