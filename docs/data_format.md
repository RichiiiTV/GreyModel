# Data Format

## External Input

GreyModel accepts external data in two main ways:

- folder-first production import
- Hugging Face import for public pretraining

## Folder-First Import

Expected image types include:

- `.npy`
- `.pgm`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

Optional sidecar files use the same base name with `.json`.

Example:

```text
station_01/
  good/
    sample_001.npy
    sample_001.json
  bad/
    sample_002.npy
    sample_002.json
```

Supported sidecar keys:

- `station_id`
- `product_family`
- `geometry_mode`
- `accept_reject`
- `defect_tags`
- `boxes`
- `mask_path`
- `split`
- `capture_metadata`
- `source_dataset`
- `review_state`

## Canonical Manifest

Each manifest row is a `DatasetRecord`.

Core fields:

- `sample_id`
- `image_path`
- `station_id`
- `product_family`
- `geometry_mode`
- `accept_reject`
- `defect_tags`
- `boxes`
- `mask_path`
- `split`
- `capture_metadata`
- `source_dataset`
- `review_state`

## Dataset Index

`dataset_index.json` stores framework metadata:

- manifest version
- ontology version
- root dir
- manifest path
- split path
- ontology path
- hard-negative path
- station configs
- grouping keys
- split assignments
- metadata

## Station Configs

Each station config includes:

- `canvas_shape`
- `station_id`
- `geometry_mode`
- `pad_value`
- `normalization_mean`
- `normalization_std`
- `tile_size`
- `tile_stride`
- `adapter_id`
- `reject_threshold`
- `defect_thresholds`

## Ontology

`ontology.json` currently records:

- ontology version
- defect tags
- product families
- stations

Use `python -m greymodel dataset ontology ...` to inspect it.

## Prediction Records

Batch prediction writes hierarchical `PredictionRecord` rows with:

- `sample_id`
- `station_id`
- `accept_reject`
- `reject_score`
- `predicted_label`
- `primary_label`
- `primary_score`
- `top_defect_family`
- `defect_family_probs`
- `evidence`
- `split`
- `defect_scale`
- `metadata`

## Failure Records

Failure bundles persist `FailureRecord` JSON with:

- failure id
- stage
- variant
- status
- error type and message
- traceback path
- offending sample IDs
- latest checkpoint metadata
- partial artifact paths
- resume metadata
