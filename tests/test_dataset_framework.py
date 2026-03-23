from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from greymodel import (
    GreyInspectionDataset,
    ModelInput,
    Sample,
    StationConfig,
    build_dataset_manifest,
    build_dataset_splits,
    build_hard_negative_subset,
    collate_batch,
    group_samples_by_station,
    ingest_dataset_manifest,
    load_dataset_manifest,
    preprocess_sample,
)
from greymodel.preprocessing import preprocess_and_stack


def _make_sample(
    image: np.ndarray,
    station_id: int,
    product_family: str = "vial",
    geometry_mode: str = "rect",
    accept_reject: int = 0,
    defect_tags=(),
) -> Sample:
    return Sample(
        image_uint8=image,
        station_id=station_id,
        product_family=product_family,
        geometry_mode=geometry_mode,
        accept_reject=accept_reject,
        defect_tags=defect_tags,
    )


def _station_config(station_id: int, geometry_mode: str, canvas_shape=(256, 704)) -> StationConfig:
    return StationConfig(
        canvas_shape=canvas_shape,
        station_id=station_id,
        geometry_mode=geometry_mode,
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id=f"{geometry_mode}-{station_id}",
        reject_threshold=0.5,
    )


def _write_image(root: Path, relative_path: str, image: np.ndarray, sidecar: dict | None = None) -> Path:
    image_path = root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    if sidecar is not None:
        image_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return image_path


def test_group_samples_by_station_groups_by_original_key_type() -> None:
    samples = [
        _make_sample(np.zeros((8, 8), dtype=np.uint8), station_id=1),
        _make_sample(np.zeros((8, 8), dtype=np.uint8), station_id=2),
        _make_sample(np.zeros((8, 8), dtype=np.uint8), station_id=1),
    ]

    grouped = group_samples_by_station(samples)

    assert sorted(grouped) == [1, 2]
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1


def test_grey_inspection_dataset_returns_prepared_sample_and_config() -> None:
    sample = _make_sample(np.zeros((8, 12), dtype=np.uint8), station_id=1)
    config = _station_config(1, "rect", canvas_shape=(16, 16))
    dataset = GreyInspectionDataset([sample], [config])

    item = dataset[0]

    assert item["sample"] is sample
    assert item["station_config"] is config
    assert item["prepared"].canvas_shape == (16, 16)
    assert item["prepared"].valid_mask.dtype == np.bool_


def test_preprocess_and_stack_preserves_station_geometry_and_batch_shapes() -> None:
    samples = [
        _make_sample(np.zeros((8, 12), dtype=np.uint8), station_id=1),
        _make_sample(np.zeros((10, 10), dtype=np.uint8), station_id=2, geometry_mode="square"),
    ]
    configs = [
        _station_config(1, "rect", canvas_shape=(16, 16)),
        _station_config(2, "square", canvas_shape=(20, 20)),
    ]

    batch = preprocess_and_stack(samples, configs, as_torch=False)

    assert batch.image.shape == (2, 1, 20, 20)
    assert batch.valid_mask.shape == (2, 1, 20, 20)
    assert list(batch.station_id) == [1, 2]
    assert list(batch.geometry_id) == [0, 1]


def test_collate_batch_returns_model_input_and_labels() -> None:
    sample = _make_sample(np.zeros((8, 12), dtype=np.uint8), station_id=1, defect_tags=("scratch",))
    config = _station_config(1, "rect", canvas_shape=(16, 16))
    dataset = GreyInspectionDataset([sample], [config])

    batch = collate_batch([dataset[0]], as_torch=False)

    assert batch["accept_reject"] == [0]
    assert batch["defect_tags"] == [("scratch",)]
    assert batch["samples"] == [sample]
    assert batch["model_input"].image.shape == (1, 1, 16, 16)


def test_preprocess_sample_accepts_public_sample_contract() -> None:
    sample = _make_sample(np.zeros((4, 4), dtype=np.uint8), station_id=7, geometry_mode="rect")
    config = _station_config(7, "rect", canvas_shape=(8, 8))

    prepared = preprocess_sample(sample, config)

    assert prepared.original_shape == (4, 4)
    assert prepared.station_id == 7
    assert prepared.image_uint8.shape == (8, 8)


def test_folder_first_dataset_import_builds_internal_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(root, "station_01/ok/sample_a.npy", np.zeros((12, 18), dtype=np.uint8))

    index = build_dataset_manifest(root)

    manifest_dir = Path(index.manifest_path).parent
    assert Path(index.manifest_path).exists()
    assert Path(index.ontology_path).exists()
    assert Path(index.splits_path).exists()
    assert Path(index.hard_negatives_path).exists()
    assert (manifest_dir / "dataset_index.json").exists()


def test_optional_box_ingestion_is_preserved_in_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(
        root,
        "station_01/reject/sample_b.npy",
        np.zeros((20, 20), dtype=np.uint8),
        sidecar={
            "accept_reject": 1,
            "boxes": [
                {
                    "xyxy": [2, 3, 7, 9],
                    "defect_tag": "particle",
                    "confidence": 0.9,
                    "annotator": "qa",
                    "is_hard_case": True,
                }
            ],
            "mask_path": "mask_b.npy",
        },
    )

    index = ingest_dataset_manifest(root)
    records = load_dataset_manifest(index.manifest_path)

    assert len(records) == 1
    assert records[0].boxes[0].xyxy == (2, 3, 7, 9)
    assert records[0].boxes[0].defect_tag == "particle"
    assert records[0].mask_path == "mask_b.npy"


def test_hard_negative_generation_writes_reusable_subset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(root, "station_01/ok/clean_a.npy", np.zeros((8, 8), dtype=np.uint8))
    _write_image(root, "station_01/reject/bad_a.npy", np.zeros((8, 8), dtype=np.uint8), sidecar={"accept_reject": 1})
    index = build_dataset_manifest(root)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"sample_id": "station_01/ok/clean_a.npy", "reject_score": 0.87}) + "\n",
        encoding="utf-8",
    )

    output_path = build_hard_negative_subset(index.manifest_path, tmp_path / "hard_negatives.jsonl", predictions_path)
    rows = output_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(rows) == 1
    assert "clean_a.npy" in rows[0]


def test_dataset_split_generation_is_station_and_batch_stable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shared_sidecar = {"capture_day": "2026-03-23", "batch_id": "lot-01", "camera_id": "cam-a"}
    _write_image(root, "station_01/ok/sample_1.npy", np.zeros((8, 8), dtype=np.uint8), sidecar=shared_sidecar)
    _write_image(root, "station_01/ok/sample_2.npy", np.zeros((8, 8), dtype=np.uint8), sidecar=shared_sidecar)
    index = build_dataset_manifest(root)

    first = build_dataset_splits(index.manifest_path, seed=23)
    second = build_dataset_splits(index.manifest_path, seed=23)
    records = load_dataset_manifest(index.manifest_path)

    assert first["assignments"] == second["assignments"]
    assert records[0].split == records[1].split
