from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from greymodel import load_dataset_manifest, run_finetune_stage, run_pretraining_stage
from greymodel.data import load_uint8_grayscale, save_dataset_manifest


class _FakeImage:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.uint8)
        self.mode = "RGB" if self._array.ndim == 3 else "L"

    def convert(self, mode: str) -> "_FakeImage":
        if mode != "L":
            raise ValueError("Only grayscale conversion is supported in the test fake.")
        if self._array.ndim == 2:
            return self
        if self._array.ndim == 3 and self._array.shape[-1] == 3:
            gray = np.mean(self._array.astype(np.float32), axis=-1).clip(0, 255).astype(np.uint8)
            return _FakeImage(gray)
        raise ValueError("Unsupported fake image shape: %s" % (self._array.shape,))

    def __array__(self, dtype=None):
        return self._array.astype(dtype) if dtype is not None else self._array


class _FakeSplit(list):
    pass


def _make_rows() -> dict[str, _FakeSplit]:
    return {
        "train": _FakeSplit(
            [
                {
                    "image": _FakeImage(np.stack([np.zeros((12, 18), dtype=np.uint8)] * 3, axis=-1)),
                    "label": 0,
                    "sample_id": "hf-train-001",
                    "station_id": "station-01",
                    "product_family": "vial",
                    "geometry_mode": "rect",
                    "capture_metadata": {"capture_day": "2026-03-23", "batch_id": "hf-a", "camera_id": "cam-1"},
                    "split": "train",
                },
                {
                    "image": _FakeImage(np.stack([np.full((12, 18), 255, dtype=np.uint8)] * 3, axis=-1)),
                    "label": 1,
                    "sample_id": "hf-train-002",
                    "station_id": "station-01",
                    "product_family": "vial",
                    "geometry_mode": "rect",
                    "capture_metadata": {"capture_day": "2026-03-23", "batch_id": "hf-a", "camera_id": "cam-1"},
                    "split": "train",
                },
            ]
        ),
        "validation": _FakeSplit(
            [
                {
                    "image": _FakeImage(np.stack([np.full((10, 14), 180, dtype=np.uint8)] * 3, axis=-1)),
                    "label": 0,
                    "sample_id": "hf-val-001",
                    "station_id": "station-02",
                    "product_family": "syringe",
                    "geometry_mode": "rect",
                    "capture_metadata": {"capture_day": "2026-03-23", "batch_id": "hf-b", "camera_id": "cam-2"},
                    "split": "validation",
                }
            ]
        ),
        "test": _FakeSplit(
            [
                {
                    "image": _FakeImage(np.stack([np.full((10, 14), 90, dtype=np.uint8)] * 3, axis=-1)),
                    "label": 1,
                    "sample_id": "hf-test-001",
                    "station_id": "station-02",
                    "product_family": "syringe",
                    "geometry_mode": "rect",
                    "capture_metadata": {"capture_day": "2026-03-23", "batch_id": "hf-b", "camera_id": "cam-2"},
                    "split": "test",
                }
            ]
        ),
    }


def _install_fake_datasets_module(monkeypatch: pytest.MonkeyPatch):
    rows_by_split = _make_rows()
    calls: list[dict[str, object]] = []
    fake_module = types.ModuleType("datasets")
    fake_module.DownloadConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    def _load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        split = kwargs.get("split")
        if split is not None:
            return rows_by_split[split]
        return rows_by_split

    fake_module.load_dataset = _load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return rows_by_split, calls


def _resolve_importer():
    import greymodel
    import greymodel.cli as cli_module
    import greymodel.data as data_module

    candidate_names = (
        "import_huggingface_dataset",
        "import_hf_dataset",
        "build_hf_dataset_manifest",
        "build_dataset_manifest_from_huggingface",
    )
    for module in (greymodel, data_module, cli_module):
        for name in candidate_names:
            if hasattr(module, name):
                return getattr(module, name)
    return None


def _call_importer(
    importer,
    *,
    dataset_name: str,
    output_dir: Path,
    split: str | None = None,
    data_dir: str | None = None,
    max_records: int | None = None,
):
    signature = inspect.signature(importer)
    kwargs = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_name,
        "repo_id": dataset_name,
        "name": dataset_name,
        "output_dir": output_dir,
        "output_path": output_dir,
        "split": split,
        "splits": ("train", "validation", "test"),
        "data_dir": data_dir,
        "max_records": max_records,
        "source_dataset": "huggingface_public",
        "strict_grayscale": False,
    }
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}
    positional = []
    required_positionals = [
        param
        for param in signature.parameters.values()
        if param.default is inspect._empty
        and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required_positionals and "dataset_name" not in filtered_kwargs:
        positional.append(dataset_name)
    return importer(*positional, **filtered_kwargs)


def _materialize_output_path(result) -> Path:
    if isinstance(result, Path):
        return result
    for attribute in ("manifest_path", "output_path", "path"):
        value = getattr(result, attribute, None)
        if value is not None:
            return Path(value)
    raise AssertionError("Importer did not return a manifest or output path.")


def _build_production_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "production"
    for index, fill in enumerate((0, 255, 0)):
        image = np.full((16, 16), fill, dtype=np.uint8)
        sample_path = root / "station_01" / ("ok" if index != 1 else "reject") / ("sample_%d.npy" % index)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(sample_path, image)
        if index == 1:
            sample_path.with_suffix(".json").write_text(
                "{\"accept_reject\": 1, \"station_id\": \"station-01\", \"geometry_mode\": \"rect\", \"capture_metadata\": {\"capture_day\": \"2026-03-23\", \"batch_id\": \"prod-a\", \"camera_id\": \"cam-1\"}}",
                encoding="utf-8",
            )
        else:
            sample_path.with_suffix(".json").write_text(
                "{\"station_id\": \"station-01\", \"geometry_mode\": \"rect\", \"capture_metadata\": {\"capture_day\": \"2026-03-23\", \"batch_id\": \"prod-a\", \"camera_id\": \"cam-1\"}}",
                encoding="utf-8",
            )
    index = __import__("greymodel").build_dataset_manifest(root)
    records = load_dataset_manifest(index.manifest_path)
    for record in records:
        if record.sample_id.endswith("sample_1.npy"):
            record.split = "val"
        else:
            record.split = "train"
    save_dataset_manifest(records, index.manifest_path)
    return Path(index.manifest_path)


def test_huggingface_import_materializes_manifest_and_preserves_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows_by_split, calls = _install_fake_datasets_module(monkeypatch)
    importer = _resolve_importer()
    if importer is None:
        pytest.xfail("Hugging Face import path is not exposed yet.")

    output_dir = tmp_path / "hf_import"
    result = _call_importer(importer, dataset_name="fake/public-grayscale", output_dir=output_dir)
    manifest_path = _materialize_output_path(result)
    index_path = manifest_path.with_name("dataset_index.json")

    assert manifest_path.exists()
    assert index_path.exists()

    records = load_dataset_manifest(manifest_path)
    assert {record.split for record in records} == {"train", "val", "test"}
    assert {record.sample_id for record in records} == {row["sample_id"] for rows in rows_by_split.values() for row in rows}
    assert calls
    assert any(call["args"] or call["kwargs"] for call in calls)
    assert all(load_uint8_grayscale(Path(record.image_path)).ndim == 2 for record in records)
    assert all(load_uint8_grayscale(Path(record.image_path)).dtype == np.uint8 for record in records)


def test_huggingface_import_accepts_data_dir_and_record_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _install_fake_datasets_module(monkeypatch)
    importer = _resolve_importer()
    if importer is None:
        pytest.xfail("Hugging Face import path is not exposed yet.")

    output_dir = tmp_path / "hf_import"
    result = _call_importer(
        importer,
        dataset_name="fake/public-grayscale",
        output_dir=output_dir,
        data_dir="DS-DAGM/image",
        max_records=2,
    )
    manifest_path = _materialize_output_path(result)
    records = load_dataset_manifest(manifest_path)

    assert len(records) == 2
    assert calls
    assert all(call["kwargs"].get("data_dir") == "DS-DAGM/image" for call in calls)


def test_huggingface_import_retries_http_429_and_uses_token_and_cache_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_by_split = _make_rows()
    calls: list[dict[str, object]] = []
    fake_module = types.ModuleType("datasets")
    fake_module.DownloadConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    def _load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if len(calls) == 1:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return rows_by_split

    fake_module.load_dataset = _load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr("greymodel.data.time.sleep", lambda *_args, **_kwargs: None)

    importer = _resolve_importer()
    if importer is None:
        pytest.xfail("Hugging Face import path is not exposed yet.")

    output_dir = tmp_path / "hf_retry_import"
    result = importer(
        dataset_name="fake/public-grayscale",
        output_dir=output_dir,
        strict_grayscale=False,
        local_files_only=True,
        max_retries=2,
        retry_backoff_seconds=0.01,
    )
    manifest_path = _materialize_output_path(result)
    records = load_dataset_manifest(manifest_path)

    assert manifest_path.exists()
    assert len(records) == 4
    assert len(calls) == 2
    assert calls[0]["kwargs"]["token"] == "hf_test_token"
    assert calls[0]["kwargs"]["download_config"].local_files_only is True
    assert calls[0]["kwargs"]["download_config"].max_retries == 2


def test_pretrained_hf_checkpoint_can_finetune_production_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets_module(monkeypatch)
    importer = _resolve_importer()
    if importer is None:
        pytest.xfail("Hugging Face import path is not exposed yet.")

    hf_output_dir = tmp_path / "hf_import"
    imported_result = _call_importer(importer, dataset_name="fake/public-grayscale", output_dir=hf_output_dir)
    imported_manifest = _materialize_output_path(imported_result)
    production_manifest = _build_production_manifest(tmp_path)

    pretrain = run_pretraining_stage(
        manifest_path=imported_manifest,
        index_path=imported_manifest.with_name("dataset_index.json"),
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
    )
    finetune = run_finetune_stage(
        manifest_path=production_manifest,
        index_path=production_manifest.with_name("dataset_index.json"),
        checkpoint_path=pretrain.checkpoint_path,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
    )

    assert pretrain.checkpoint_path is not None and pretrain.checkpoint_path.exists()
    assert finetune.checkpoint_path is not None and finetune.checkpoint_path.exists()
