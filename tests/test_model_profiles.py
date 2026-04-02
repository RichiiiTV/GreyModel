from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from greymodel import ModelInput, ModelProfile, StationConfig, cli_main
from greymodel.data import build_dataset_manifest, load_dataset_manifest
from greymodel.explainability import build_explanation_bundle
from greymodel.hf_backends import build_huggingface_model_backend
from greymodel.model_profiles import load_model_profile, list_model_profiles, save_model_profile


def _station_config() -> StationConfig:
    return StationConfig(
        canvas_shape=(64, 64),
        station_id="station-01",
        geometry_mode="rect",
        pad_value=0,
        normalization_mean=0.0,
        normalization_std=1.0,
        tile_size=16,
        tile_stride=8,
        adapter_id="rect-a",
        reject_threshold=0.5,
    )


def _sample_input(fill: int = 0) -> ModelInput:
    return ModelInput(
        image_uint8=np.full((32, 32), fill, dtype=np.uint8),
        station_id="station-01",
        geometry_mode="rect",
    )


def _build_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    sample_path = root / "station_01" / "ok" / "sample.npy"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sample_path, np.zeros((32, 32), dtype=np.uint8))
    sample_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "station_id": "station-01",
                "product_family": "vial",
                "geometry_mode": "rect",
                "capture_metadata": {"capture_day": "2026-03-23", "batch_id": "lot-01", "camera_id": "cam-a"},
            }
        ),
        encoding="utf-8",
    )
    index = build_dataset_manifest(root)
    return Path(index.manifest_path)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    fake_module = types.ModuleType("transformers")

    class _BaseProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def _shape(self, images):
            array = np.asarray(images)
            return int(array.shape[0]), int(array.shape[1])

        def __call__(self, images, return_tensors="pt"):
            height, width = self._shape(images)
            return {"pixel_values": torch.ones((1, 3, height, width), dtype=torch.float32)}

    class _ClassificationProcessor(_BaseProcessor):
        pass

    class _DetectionProcessor(_BaseProcessor):
        def post_process_object_detection(self, outputs, threshold=0.0, target_sizes=None):
            return [
                {
                    "scores": torch.tensor([0.95, 0.3], dtype=torch.float32),
                    "labels": torch.tensor([1, 2], dtype=torch.long),
                    "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0], [24.0, 24.0, 40.0, 40.0]], dtype=torch.float32),
                }
            ]

    class _SegmentationProcessor(_BaseProcessor):
        def post_process_semantic_segmentation(self, outputs, target_sizes=None):
            mask = torch.zeros((64, 64), dtype=torch.float32)
            mask[16:48, 16:48] = 1.0
            return [mask]

    class _ClassificationModel:
        config = types.SimpleNamespace(id2label={0: "good", 1: "scratch", 2: "bad"})

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=torch.tensor([[0.1, 1.0, 3.0]], dtype=torch.float32))

    class _DetectionModel:
        config = types.SimpleNamespace(id2label={0: "background", 1: "scratch", 2: "leak"})

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return types.SimpleNamespace(
                logits=torch.tensor([[[0.1, 0.9, 0.0], [0.1, 0.3, 0.6]]], dtype=torch.float32),
                pred_boxes=torch.tensor([[[4.0, 4.0, 20.0, 20.0], [24.0, 24.0, 40.0, 40.0]]], dtype=torch.float32),
            )

    class _SegmentationModel:
        config = types.SimpleNamespace(id2label={0: "background", 1: "scratch"})

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, **kwargs):
            logits = torch.zeros((1, 2, 64, 64), dtype=torch.float32)
            logits[:, 1, 16:48, 16:48] = 4.0
            return types.SimpleNamespace(logits=logits)

    fake_module.AutoImageProcessor = _ClassificationProcessor
    fake_module.AutoModelForImageClassification = _ClassificationModel
    fake_module.AutoModelForObjectDetection = _DetectionModel
    fake_module.AutoModelForSemanticSegmentation = _SegmentationModel
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    # Distinct processor classes are exposed through the same module name so the adapter
    # can exercise the detection and segmentation code paths with the same fake backend.
    fake_module.AutoImageProcessor = _ClassificationProcessor
    fake_module.AutoFeatureExtractor = _ClassificationProcessor
    fake_module.AutoModelForImageClassification = _ClassificationModel
    fake_module.AutoModelForObjectDetection = _DetectionModel
    fake_module.AutoModelForSemanticSegmentation = _SegmentationModel


def test_model_profile_registry_roundtrip(tmp_path: Path) -> None:
    registry_root = tmp_path / "profiles"
    profile = ModelProfile(
        profile_id="hf_cls",
        backend_family="huggingface",
        task_type="classification",
        model_id="demo/model",
        label_mapping={"good": "good", "scratch": "scratch", "bad": "bad"},
        defect_family_mapping={"scratch": "scratch"},
        metadata={"cache_dir": "/tmp/cache"},
    )
    save_model_profile(profile, registry_root)

    listed = list_model_profiles(registry_root)
    loaded = load_model_profile("hf_cls", registry_root=registry_root)

    assert listed and listed[0].profile_id == "hf_cls"
    assert loaded.profile_id == "hf_cls"
    assert loaded.is_huggingface is True
    assert loaded.canonical_defect_families() == ("scratch",)


def test_hf_classification_adapter_normalizes_to_hierarchical_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    profile = ModelProfile(
        profile_id="hf_cls",
        backend_family="huggingface",
        task_type="classification",
        model_id="demo/model",
        label_mapping={"good": "good", "scratch": "scratch", "bad": "bad"},
        defect_family_mapping={"scratch": "scratch"},
        good_labels=("good",),
        bad_labels=("bad",),
    )
    backend = build_huggingface_model_backend(profile, num_defect_families=1, defect_families=("scratch",))
    output = backend.forward(_sample_input(fill=16), _station_config())

    assert output.reject_score > 0.5
    assert output.defect_heatmap.shape == (64, 64)
    assert output.top_tiles.shape[1] == 5
    assert output.metadata["backend"] == "huggingface"
    assert output.metadata["profile_id"] == "hf_cls"
    assert output.metadata["mode"] in {"transformers", "heuristic"}


def test_hf_detection_adapter_rasterizes_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    profile = ModelProfile(
        profile_id="hf_det",
        backend_family="huggingface",
        task_type="detection",
        model_id="demo/detector",
        defect_family_mapping={"scratch": "scratch", "leak": "leak"},
    )
    backend = build_huggingface_model_backend(profile, num_defect_families=2, defect_families=("scratch", "leak"))
    output = backend.forward(_sample_input(fill=24), _station_config())

    assert output.defect_heatmap.shape == (64, 64)
    assert output.top_tile_boxes.shape[1] == 4
    assert output.top_tiles.shape[1] == 5
    assert output.reject_score > 0.0


def test_hf_segmentation_adapter_rasterizes_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    profile = ModelProfile(
        profile_id="hf_seg",
        backend_family="huggingface",
        task_type="segmentation",
        model_id="demo/segmenter",
        defect_family_mapping={"scratch": "scratch"},
    )
    backend = build_huggingface_model_backend(profile, num_defect_families=1, defect_families=("scratch",))
    output = backend.forward(_sample_input(fill=8), _station_config())

    assert output.defect_heatmap.shape == (64, 64)
    assert float(np.asarray(output.defect_heatmap).max()) > 0.0
    assert output.top_tiles.shape[1] == 5


def test_native_fast_profile_supports_explanation_bundle(tmp_path: Path) -> None:
    profile = ModelProfile(
        profile_id="prod_fast_native",
        backend_family="native",
        task_type="native",
        metadata={"variant": "fast"},
    )
    backend = build_huggingface_model_backend(profile, num_defect_families=1, defect_families=("scratch",))
    bundle = build_explanation_bundle(backend, _sample_input(fill=12), _station_config(), tmp_path / "bundle")

    assert bundle["bundle_path"].exists()
    assert bundle["prediction_path"].exists()


def test_cli_models_registry_and_predict_can_use_hf_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    manifest = _build_manifest(tmp_path)
    registry_root = tmp_path / "profiles"
    run_root = tmp_path / "runs"

    cli_main(
        [
            "models",
            "register",
            "hf_cls",
            "--backend-family",
            "huggingface",
            "--task-type",
            "classification",
            "--model-id",
            "demo/model",
            "--label-mapping-json",
            "{\"good\": \"good\", \"scratch\": \"scratch\", \"bad\": \"bad\"}",
            "--defect-family-mapping-json",
            "{\"scratch\": \"scratch\"}",
            "--registry-root",
            str(registry_root),
        ]
    )

    listed = cli_main(["models", "list", "--registry-root", str(registry_root)])
    shown = cli_main(["models", "show", "hf_cls", "--registry-root", str(registry_root)])

    assert any(row["profile_id"] == "hf_cls" for row in listed)
    assert shown["profile_id"] == "hf_cls"

    cli_main(
        [
            "predict",
            "--manifest",
            str(manifest),
            "--index",
            str(manifest.with_name("dataset_index.json")),
            "--run-root",
            str(run_root),
            "--model-profile",
            "hf_cls",
            "--model-registry-root",
            str(registry_root),
        ]
    )

    predictions_path = run_root / "predict-base" / "predictions" / "predictions.jsonl"
    assert predictions_path.exists()
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["metadata"]["backend"] == "huggingface"
    assert rows[0]["metadata"]["profile_id"] == "hf_cls"
