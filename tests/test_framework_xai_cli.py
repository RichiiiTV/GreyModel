from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from greymodel import BaseModel, ModelInput, StationConfig, build_dataset_manifest, cli_main


def _sample_input() -> ModelInput:
    return ModelInput(
        image_uint8=np.zeros((32, 32), dtype=np.uint8),
        station_id="station-01",
        geometry_mode="rect",
    )


def _station_config() -> StationConfig:
    return StationConfig(
        canvas_shape=(64, 64),
        station_id="station-01",
        geometry_mode="rect",
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=16,
        tile_stride=8,
        adapter_id="rect-a",
        reject_threshold=0.5,
    )


def _build_manifest(root: Path) -> Path:
    dataset_root = root / "dataset"
    sample_path = dataset_root / "station_01" / "ok" / "sample.npy"
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
    index = build_dataset_manifest(dataset_root)
    return Path(index.manifest_path)


def _install_fake_hf_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda *args, **kwargs: {
        "train": [{"image": np.zeros((12, 12), dtype=np.uint8), "sample_id": "hf-train-001"}],
        "validation": [{"image": np.full((12, 12), 255, dtype=np.uint8), "sample_id": "hf-val-001"}],
    }
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_torch_fx_can_trace_the_graph_export_adapter() -> None:
    from greymodel.graphing import GraphExportAdapter

    model = BaseModel(num_defect_families=4).backend.model
    model.eval()
    traced = torch.fx.symbolic_trace(GraphExportAdapter(model, image_shape=(64, 64)))

    assert traced is not None


def test_graph_export_writes_mermaid_and_json(tmp_path: Path) -> None:
    from greymodel import export_model_graph

    model = BaseModel(num_defect_families=4).backend.model
    artifacts = export_model_graph(model, tmp_path, image_shape=(64, 64))

    assert artifacts["json_path"].exists()
    assert artifacts["mermaid_path"].exists()

def test_explainability_bundle_contains_heatmap_and_attribution(tmp_path: Path) -> None:
    from greymodel import build_explanation_bundle

    bundle = build_explanation_bundle(BaseModel(num_defect_families=4), _sample_input(), _station_config(), tmp_path)
    assert bundle["image_path"].exists()
    assert bundle["heatmap_path"].exists()
    assert bundle["attribution_path"].exists()
    assert bundle["prediction_path"].exists()
    assert bundle["top_tiles_path"].exists()


def test_cli_smoke_dataset_train_eval_explain(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    index = manifest.with_name("dataset_index.json")
    graph_dir = tmp_path / "graphs"
    artifacts_dir = tmp_path / "artifacts"

    cli_main(["dataset", "validate", str(manifest)])
    cli_main(["train", "pretrain", "--manifest", str(manifest), "--index", str(index), "--run-root", str(artifacts_dir)])
    cli_main(["train", "domain-adapt", "--manifest", str(manifest), "--index", str(index), "--run-root", str(artifacts_dir)])
    cli_main(["train", "finetune", "--manifest", str(manifest), "--index", str(index), "--run-root", str(artifacts_dir)])
    cli_main(["train", "calibrate", "--manifest", str(manifest), "--index", str(index), "--run-root", str(artifacts_dir)])
    cli_main(["eval", "benchmark", "--manifest", str(manifest), "--index", str(index)])
    cli_main(["explain", "graph", "--output-dir", str(graph_dir), "--height", "64", "--width", "64"])
    cli_main(["explain", "sample", "--manifest", str(manifest), "--index", str(index), "--output-dir", str(tmp_path / "sample_bundle")])

    assert (graph_dir / "model_graph.mmd").exists()
    assert (graph_dir / "model_graph.json").exists()
    assert (artifacts_dir / "finetune-base" / "reports" / "finetune_report.json").exists()


def test_cli_can_build_huggingface_manifest_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_hf_module(monkeypatch)
    output_dir = tmp_path / "hf_bundle"

    cli_main(["dataset", "build-hf", "--dataset-name", "fake/public-grayscale", "--output-dir", str(output_dir)])

    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "dataset_index.json").exists()


def test_model_output_contract_still_has_top_tiles_and_heatmap() -> None:
    output = BaseModel(num_defect_families=4).forward(_sample_input(), _station_config())

    assert output.defect_heatmap.shape == (64, 64)
    assert output.top_tiles.shape[1] == 5
    assert output.metadata["backend"] in {"numpy", "torch"}
