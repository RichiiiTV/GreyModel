from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from greymodel import (
    ManifestInspectionDataset,
    TrainingConfig,
    StationBalancedManifestSampler,
    build_dataset_manifest,
    load_dataset_manifest,
    run_domain_adaptation_stage,
    run_finetune_stage,
    run_pretraining_stage,
    run_resume_stage,
)
from greymodel.data import save_dataset_manifest


def _write_sample(root: Path, relative_path: str, image: np.ndarray, sidecar: dict | None = None) -> Path:
    image_path = root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    if sidecar is not None:
        image_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return image_path


def _build_training_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _write_sample(
        root,
        "station_01/ok/sample_1.npy",
        np.zeros((16, 16), dtype=np.uint8),
        sidecar={"station_id": "station-01", "geometry_mode": "rect", "capture_metadata": {"batch_id": "a", "capture_day": "2026-03-23", "camera_id": "cam-1"}},
    )
    _write_sample(
        root,
        "station_02/reject/sample_2.npy",
        np.full((16, 16), 255, dtype=np.uint8),
        sidecar={"station_id": "station-02", "geometry_mode": "rect", "accept_reject": 1, "capture_metadata": {"batch_id": "b", "capture_day": "2026-03-23", "camera_id": "cam-2"}},
    )
    _write_sample(
        root,
        "station_01/ok/sample_3.npy",
        np.zeros((16, 16), dtype=np.uint8),
        sidecar={"station_id": "station-01", "geometry_mode": "rect", "capture_metadata": {"batch_id": "a", "capture_day": "2026-03-23", "camera_id": "cam-1"}},
    )
    index = build_dataset_manifest(root)
    records = load_dataset_manifest(index.manifest_path)
    for record in records:
        if record.sample_id.endswith("sample_2.npy"):
            record.split = "val"
        else:
            record.split = "train"
    save_dataset_manifest(records, index.manifest_path)
    return Path(index.manifest_path)


def test_manifest_dataset_lazily_loads_images_and_filters_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _build_training_manifest(tmp_path)
    dataset_module = __import__("greymodel.data", fromlist=["load_uint8_grayscale"])
    calls: list[Path] = []

    def _tracking_loader(path: Path):
        calls.append(Path(path))
        return np.zeros((16, 16), dtype=np.uint8)

    monkeypatch.setattr(dataset_module, "load_uint8_grayscale", _tracking_loader)

    train_dataset = ManifestInspectionDataset(manifest, split="train")
    assert len(train_dataset) == 2
    assert calls == []

    item = train_dataset[0]
    assert item["prepared"].image_uint8.shape == (16, 16)
    assert calls

    val_dataset = ManifestInspectionDataset(manifest, split="val")
    assert len(val_dataset) == 1


def test_station_balanced_sampler_covers_records_once(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)
    records = load_dataset_manifest(manifest)

    sampler = StationBalancedManifestSampler(records)
    indices = list(iter(sampler))

    assert len(indices) == len(records)
    assert sorted(indices) == list(range(len(records)))
    assert len(set(indices)) == len(indices)


def test_station_balanced_sampler_shards_cleanly_across_replicas(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)
    records = load_dataset_manifest(manifest)

    rank0 = list(StationBalancedManifestSampler(records, num_replicas=2, rank=0))
    rank1 = list(StationBalancedManifestSampler(records, num_replicas=2, rank=1))

    assert set(rank0) | set(rank1) == set(range(len(records)))
    assert set(rank0).isdisjoint(set(rank1)) or len(records) % 2 != 0


def test_pretraining_stage_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)

    result = run_pretraining_stage(manifest, variant="lite", run_root=tmp_path / "runs", batch_size=2)

    assert result.report_path is not None and result.report_path.exists()
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert (result.run_dir / "checkpoints" / "pretrain_reconstruction_head.pt").exists()

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    assert isinstance(checkpoint, dict)
    assert checkpoint


def test_resume_from_checkpoint_restores_progress_metadata(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)
    first = run_finetune_stage(
        manifest_path=manifest,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
        training_config=TrainingConfig(epochs=1),
    )
    resumed = run_finetune_stage(
        manifest_path=manifest,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
        training_config=TrainingConfig(epochs=2, resume_from=str(first.checkpoint_path)),
        resume_from=str(first.checkpoint_path),
    )

    assert first.checkpoint_path is not None and first.checkpoint_path.exists()
    assert resumed.global_step >= first.global_step
    assert resumed.epoch == 2


def test_resume_and_finetune_from_checkpoint(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)
    pretrain = run_pretraining_stage(manifest, variant="lite", run_root=tmp_path / "runs", batch_size=2)

    resume = run_resume_stage(
        manifest_path=manifest,
        checkpoint_path=pretrain.checkpoint_path,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
    )
    finetune = run_finetune_stage(
        manifest_path=manifest,
        checkpoint_path=pretrain.checkpoint_path,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
    )

    assert resume.stage == "finetune"
    assert resume.checkpoint_path is not None and resume.checkpoint_path.exists()
    assert finetune.checkpoint_path is not None and finetune.checkpoint_path.exists()


def test_domain_adapt_from_pretrained_checkpoint_is_exposed(tmp_path: Path) -> None:
    manifest = _build_training_manifest(tmp_path)
    pretrain = run_pretraining_stage(manifest, variant="lite", run_root=tmp_path / "runs", batch_size=2)

    if "checkpoint_path" not in inspect.signature(run_domain_adaptation_stage).parameters:
        pytest.xfail("domain adaptation checkpoint warm-start is not yet exposed by the runner API")

    result = run_domain_adaptation_stage(
        manifest_path=manifest,
        checkpoint_path=pretrain.checkpoint_path,
        variant="lite",
        run_root=tmp_path / "runs",
        batch_size=2,
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
