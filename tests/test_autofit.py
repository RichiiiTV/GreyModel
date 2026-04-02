from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from greymodel import GreyModel, cli_main
from greymodel.autofit import resolve_autofit_plan
from greymodel.model_profiles import ModelProfile


def _write_sample(root: Path, relative_path: str, image: np.ndarray, sidecar: dict | None = None) -> Path:
    image_path = root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    if sidecar is not None:
        image_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return image_path


def _build_labeled_folder(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _write_sample(
        root,
        "station_01/ok/sample_1.npy",
        np.zeros((16, 16), dtype=np.uint8),
        sidecar={
            "station_id": "station-01",
            "geometry_mode": "rect",
            "capture_metadata": {"batch_id": "a", "capture_day": "2026-04-02", "camera_id": "cam-1"},
        },
    )
    _write_sample(
        root,
        "station_02/reject/sample_2.npy",
        np.full((16, 16), 255, dtype=np.uint8),
        sidecar={
            "station_id": "station-02",
            "geometry_mode": "rect",
            "accept_reject": 1,
            "defect_tags": ["particle"],
            "capture_metadata": {"batch_id": "b", "capture_day": "2026-04-03", "camera_id": "cam-2"},
        },
    )
    _write_sample(
        root,
        "station_03/ok/sample_3.npy",
        np.zeros((16, 16), dtype=np.uint8),
        sidecar={
            "station_id": "station-03",
            "geometry_mode": "rect",
            "capture_metadata": {"batch_id": "c", "capture_day": "2026-04-04", "camera_id": "cam-3"},
        },
    )
    return root


def test_autofit_plan_builds_folder_bundle_and_resolves_native_profile(tmp_path: Path) -> None:
    data_root = _build_labeled_folder(tmp_path)

    plan = resolve_autofit_plan(
        data=data_root,
        model="lite",
        run_root=tmp_path / "runs",
        execution="local",
    )

    assert plan.data_kind == "folder"
    assert Path(plan.manifest_path).exists()
    assert Path(plan.index_path).exists()
    assert plan.variant == "lite"
    assert plan.execution_backend == "local"
    assert Path(plan.run_dir).exists()


def test_greymodel_fit_runs_autofit_and_writes_human_summaries(tmp_path: Path) -> None:
    data_root = _build_labeled_folder(tmp_path)

    model = GreyModel("lite", home=tmp_path / "home")
    result = model.fit(
        data_root,
        model="lite",
        execution="local",
        run_root=tmp_path / "runs",
        overrides={"epochs": 1, "batch_size": 1, "learning_rate": 1e-3, "num_workers": 0, "precision": "fp32"},
    )

    assert result.status == "completed"
    assert result.execution_backend == "local"
    assert Path(result.run_dir).exists()
    assert Path(result.summary_path).exists()
    assert Path(result.markdown_summary_path).exists()
    assert Path(result.report_path).exists()
    assert Path(result.best_checkpoint_path).exists()
    assert "overall" in dict(result.metrics)
    assert Path(result.log_path).read_text(encoding="utf-8")


def test_cli_auto_plan_and_fit_return_structured_payloads(tmp_path: Path) -> None:
    data_root = _build_labeled_folder(tmp_path)

    plan_payload = cli_main(
        [
            "auto",
            "plan",
            "--data",
            str(data_root),
            "--model",
            "lite",
            "--run-root",
            str(tmp_path / "runs"),
            "--execution",
            "local",
        ]
    )
    assert plan_payload["variant"] == "lite"
    assert Path(plan_payload["manifest_path"]).exists()

    fit_payload = cli_main(
        [
            "auto",
            "fit",
            "--data",
            str(data_root),
            "--model",
            "lite",
            "--run-root",
            str(tmp_path / "runs_cli"),
            "--execution",
            "local",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--precision",
            "fp32",
        ]
    )
    assert fit_payload["status"] == "completed"
    assert Path(fit_payload["summary_path"]).exists()


def test_autofit_rejects_non_native_training_profiles(tmp_path: Path) -> None:
    data_root = _build_labeled_folder(tmp_path)

    with pytest.raises(ValueError, match="supports native GreyModel profiles only"):
        resolve_autofit_plan(
            data=data_root,
            model=ModelProfile(
                profile_id="hf_det",
                backend_family="huggingface",
                task_type="detection",
            ),
            run_root=tmp_path / "runs",
            execution="local",
        )
