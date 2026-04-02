from __future__ import annotations

from pathlib import Path

import numpy as np

from greymodel import GreyModel, cli_main, ensure_settings


def test_ensure_settings_creates_default_home_and_profiles(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "gm_home"
    monkeypatch.setenv("GREYMODEL_HOME", str(home))

    settings = ensure_settings()

    assert Path(settings.home) == home
    assert Path(settings.registry_root).exists()
    assert Path(settings.cache_root).exists()
    assert Path(settings.run_root).exists()
    assert Path(settings.data_root).exists()
    assert (Path(settings.registry_root) / "prod_fast_native.json").exists()
    assert (Path(settings.registry_root) / "review_native_lite.json").exists()


def test_env_doctor_reports_home_and_available_profiles(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "gm_home"
    monkeypatch.setenv("GREYMODEL_HOME", str(home))

    payload = cli_main(["env", "doctor"])

    assert payload["home"] == str(home)
    assert "prod_fast_native" in payload["available_profiles"]
    assert payload["dependencies"]["torch"]["installed"] is True


def test_models_list_uses_settings_registry_by_default(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "gm_home"
    monkeypatch.setenv("GREYMODEL_HOME", str(home))

    payload = cli_main(["models", "list"])

    assert any(row["profile_id"] == "prod_fast_native" for row in payload)
    assert any(row["profile_id"] == "review_native_lite" for row in payload)


def test_greymodel_high_level_runtime_predicts_and_explains(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "gm_home"
    monkeypatch.setenv("GREYMODEL_HOME", str(home))

    model = GreyModel("lite")
    prediction = model.predict(np.zeros((32, 32), dtype=np.uint8), station_id="station-01", geometry_mode="rect")
    bundle = model.explain(
        np.zeros((32, 32), dtype=np.uint8),
        tmp_path / "explain",
        station_id="station-01",
        geometry_mode="rect",
    )

    assert prediction.metadata["profile_id"] == "review_native_lite"
    assert prediction.primary_label in {"good", "bad"}
    assert bundle["image_path"].suffix == ".png"
    assert bundle["heatmap_path"].suffix == ".png"
    assert bundle["bundle_path"].exists()

    info = model.info()
    assert info["profile"]["profile_id"] == "review_native_lite"
    assert info["environment"]["active_profile"] == "prod_fast_native"
