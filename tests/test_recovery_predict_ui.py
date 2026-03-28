from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from greymodel import (
    build_dataset_manifest,
    cli_main,
    list_failure_records,
    list_run_statuses,
    run_prediction_stage,
    run_pretraining_stage,
)
from greymodel.ui import UIExecutionDefaults, build_streamlit_command, launch_streamlit_ui, resolve_ui_proxy_configuration
from greymodel.ui_app import _launch_managed_job, collect_ui_state, render_app


def _write_sample(root: Path, relative_path: str, image: np.ndarray, sidecar: dict | None = None) -> Path:
    image_path = root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    if sidecar is not None:
        image_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return image_path


def _build_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _write_sample(
        root,
        "station_01/ok/good.npy",
        np.zeros((32, 32), dtype=np.uint8),
        sidecar={"station_id": "station-01", "geometry_mode": "rect"},
    )
    _write_sample(
        root,
        "station_01/ok/broken.npy",
        np.zeros((32, 32), dtype=np.uint8),
        sidecar={"station_id": "station-01", "geometry_mode": "rect"},
    )
    index = build_dataset_manifest(root)
    return Path(index.manifest_path)


def test_training_failure_writes_failure_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _build_manifest(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("greymodel.runners.compute_masked_pretrain_objective", _boom)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_pretraining_stage(manifest, variant="lite", run_root=tmp_path / "runs", batch_size=1)

    failures = list_failure_records(tmp_path / "runs")
    assert failures
    assert failures[0].stage == "pretrain"
    statuses = list_run_statuses(tmp_path / "runs")
    assert statuses
    assert statuses[0].status == "failed"


def test_prediction_stage_quarantines_failing_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _build_manifest(tmp_path)

    original_loader = __import__("greymodel.evaluation", fromlist=["load_uint8_grayscale"]).load_uint8_grayscale

    def _loader(path: Path):
        if Path(path).name == "broken.npy":
            raise RuntimeError("broken sample")
        return original_loader(path)

    monkeypatch.setattr("greymodel.evaluation.load_uint8_grayscale", _loader)

    result = run_prediction_stage(manifest, variant="lite", run_root=tmp_path / "runs", evidence_policy="none")

    assert result.report_path is not None and result.report_path.exists()
    predictions_path = Path(result.extra_paths["predictions_path"])
    assert predictions_path.exists()
    prediction_rows = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(prediction_rows) == 1

    failures = list_failure_records(tmp_path / "runs")
    assert failures
    assert failures[0].stage == "predict"
    assert failures[0].offending_sample_ids == ("station_01/ok/broken.npy",)

    statuses = list_run_statuses(tmp_path / "runs")
    assert statuses
    assert statuses[0].status == "completed_with_failures"


class _FakeSidebar:
    def __init__(self, page: str) -> None:
        self._page = page

    def title(self, *args, **kwargs):
        return None

    def radio(self, *args, **kwargs):
        return self._page

    def write(self, *args, **kwargs):
        return None


class _FakeStreamlit(types.SimpleNamespace):
    def __init__(self, page: str) -> None:
        super().__init__()
        self.sidebar = _FakeSidebar(page)

    def set_page_config(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def image(self, *args, **kwargs):
        return None

    def selectbox(self, _label, options, **kwargs):
        if isinstance(options, Path):
            return options
        if not options:
            return None
        return list(options)[0]

    def text_input(self, _label, value="", **kwargs):
        return value

    def number_input(self, _label, min_value=0, value=0, **kwargs):
        return value

    def code(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def success(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None


def test_ui_dry_run_and_render_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    _build_manifest(data_root)

    payload = cli_main(["ui", "--run-root", str(tmp_path / "runs"), "--dataset-root", str(data_root), "--dry-run"])
    assert "streamlit" in " ".join(payload["command"])
    assert payload["proxy_mode"] == "off"
    assert payload["local_url"] == payload["proxy_url"]

    state = collect_ui_state(tmp_path / "runs", data_root)
    assert "datasets" in state
    assert state["datasets"]

    for page in ("Overview", "Datasets", "Train", "Predict", "Evaluate", "Explain", "Failures"):
        fake_st = _FakeStreamlit(page)
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        render_app(run_root=tmp_path / "runs", data_root=data_root)


def test_ui_dry_run_carries_slurm_defaults(tmp_path: Path) -> None:
    payload = cli_main(
        [
            "ui",
            "--run-root",
            str(tmp_path / "runs"),
            "--data-root",
            str(tmp_path / "data"),
            "--dry-run",
            "--default-execution-backend",
            "slurm",
            "--slurm-cpus",
            "8",
            "--slurm-mem",
            "50G",
            "--slurm-gres",
            "gpu:8",
            "--slurm-partition",
            "batch_gpu",
            "--slurm-queue",
            "3h",
            "--slurm-nproc-per-node",
            "8",
            "--proxy-mode",
            "jupyter_service",
            "--base-url-path",
            "/services/greymodel/",
        ]
    )

    assert payload["default_execution_backend"] == "slurm"
    command = payload["command"]
    command_text = " ".join(command)
    assert "--default-execution-backend" in command_text
    assert "--slurm-partition" in command_text
    assert "batch_gpu" in command_text
    assert "--slurm-queue" in command_text
    assert "3h" in command_text
    assert "--server.baseUrlPath=services/greymodel" in command
    assert command.index("--server.baseUrlPath=services/greymodel") < command.index("--")


def test_ui_slurm_submission_writes_job_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _fake_run(command, cwd=None, capture_output=False, text=False, check=False):
        calls.append({"command": command, "cwd": cwd})
        return types.SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr("greymodel.ui_app.subprocess.run", _fake_run)

    payload = _launch_managed_job(
        ["train", "pretrain", "--manifest", "data/manifest.jsonl", "--variant", "base", "--run-root", str(tmp_path / "runs")],
        cwd=tmp_path,
        run_root=tmp_path / "runs",
        kind="train",
        execution_backend="slurm",
        nproc_per_node=8,
        execution_defaults=UIExecutionDefaults(
            execution_backend="slurm",
            slurm_cpus=8,
            slurm_mem="50G",
            slurm_gres="gpu:8",
            slurm_partition="batch_gpu",
            slurm_queue="3h",
            slurm_nproc_per_node=8,
            slurm_python="/usr/bin/python3",
        ),
    )

    assert payload["backend"] == "slurm"
    assert payload["job_id"] == "12345"
    assert Path(payload["metadata_path"]).exists()
    assert calls
    submit_command = calls[0]["command"]
    assert submit_command[0] == "sbatch"
    assert "--parsable" in submit_command
    assert "-p" in submit_command and "batch_gpu" in submit_command
    assert "-q" in submit_command and "3h" in submit_command
    assert "--wrap" in submit_command
    assert "torch.distributed.run" in " ".join(payload["command"])


def test_resolve_ui_proxy_auto_port_and_service_modes() -> None:
    off = resolve_ui_proxy_configuration(proxy_mode="auto", env={})
    assert off.proxy_mode == "off"
    assert off.bind_address == "127.0.0.1"
    assert off.base_url_path == ""
    assert off.proxy_url == off.local_url

    notebook = resolve_ui_proxy_configuration(
        proxy_mode="auto",
        bind_port=8899,
        env={"JPY_PARENT_PID": "1", "JUPYTERHUB_SERVICE_PREFIX": "/user/ricardo/"},
    )
    assert notebook.proxy_mode == "jupyter_port"
    assert notebook.bind_address == "0.0.0.0"
    assert notebook.base_url_path == ""
    assert notebook.proxy_url == "/user/ricardo/proxy/8899/"

    service = resolve_ui_proxy_configuration(
        proxy_mode="auto",
        env={
            "JUPYTERHUB_SERVICE_URL": "http://127.0.0.1:9911",
            "JUPYTERHUB_SERVICE_PREFIX": "/services/greymodel/",
        },
    )
    assert service.proxy_mode == "jupyter_service"
    assert service.bind_address == "127.0.0.1"
    assert service.bind_port == 9911
    assert service.base_url_path == "services/greymodel"
    assert service.proxy_url == "/services/greymodel/"


def test_build_streamlit_command_emits_hpc_proxy_flags() -> None:
    command = build_streamlit_command(
        proxy_mode="jupyter_service",
        bind_address="0.0.0.0",
        bind_port=9001,
        base_url_path="/services/greymodel/",
        browser_server_address="cluster.example.org",
        browser_server_port=443,
    )

    assert "--server.address=0.0.0.0" in command
    assert "--server.port=9001" in command
    assert "--server.baseUrlPath=services/greymodel" in command
    assert "--browser.serverAddress=cluster.example.org" in command
    assert "--browser.serverPort=443" in command
    assert command.index("--server.baseUrlPath=services/greymodel") < command.index("--")


def test_ui_dry_run_prefers_explicit_proxy_override_and_prints_url(capsys: pytest.CaptureFixture[str]) -> None:
    payload = launch_streamlit_ui(
        dry_run=True,
        print_url=True,
        proxy_mode="auto",
        public_base_url="https://cluster.example.org/user/ricardo/",
        bind_port=9010,
        base_url_path="/services/explicit/",
        env={
            "JPY_PARENT_PID": "1",
            "JUPYTERHUB_SERVICE_URL": "http://127.0.0.1:9911",
            "JUPYTERHUB_SERVICE_PREFIX": "/services/env/",
        },
    )

    captured = capsys.readouterr()
    assert payload["proxy_mode"] == "jupyter_service"
    assert payload["base_url_path"] == "services/explicit"
    assert payload["proxy_url"] == "https://cluster.example.org/user/ricardo/services/explicit/"
    assert "https://cluster.example.org/user/ricardo/services/explicit/" in captured.out
