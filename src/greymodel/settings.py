from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Optional

from .model_profiles import ensure_default_model_profiles, list_model_profiles
from .utils import ensure_dir, read_json, write_json


GREYMODEL_HOME_ENV = "GREYMODEL_HOME"
DEFAULT_HOME_DIRNAME = ".greymodel"
SETTINGS_FILENAME = "settings.json"


@dataclass(frozen=True)
class GreyModelSettings:
    home: str
    registry_root: str
    cache_root: str
    run_root: str
    data_root: str
    active_profile: str = "prod_fast_native"
    ui_host: str = "127.0.0.1"
    ui_port: int = 8501

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GreyModelSettings":
        data = dict(payload)
        defaults = default_settings(home=data.get("home"))
        merged = {**defaults.to_dict(), **data}
        return cls(**merged)


def default_greymodel_home(home: Path | str | None = None) -> Path:
    if home not in (None, ""):
        return Path(home).expanduser().resolve()
    env_home = os.getenv(GREYMODEL_HOME_ENV)
    if env_home not in (None, ""):
        return Path(env_home).expanduser().resolve()
    return (Path.home() / DEFAULT_HOME_DIRNAME).resolve()


def settings_path(home: Path | str | None = None) -> Path:
    return default_greymodel_home(home) / SETTINGS_FILENAME


def default_settings(home: Path | str | None = None) -> GreyModelSettings:
    resolved_home = default_greymodel_home(home)
    return GreyModelSettings(
        home=str(resolved_home),
        registry_root=str(resolved_home / "model_profiles"),
        cache_root=str(resolved_home / "cache"),
        run_root=str(resolved_home / "runs"),
        data_root=str(resolved_home / "data"),
        active_profile="prod_fast_native",
        ui_host="127.0.0.1",
        ui_port=8501,
    )


def load_settings(path: Path | str | None = None, *, home: Path | str | None = None) -> GreyModelSettings:
    resolved_path = Path(path) if path is not None else settings_path(home)
    if not resolved_path.exists():
        return default_settings(home=resolved_path.parent)
    return GreyModelSettings.from_dict(read_json(resolved_path))


def save_settings(settings: GreyModelSettings, path: Path | str | None = None) -> Path:
    resolved_path = Path(path) if path is not None else settings_path(settings.home)
    ensure_dir(resolved_path.parent)
    return write_json(resolved_path, settings.to_dict())


def ensure_settings(home: Path | str | None = None) -> GreyModelSettings:
    resolved_home = default_greymodel_home(home)
    settings_file = settings_path(resolved_home)
    settings = load_settings(settings_file, home=resolved_home)
    ensure_dir(Path(settings.home))
    ensure_dir(Path(settings.registry_root))
    ensure_dir(Path(settings.cache_root))
    ensure_dir(Path(settings.run_root))
    ensure_dir(Path(settings.data_root))
    ensure_default_model_profiles(settings.registry_root)
    if not settings_file.exists():
        save_settings(settings, settings_file)
    return settings


def _dependency_report(module_name: str, *, import_name: Optional[str] = None) -> dict[str, Any]:
    target = import_name or module_name
    installed = importlib.util.find_spec(target) is not None
    report: dict[str, Any] = {"installed": installed}
    if not installed:
        return report
    try:
        module = importlib.import_module(target)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    version = getattr(module, "__version__", None)
    if version is not None:
        report["version"] = str(version)
    if target == "torch":
        try:
            report["cuda_available"] = bool(module.cuda.is_available())
            report["cuda_device_count"] = int(module.cuda.device_count()) if module.cuda.is_available() else 0
            report["cuda_version"] = getattr(module.version, "cuda", None)
        except Exception as exc:
            report["cuda_error"] = str(exc)
    return report


def build_environment_report(home: Path | str | None = None) -> dict[str, Any]:
    settings = ensure_settings(home)
    profiles = [profile.profile_id for profile in list_model_profiles(settings.registry_root)]
    return {
        "home": settings.home,
        "settings_path": str(settings_path(settings.home)),
        "registry_root": settings.registry_root,
        "cache_root": settings.cache_root,
        "run_root": settings.run_root,
        "data_root": settings.data_root,
        "active_profile": settings.active_profile,
        "available_profiles": profiles,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "torch": _dependency_report("torch"),
            "onnxruntime": _dependency_report("onnxruntime"),
            "transformers": _dependency_report("transformers"),
            "datasets": _dependency_report("datasets"),
            "Pillow": _dependency_report("Pillow", import_name="PIL"),
            "streamlit": _dependency_report("streamlit"),
        },
    }
