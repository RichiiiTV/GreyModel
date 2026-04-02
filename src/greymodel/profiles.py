from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, Mapping

from .types import ModelProfile, WorkspaceConfig
from .utils import ensure_dir, read_json, write_json


DEFAULT_WORKSPACE_FILENAME = "workspace.json"


def _default_slurm_defaults() -> dict[str, object]:
    return {
        "execution_backend": "local",
        "slurm_cpus": 8,
        "slurm_mem": "50G",
        "slurm_gres": "gpu:8",
        "slurm_partition": "",
        "slurm_queue": "",
        "slurm_nproc_per_node": 8,
    }


def default_workspace_config(workspace_root: Path | str) -> WorkspaceConfig:
    workspace_root = Path(workspace_root)
    profiles_dir = workspace_root / "profiles"
    return WorkspaceConfig(
        workspace_root=str(workspace_root),
        run_root=str(workspace_root / "artifacts"),
        data_root=str(workspace_root / "data"),
        profiles_dir=str(profiles_dir),
        cache_root=str(workspace_root / ".cache" / "huggingface"),
        slurm_defaults=_default_slurm_defaults(),
        ui_preferences={"theme": "industrial_light"},
        metadata={"version": 1},
    )


def workspace_config_path(workspace_root: Path | str) -> Path:
    return Path(workspace_root) / DEFAULT_WORKSPACE_FILENAME


def save_workspace_config(config: WorkspaceConfig, path: Path | str | None = None) -> Path:
    resolved_path = Path(path) if path is not None else workspace_config_path(config.workspace_root)
    return write_json(resolved_path, asdict(config))


def load_workspace_config(path: Path | str) -> WorkspaceConfig:
    payload = read_json(Path(path))
    return WorkspaceConfig(**payload)


def ensure_workspace(workspace_root: Path | str) -> WorkspaceConfig:
    workspace_root = Path(workspace_root)
    ensure_dir(workspace_root)
    config_path = workspace_config_path(workspace_root)
    if config_path.exists():
        config = load_workspace_config(config_path)
    else:
        config = default_workspace_config(workspace_root)
        save_workspace_config(config, config_path)
    ensure_dir(Path(config.run_root))
    ensure_dir(Path(config.data_root))
    ensure_dir(Path(config.profiles_dir))
    ensure_dir(Path(config.cache_root or workspace_root / ".cache" / "huggingface"))
    ensure_default_model_profiles(config)
    return config


def _profile_path(profiles_dir: Path | str, profile_id: str) -> Path:
    return Path(profiles_dir) / ("%s.json" % str(profile_id))


def save_model_profile(profile: ModelProfile, profiles_dir: Path | str) -> Path:
    profiles_dir = ensure_dir(Path(profiles_dir))
    return write_json(_profile_path(profiles_dir, profile.profile_id), asdict(profile))


def _coerce_model_profile_payload(payload: Mapping[str, object]) -> dict[str, object]:
    data = dict(payload)
    schema_version = data.pop("schema_version", None)
    if schema_version is not None:
        # New registry/profile schema emitted by src/greymodel/model_profiles.py.
        backend_family = str(data.get("backend_family", "")).strip().lower()
        task_type = str(data.get("task_type", "")).strip().lower()
        variant = str(data.get("metadata", {}).get("variant", "")).strip().lower() if isinstance(data.get("metadata"), Mapping) else ""

        if backend_family == "huggingface":
            data["backend_family"] = "hf_%s" % (task_type or "classification")
            data["hf_model_id"] = data.pop("model_id", None)
            data["hf_revision"] = data.pop("revision", None)
            data["hf_local_path"] = data.pop("local_path", None)
            data["hf_processor_id"] = data["hf_model_id"]
            data["cache_policy"] = "online_cache"
            thresholds = {
                "reject_threshold": float(data.get("reject_threshold", 0.5)),
                "good_max": float(data.get("uncertainty_low", 0.35)),
                "bad_min": float(data.get("uncertainty_high", 0.65)),
            }
            data["thresholds"] = thresholds
            data["supports_training"] = task_type == "classification"
            data["supports_prediction"] = True
            data["supports_explain"] = True
            data["supports_benchmark"] = True
        elif backend_family == "native":
            if variant == "lite":
                data["backend_family"] = "native_review"
                data["variant"] = "lite"
            elif variant == "base":
                data["backend_family"] = "native_review"
                data["variant"] = "base"
            else:
                data["backend_family"] = "native_fast"
            data["checkpoint_path"] = data.pop("local_path", None)
            data["thresholds"] = {
                "reject_threshold": float(data.get("reject_threshold", 0.5)),
                "good_max": float(data.get("uncertainty_low", 0.35)),
                "bad_min": float(data.get("uncertainty_high", 0.65)),
            }

        data.setdefault("display_name", data.get("profile_id", "profile"))
        data.setdefault("cache_policy", "online_cache" if backend_family == "huggingface" else "local")
        data.setdefault("supports_training", backend_family in {"huggingface", "native"})
        data.setdefault("supports_prediction", True)
        data.setdefault("supports_explain", True)
        data.setdefault("supports_benchmark", True)

        allowed = {field.name for field in fields(ModelProfile)}
        return {key: value for key, value in data.items() if key in allowed}
    return data


def load_model_profile(path: Path | str) -> ModelProfile:
    payload = read_json(Path(path))
    coerced = _coerce_model_profile_payload(payload)
    return ModelProfile(**coerced)


def list_model_profiles(profiles_dir: Path | str) -> list[ModelProfile]:
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.exists():
        return []
    profiles = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profiles.append(load_model_profile(path))
        except Exception:
            continue
    return profiles


def resolve_model_profile(profile_ref: Path | str, profiles_dir: Path | str | None = None) -> ModelProfile:
    profile_path = Path(profile_ref)
    if profile_path.exists():
        return load_model_profile(profile_path)
    if profiles_dir is None:
        raise FileNotFoundError("Model profile %r was not found." % str(profile_ref))
    candidate = _profile_path(profiles_dir, str(profile_ref))
    if not candidate.exists():
        raise FileNotFoundError("Model profile %r was not found in %s." % (str(profile_ref), str(profiles_dir)))
    return load_model_profile(candidate)


def default_model_profiles() -> Iterable[ModelProfile]:
    yield ModelProfile(
        profile_id="prod_fast_native",
        display_name="Production Fast Native",
        backend_family="native_fast",
        task_type="classification",
        runtime_engine="onnxruntime",
        thresholds={"reject_threshold": 0.5, "good_max": 0.35, "bad_min": 0.65, "target_ms": 5.0},
        supports_training=True,
        metadata={"description": "Two-stage native cascade optimized for production latency."},
    )
    yield ModelProfile(
        profile_id="review_native_base",
        display_name="Review Native Base",
        backend_family="native_review",
        task_type="classification",
        variant="base",
        runtime_engine="pytorch",
        thresholds={"reject_threshold": 0.5},
        supports_training=True,
        metadata={"description": "Full review model with richer evidence and slower runtime."},
    )
    yield ModelProfile(
        profile_id="review_native_lite",
        display_name="Review Native Lite",
        backend_family="native_review",
        task_type="classification",
        variant="lite",
        runtime_engine="pytorch",
        thresholds={"reject_threshold": 0.5},
        supports_training=True,
        metadata={"description": "Lightweight review model with Base/Lite parity."},
    )
    yield ModelProfile(
        profile_id="hf_classification",
        display_name="Hugging Face Classification",
        backend_family="hf_classification",
        task_type="classification",
        runtime_engine="transformers",
        cache_policy="online_cache",
        thresholds={"reject_threshold": 0.5},
        supports_training=True,
        metadata={"description": "Transformers image classification backend with label mapping."},
    )
    yield ModelProfile(
        profile_id="hf_detection",
        display_name="Hugging Face Detection",
        backend_family="hf_detection",
        task_type="detection",
        runtime_engine="transformers",
        cache_policy="online_cache",
        thresholds={"reject_threshold": 0.5},
        supports_training=False,
        metadata={"description": "Transformers object detection backend for review and evaluation."},
    )
    yield ModelProfile(
        profile_id="hf_segmentation",
        display_name="Hugging Face Segmentation",
        backend_family="hf_segmentation",
        task_type="segmentation",
        runtime_engine="transformers",
        cache_policy="online_cache",
        thresholds={"reject_threshold": 0.5},
        supports_training=False,
        metadata={"description": "Transformers segmentation backend for review and evaluation."},
    )


def ensure_default_model_profiles(config: WorkspaceConfig | Mapping[str, object]) -> list[Path]:
    profiles_dir = Path(config.profiles_dir if isinstance(config, WorkspaceConfig) else str(config["profiles_dir"]))
    ensure_dir(profiles_dir)
    written = []
    existing = {profile.profile_id for profile in list_model_profiles(profiles_dir)}
    for profile in default_model_profiles():
        if profile.profile_id in existing:
            continue
        written.append(save_model_profile(profile, profiles_dir))
    return written


def workspace_profile_lookup(config: WorkspaceConfig) -> dict[str, ModelProfile]:
    return {profile.profile_id: profile for profile in list_model_profiles(config.profiles_dir)}
