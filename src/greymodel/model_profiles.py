from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .utils import ensure_dir, read_json, write_json


PROFILE_SCHEMA_VERSION = "1"
MODEL_PROFILE_REGISTRY_DIRNAME = "model_profiles"


def _normalized_optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


def _as_tuple(values: Sequence[str] | None, default: Sequence[str]) -> Tuple[str, ...]:
    source = default if values is None else values
    return tuple(str(value).strip() for value in source if str(value).strip())


def _as_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    if mapping is None:
        return {}
    return {str(key): value for key, value in dict(mapping).items()}


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    backend_family: str = "native"
    task_type: str = "native"
    model_id: Optional[str] = None
    local_path: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    runtime_engine: str = "pytorch"
    reject_threshold: float = 0.5
    uncertainty_low: float = 0.35
    uncertainty_high: float = 0.65
    good_labels: Tuple[str, ...] = ("good", "ok", "pass")
    bad_labels: Tuple[str, ...] = ("bad", "reject", "ng")
    label_mapping: Mapping[str, str] = field(default_factory=dict)
    defect_family_mapping: Mapping[str, str] = field(default_factory=dict)
    grayscale_mode: str = "replicate_rgb"
    evidence_policy: str = "bad"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profile_id = _normalized_optional_text(self.profile_id)
        if not profile_id:
            raise ValueError("ModelProfile.profile_id must not be empty.")
        backend_family = _normalized_optional_text(self.backend_family) or "native"
        task_type = _normalized_optional_text(self.task_type) or "native"
        runtime_engine = _normalized_optional_text(self.runtime_engine) or "pytorch"
        grayscale_mode = _normalized_optional_text(self.grayscale_mode) or "replicate_rgb"
        evidence_policy = _normalized_optional_text(self.evidence_policy) or "bad"
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "backend_family", backend_family.lower())
        object.__setattr__(self, "task_type", task_type.lower())
        object.__setattr__(self, "runtime_engine", runtime_engine.lower())
        object.__setattr__(self, "grayscale_mode", grayscale_mode.lower())
        object.__setattr__(self, "evidence_policy", evidence_policy.lower())
        object.__setattr__(self, "good_labels", _as_tuple(self.good_labels, ("good", "ok", "pass")))
        object.__setattr__(self, "bad_labels", _as_tuple(self.bad_labels, ("bad", "reject", "ng")))
        object.__setattr__(self, "label_mapping", _as_mapping(self.label_mapping))
        object.__setattr__(self, "defect_family_mapping", _as_mapping(self.defect_family_mapping))
        object.__setattr__(self, "metadata", _as_mapping(self.metadata))

    @property
    def is_huggingface(self) -> bool:
        return self.backend_family == "huggingface"

    @property
    def is_native(self) -> bool:
        return self.backend_family == "native"

    @property
    def source(self) -> Optional[str]:
        return _normalized_optional_text(self.local_path) or _normalized_optional_text(self.model_id)

    @property
    def native_variant(self) -> str:
        return _normalized_optional_text(self.metadata.get("variant")) or "base"

    @property
    def latency_target_ms(self) -> float:
        target = self.metadata.get("latency_target_ms", 5.0 if self.native_variant == "fast" else 20.0)
        try:
            return float(target)
        except Exception:
            return 20.0

    def canonical_defect_families(self) -> Tuple[str, ...]:
        families = []
        for canonical in self.defect_family_mapping.values():
            normalized = _normalized_optional_text(canonical)
            if normalized and normalized not in {"good", "bad"} and normalized not in families:
                families.append(normalized)
        for canonical in self.label_mapping.values():
            normalized = _normalized_optional_text(canonical)
            if normalized and normalized not in {"good", "bad"} and normalized not in families:
                families.append(normalized)
        return tuple(families)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = PROFILE_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelProfile":
        data = dict(payload)
        data.pop("schema_version", None)
        return cls(**data)


def model_profile_registry_dir(registry_root: Path | str) -> Path:
    return ensure_dir(Path(registry_root))


def model_profile_path(registry_root: Path | str, profile_id: str) -> Path:
    return model_profile_registry_dir(registry_root) / ("%s.json" % _normalized_optional_text(profile_id))


def save_model_profile(profile: ModelProfile, registry_root: Path | str) -> Path:
    path = model_profile_path(registry_root, profile.profile_id)
    return write_json(path, profile.to_dict())


def register_model_profile(profile: ModelProfile, registry_root: Path | str) -> ModelProfile:
    save_model_profile(profile, registry_root)
    return profile


def load_model_profile(profile_reference: ModelProfile | Mapping[str, Any] | Path | str, registry_root: Path | str | None = None) -> ModelProfile:
    if isinstance(profile_reference, ModelProfile):
        return profile_reference
    if isinstance(profile_reference, Mapping):
        return ModelProfile.from_dict(profile_reference)
    path = Path(profile_reference)
    if path.exists():
        return ModelProfile.from_dict(read_json(path))
    if registry_root is not None:
        registry_path = model_profile_path(registry_root, str(profile_reference))
        if registry_path.exists():
            return ModelProfile.from_dict(read_json(registry_path))
    raise FileNotFoundError("Model profile %r could not be resolved." % (str(profile_reference),))


def list_model_profiles(registry_root: Path | str) -> list[ModelProfile]:
    root = Path(registry_root)
    if not root.exists():
        return []
    profiles = []
    for path in sorted(root.glob("*.json")):
        try:
            profiles.append(ModelProfile.from_dict(read_json(path)))
        except Exception:
            continue
    profiles.sort(key=lambda profile: profile.profile_id)
    return profiles


def delete_model_profile(registry_root: Path | str, profile_id: str) -> bool:
    path = model_profile_path(registry_root, profile_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def ensure_default_model_profiles(registry_root: Path | str) -> list[ModelProfile]:
    root = model_profile_registry_dir(registry_root)
    defaults = [
        ModelProfile(
            profile_id="prod_fast_native",
            backend_family="native",
            task_type="native",
            runtime_engine="onnxruntime",
            metadata={"variant": "fast", "latency_target_ms": 5.0},
        ),
        ModelProfile(
            profile_id="review_native_base",
            backend_family="native",
            task_type="native",
            metadata={"variant": "base", "latency_target_ms": 20.0},
        ),
        ModelProfile(
            profile_id="review_native_lite",
            backend_family="native",
            task_type="native",
            metadata={"variant": "lite", "latency_target_ms": 12.0},
        ),
        ModelProfile(
            profile_id="hf_classification",
            backend_family="huggingface",
            task_type="classification",
            runtime_engine="pytorch",
            evidence_policy="bad",
            metadata={"latency_target_ms": 100.0},
        ),
        ModelProfile(
            profile_id="hf_detection",
            backend_family="huggingface",
            task_type="detection",
            runtime_engine="pytorch",
            evidence_policy="bad",
            metadata={"latency_target_ms": 120.0},
        ),
        ModelProfile(
            profile_id="hf_segmentation",
            backend_family="huggingface",
            task_type="segmentation",
            runtime_engine="pytorch",
            evidence_policy="bad",
            metadata={"latency_target_ms": 150.0},
        ),
    ]
    for profile in defaults:
        path = model_profile_path(root, profile.profile_id)
        if not path.exists():
            save_model_profile(profile, root)
    return list_model_profiles(root)
