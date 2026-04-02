from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from .utils import ensure_dir, read_json, utc_timestamp, write_json


MODEL_BACKEND_FAMILIES = (
    "native",
    "hf_classification",
    "hf_detection",
    "hf_segmentation",
)

RUNTIME_ENGINES = ("pytorch", "onnxruntime", "tensorrt")

CACHE_POLICIES = ("online_and_cache", "offline_cache", "local_only")

PROFILE_TASK_TYPES = ("train", "predict", "review", "benchmark", "calibrate")


@dataclass
class ModelProfile:
    profile_id: str
    display_name: str = ""
    backend_family: str = "native"
    task_type: str = "predict"
    model_id: Optional[str] = None
    local_path: Optional[str] = None
    model_revision: Optional[str] = None
    native_variant: str = "base"
    runtime_engine: str = "pytorch"
    cache_policy: str = "online_and_cache"
    latency_target_ms: float = 5.0
    input_mode: str = "grayscale"
    output_mode: str = "hierarchical"
    label_mapping: Mapping[str, str] = field(default_factory=dict)
    defect_family_mapping: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.profile_id = str(self.profile_id).strip()
        if not self.profile_id:
            raise ValueError("ModelProfile.profile_id cannot be empty.")
        if self.backend_family not in MODEL_BACKEND_FAMILIES:
            raise ValueError("Unsupported backend family %r." % self.backend_family)
        if self.task_type not in PROFILE_TASK_TYPES:
            raise ValueError("Unsupported profile task type %r." % self.task_type)
        if self.runtime_engine not in RUNTIME_ENGINES:
            raise ValueError("Unsupported runtime engine %r." % self.runtime_engine)
        if self.cache_policy not in CACHE_POLICIES:
            raise ValueError("Unsupported cache policy %r." % self.cache_policy)

    @property
    def is_native(self) -> bool:
        return self.backend_family == "native"

    @property
    def is_huggingface(self) -> bool:
        return self.backend_family.startswith("hf_")

    @property
    def cache_key(self) -> str:
        return "|".join(
            [
                self.profile_id,
                self.backend_family,
                self.task_type,
                self.model_id or "",
                self.local_path or "",
                self.model_revision or "",
                self.native_variant,
                self.runtime_engine,
                self.cache_policy,
            ]
        )


@dataclass
class WorkspaceConfig:
    workspace_name: str = "GreyModel Workspace"
    version: int = 1
    run_root: str = "artifacts"
    data_root: str = "data"
    cache_root: str = ".cache/greymodel"
    active_dataset_index: Optional[str] = None
    active_manifest: Optional[str] = None
    active_model_profile: Optional[str] = None
    default_execution_backend: str = "local"
    slurm_cpus: int = 8
    slurm_mem: str = "50G"
    slurm_gres: str = "gpu:8"
    slurm_partition: str = ""
    slurm_queue: str = ""
    slurm_nproc_per_node: int = 8
    proxy_mode: str = "auto"
    ui_theme: str = "clean"
    recent_dataset_indexes: tuple[str, ...] = ()
    recent_run_dirs: tuple[str, ...] = ()
    model_profiles: Mapping[str, ModelProfile] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.run_root = str(self.run_root)
        self.data_root = str(self.data_root)
        self.cache_root = str(self.cache_root)
        self.recent_dataset_indexes = tuple(str(value) for value in self.recent_dataset_indexes)
        self.recent_run_dirs = tuple(str(value) for value in self.recent_run_dirs)
        if not isinstance(self.model_profiles, dict):
            self.model_profiles = dict(self.model_profiles)


def workspace_path_for(run_root: Path | str) -> Path:
    return Path(run_root) / "workspace.json"


def _default_model_profiles() -> dict[str, ModelProfile]:
    created = utc_timestamp()
    profiles = {
        "prod_fast_native": ModelProfile(
            profile_id="prod_fast_native",
            display_name="Production Fast Path",
            backend_family="native",
            task_type="predict",
            model_id="base",
            native_variant="fast",
            runtime_engine="onnxruntime",
            cache_policy="local_only",
            latency_target_ms=5.0,
            notes="Fast cascade profile for production screening.",
            created_at=created,
            updated_at=created,
        ),
        "review_native_base": ModelProfile(
            profile_id="review_native_base",
            display_name="Review Base",
            backend_family="native",
            task_type="review",
            model_id="base",
            native_variant="base",
            runtime_engine="pytorch",
            cache_policy="local_only",
            latency_target_ms=20.0,
            notes="Higher-capacity native review profile.",
            created_at=created,
            updated_at=created,
        ),
        "review_native_lite": ModelProfile(
            profile_id="review_native_lite",
            display_name="Review Lite",
            backend_family="native",
            task_type="review",
            model_id="lite",
            native_variant="lite",
            runtime_engine="pytorch",
            cache_policy="local_only",
            latency_target_ms=12.0,
            notes="Compact native review profile.",
            created_at=created,
            updated_at=created,
        ),
        "hf_classification": ModelProfile(
            profile_id="hf_classification",
            display_name="Hugging Face Classification",
            backend_family="hf_classification",
            task_type="review",
            runtime_engine="pytorch",
            cache_policy="online_and_cache",
            latency_target_ms=100.0,
            notes="Registration template for HF image-classification backends.",
            created_at=created,
            updated_at=created,
        ),
        "hf_detection": ModelProfile(
            profile_id="hf_detection",
            display_name="Hugging Face Detection",
            backend_family="hf_detection",
            task_type="review",
            runtime_engine="pytorch",
            cache_policy="online_and_cache",
            latency_target_ms=120.0,
            notes="Registration template for HF detection backends.",
            created_at=created,
            updated_at=created,
        ),
        "hf_segmentation": ModelProfile(
            profile_id="hf_segmentation",
            display_name="Hugging Face Segmentation",
            backend_family="hf_segmentation",
            task_type="review",
            runtime_engine="pytorch",
            cache_policy="online_and_cache",
            latency_target_ms=150.0,
            notes="Registration template for HF segmentation backends.",
            created_at=created,
            updated_at=created,
        ),
    }
    return profiles


def default_workspace_config(run_root: Path | str = "artifacts", data_root: Path | str = "data") -> WorkspaceConfig:
    created = utc_timestamp()
    return WorkspaceConfig(
        workspace_name="GreyModel Workspace",
        run_root=str(run_root),
        data_root=str(data_root),
        cache_root=str(Path(run_root) / ".cache"),
        active_model_profile="prod_fast_native",
        default_execution_backend="local",
        slurm_cpus=8,
        slurm_mem="50G",
        slurm_gres="gpu:8",
        slurm_partition="",
        slurm_queue="3h",
        slurm_nproc_per_node=8,
        proxy_mode="auto",
        ui_theme="clean",
        recent_dataset_indexes=(),
        recent_run_dirs=(),
        model_profiles=_default_model_profiles(),
        notes="",
        created_at=created,
        updated_at=created,
    )


def _profile_from_payload(payload: Mapping[str, Any]) -> ModelProfile:
    return ModelProfile(
        profile_id=str(payload.get("profile_id", "")).strip(),
        display_name=str(payload.get("display_name", "")),
        backend_family=str(payload.get("backend_family", "native")),
        task_type=str(payload.get("task_type", "predict")),
        model_id=payload.get("model_id"),
        local_path=payload.get("local_path"),
        model_revision=payload.get("model_revision"),
        native_variant=str(payload.get("native_variant", "base")),
        runtime_engine=str(payload.get("runtime_engine", "pytorch")),
        cache_policy=str(payload.get("cache_policy", "online_and_cache")),
        latency_target_ms=float(payload.get("latency_target_ms", 5.0) or 5.0),
        input_mode=str(payload.get("input_mode", "grayscale")),
        output_mode=str(payload.get("output_mode", "hierarchical")),
        label_mapping=dict(payload.get("label_mapping", {})),
        defect_family_mapping=dict(payload.get("defect_family_mapping", {})),
        notes=str(payload.get("notes", "")),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )


def _workspace_from_payload(payload: Mapping[str, Any], *, run_root: Path | str, data_root: Path | str) -> WorkspaceConfig:
    profiles_payload = payload.get("model_profiles", {}) or {}
    profiles = {str(profile_id): _profile_from_payload(profile_payload) for profile_id, profile_payload in profiles_payload.items()}
    workspace = WorkspaceConfig(
        workspace_name=str(payload.get("workspace_name", "GreyModel Workspace")),
        version=int(payload.get("version", 1) or 1),
        run_root=str(payload.get("run_root", run_root)),
        data_root=str(payload.get("data_root", data_root)),
        cache_root=str(payload.get("cache_root", Path(run_root) / ".cache")),
        active_dataset_index=payload.get("active_dataset_index"),
        active_manifest=payload.get("active_manifest"),
        active_model_profile=payload.get("active_model_profile"),
        default_execution_backend=str(payload.get("default_execution_backend", "local")),
        slurm_cpus=int(payload.get("slurm_cpus", 8) or 8),
        slurm_mem=str(payload.get("slurm_mem", "50G")),
        slurm_gres=str(payload.get("slurm_gres", "gpu:8")),
        slurm_partition=str(payload.get("slurm_partition", "")),
        slurm_queue=str(payload.get("slurm_queue", "3h")),
        slurm_nproc_per_node=int(payload.get("slurm_nproc_per_node", 8) or 8),
        proxy_mode=str(payload.get("proxy_mode", "auto")),
        ui_theme=str(payload.get("ui_theme", "clean")),
        recent_dataset_indexes=tuple(payload.get("recent_dataset_indexes", ()) or ()),
        recent_run_dirs=tuple(payload.get("recent_run_dirs", ()) or ()),
        model_profiles=profiles,
        notes=str(payload.get("notes", "")),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )
    return workspace


def workspace_to_payload(workspace: WorkspaceConfig) -> dict[str, Any]:
    payload = asdict(workspace)
    payload["model_profiles"] = {profile_id: asdict(profile) for profile_id, profile in workspace.model_profiles.items()}
    payload["recent_dataset_indexes"] = list(workspace.recent_dataset_indexes)
    payload["recent_run_dirs"] = list(workspace.recent_run_dirs)
    return payload


def load_workspace(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    workspace_path: Path | str | None = None,
) -> WorkspaceConfig:
    run_root = Path(run_root)
    data_root = Path(data_root)
    resolved_path = Path(workspace_path) if workspace_path is not None else workspace_path_for(run_root)
    if resolved_path.exists():
        workspace = _workspace_from_payload(read_json(resolved_path), run_root=run_root, data_root=data_root)
    else:
        workspace = default_workspace_config(run_root=run_root, data_root=data_root)
    workspace.model_profiles = {
        **_default_model_profiles(),
        **dict(workspace.model_profiles),
    }
    if not workspace.active_model_profile or workspace.active_model_profile not in workspace.model_profiles:
        workspace.active_model_profile = "prod_fast_native"
    if not workspace.active_dataset_index:
        workspace.active_dataset_index = None
    if not workspace.active_manifest:
        workspace.active_manifest = None
    return workspace


def save_workspace(workspace: WorkspaceConfig, workspace_path: Path | str | None = None) -> Path:
    resolved_path = Path(workspace_path) if workspace_path is not None else workspace_path_for(workspace.run_root)
    workspace.updated_at = utc_timestamp()
    payload = workspace_to_payload(workspace)
    payload["updated_at"] = workspace.updated_at
    if not workspace.created_at:
        workspace.created_at = workspace.updated_at
        payload["created_at"] = workspace.created_at
    ensure_dir(resolved_path.parent)
    return write_json(resolved_path, payload)


def upsert_model_profile(workspace: WorkspaceConfig, profile: ModelProfile) -> WorkspaceConfig:
    profile.created_at = profile.created_at or utc_timestamp()
    profile.updated_at = utc_timestamp()
    model_profiles: MutableMapping[str, ModelProfile] = dict(workspace.model_profiles)
    model_profiles[profile.profile_id] = profile
    workspace.model_profiles = dict(sorted(model_profiles.items(), key=lambda item: item[0]))
    workspace.active_model_profile = profile.profile_id
    return workspace


def delete_model_profile(workspace: WorkspaceConfig, profile_id: str) -> WorkspaceConfig:
    model_profiles: MutableMapping[str, ModelProfile] = dict(workspace.model_profiles)
    model_profiles.pop(str(profile_id), None)
    workspace.model_profiles = dict(sorted(model_profiles.items(), key=lambda item: item[0]))
    if workspace.active_model_profile == profile_id:
        workspace.active_model_profile = next(iter(workspace.model_profiles), None)
    return workspace


def set_recent_dataset_index(workspace: WorkspaceConfig, dataset_index: str, limit: int = 10) -> WorkspaceConfig:
    values = [str(dataset_index)] + [value for value in workspace.recent_dataset_indexes if value != dataset_index]
    workspace.recent_dataset_indexes = tuple(values[: max(1, int(limit))])
    workspace.active_dataset_index = str(dataset_index)
    return workspace


def set_recent_run_dir(workspace: WorkspaceConfig, run_dir: str, limit: int = 10) -> WorkspaceConfig:
    values = [str(run_dir)] + [value for value in workspace.recent_run_dirs if value != run_dir]
    workspace.recent_run_dirs = tuple(values[: max(1, int(limit))])
    return workspace


def workspace_summary(workspace: WorkspaceConfig) -> dict[str, Any]:
    return {
        "workspace_name": workspace.workspace_name,
        "run_root": workspace.run_root,
        "data_root": workspace.data_root,
        "cache_root": workspace.cache_root,
        "active_dataset_index": workspace.active_dataset_index,
        "active_manifest": workspace.active_manifest,
        "active_model_profile": workspace.active_model_profile,
        "default_execution_backend": workspace.default_execution_backend,
        "slurm_partition": workspace.slurm_partition,
        "profile_count": len(workspace.model_profiles),
    }
