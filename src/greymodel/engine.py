from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .autofit import AutoFitResult, run_autofit
from .evaluation import benchmark_manifest
from .explainability import build_explanation_bundle
from .hf_backends import build_huggingface_model_backend
from .model_profiles import ModelProfile, load_model_profile
from .runners import (
    run_domain_adaptation_stage,
    run_finetune_stage,
    run_pretraining_stage,
    run_resume_stage,
)
from .settings import GreyModelSettings, build_environment_report, ensure_settings
from .training import TrainingConfig
from .types import LatencyReport, ModelInput, PredictionEvidence, PredictionRecord, StationConfig
from .utils import load_uint8_grayscale


def _profile_alias(profile_reference: str | Path | ModelProfile | Mapping[str, Any] | None) -> str | Path | ModelProfile | Mapping[str, Any]:
    if profile_reference in (None, ""):
        return "prod_fast_native"
    if isinstance(profile_reference, (Path, ModelProfile, Mapping)):
        return profile_reference
    aliases = {
        "fast": "prod_fast_native",
        "prod_fast": "prod_fast_native",
        "base": "review_native_base",
        "lite": "review_native_lite",
    }
    normalized = str(profile_reference).strip().lower()
    return aliases.get(normalized, profile_reference)


def _default_geometry_mode(image_uint8: np.ndarray, geometry_mode: str | None = None) -> str:
    if geometry_mode not in (None, ""):
        return str(geometry_mode)
    return "square" if int(image_uint8.shape[0]) == int(image_uint8.shape[1]) else "rect"


def _geometry_mode_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(getattr(value, "value", value))


def _default_station_config(
    image_uint8: np.ndarray,
    *,
    station_id: Any,
    geometry_mode: str | None = None,
    reject_threshold: float = 0.5,
) -> StationConfig:
    height, width = [int(value) for value in image_uint8.shape]
    tile_size = max(16, min(64, height, width))
    tile_stride = max(8, tile_size // 2)
    return StationConfig(
        canvas_shape=(height, width),
        station_id=station_id,
        geometry_mode=_default_geometry_mode(image_uint8, geometry_mode),
        pad_value=0,
        normalization_mean=127.5,
        normalization_std=50.0,
        tile_size=(tile_size, tile_size),
        tile_stride=(tile_stride, tile_stride),
        adapter_id="auto",
        reject_threshold=float(reject_threshold),
    )


class GreyModel:
    """Ultralytics-style high-level runtime wrapper for GreyModel backends."""

    def __init__(
        self,
        model: str | Path | ModelProfile | Mapping[str, Any] | None = None,
        *,
        home: Path | str | None = None,
        registry_root: Path | str | None = None,
        cache_root: Path | str | None = None,
        defect_families: Sequence[str] = (),
        local_files_only: bool = False,
        num_stations: int = 32,
    ) -> None:
        self.settings: GreyModelSettings = ensure_settings(home)
        self.registry_root = Path(registry_root) if registry_root is not None else Path(self.settings.registry_root)
        self.cache_root = Path(cache_root) if cache_root is not None else Path(self.settings.cache_root)
        self.profile = load_model_profile(_profile_alias(model or self.settings.active_profile), registry_root=self.registry_root)
        self.defect_families = tuple(defect_families) or tuple(self.profile.canonical_defect_families())
        self.local_files_only = bool(local_files_only)
        self.num_stations = int(num_stations)
        self.model = build_huggingface_model_backend(
            self.profile,
            num_defect_families=max(len(self.defect_families), 1),
            defect_families=self.defect_families,
            num_stations=self.num_stations,
            registry_root=self.registry_root,
        )

    def info(self) -> dict[str, Any]:
        environment = build_environment_report(self.settings.home)
        return {
            "profile": self.profile.to_dict(),
            "defect_families": list(self.defect_families),
            "registry_root": str(self.registry_root),
            "cache_root": str(self.cache_root),
            "local_files_only": self.local_files_only,
            "environment": environment,
        }

    def _build_model_input(
        self,
        source: np.ndarray | Path | str | ModelInput,
        *,
        station_id: Any = "station-auto",
        geometry_mode: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelInput:
        if isinstance(source, ModelInput):
            return source
        if isinstance(source, np.ndarray):
            image_uint8 = source
        else:
            image_uint8 = load_uint8_grayscale(Path(source))
        return ModelInput(
            image_uint8=np.asarray(image_uint8, dtype=np.uint8),
            station_id=station_id,
            geometry_mode=_default_geometry_mode(np.asarray(image_uint8, dtype=np.uint8), geometry_mode),
            metadata=metadata,
        )

    def predict(
        self,
        source: np.ndarray | Path | str | ModelInput,
        *,
        station_config: StationConfig | None = None,
        station_id: Any = "station-auto",
        geometry_mode: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PredictionRecord:
        model_input = self._build_model_input(
            source,
            station_id=station_id,
            geometry_mode=geometry_mode,
            metadata=metadata,
        )
        if station_config is None:
            station_config = _default_station_config(
                model_input.image_uint8,
                station_id=model_input.station_id,
                geometry_mode=_geometry_mode_value(model_input.geometry_mode),
                reject_threshold=float(self.profile.reject_threshold),
            )
        output = self.model.forward(model_input, station_config)
        reject_score = float(np.asarray(output.reject_score).reshape(()))
        predicted_label = int(reject_score >= float(station_config.reject_threshold))
        defect_probs = {
            family: float(value)
            for family, value in zip(
                self.defect_families,
                np.asarray(output.defect_family_probs, dtype=np.float32).reshape(-1).tolist(),
            )
        }
        evidence = PredictionEvidence(
            station_decision={
                "station_id": str(station_config.station_id),
                "threshold": float(station_config.reject_threshold),
                "reject": bool(predicted_label),
            },
            metadata={
                "profile_id": self.profile.profile_id,
                "backend": dict(getattr(output, "metadata", {}) or {}).get("backend"),
                "runtime_engine": self.profile.runtime_engine,
                "top_tile_indices": np.asarray(output.top_tile_indices).reshape(-1).tolist()
                if output.top_tile_indices is not None
                else [],
                "top_tile_boxes": np.asarray(output.top_tile_boxes).tolist()
                if output.top_tile_boxes is not None
                else [],
            },
        )
        return PredictionRecord(
            sample_id=str(getattr(source, "sample_id", "inference")),
            station_id=model_input.station_id,
            accept_reject=predicted_label,
            reject_score=reject_score,
            predicted_label=predicted_label,
            primary_label="bad" if predicted_label else "good",
            primary_score=reject_score,
            top_defect_family=max(defect_probs.items(), key=lambda item: item[1])[0] if defect_probs else None,
            defect_family_probs=defect_probs,
            evidence=evidence,
            metadata={"profile_id": self.profile.profile_id, **dict(metadata or {}), **dict(getattr(output, "metadata", {}) or {})},
        )

    __call__ = predict

    def explain(
        self,
        source: np.ndarray | Path | str | ModelInput,
        output_dir: Path | str,
        *,
        station_config: StationConfig | None = None,
        station_id: Any = "station-auto",
        geometry_mode: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Path]:
        model_input = self._build_model_input(
            source,
            station_id=station_id,
            geometry_mode=geometry_mode,
            metadata=metadata,
        )
        if station_config is None:
            station_config = _default_station_config(
                model_input.image_uint8,
                station_id=model_input.station_id,
                geometry_mode=_geometry_mode_value(model_input.geometry_mode),
                reject_threshold=float(self.profile.reject_threshold),
            )
        return build_explanation_bundle(self.model, model_input, station_config, output_dir)

    def benchmark(
        self,
        source: np.ndarray | Path | str | ModelInput | None = None,
        *,
        station_config: StationConfig | None = None,
        station_id: Any = "station-auto",
        geometry_mode: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        iterations: int = 20,
        warmup_iterations: int = 3,
    ) -> LatencyReport:
        if source is None:
            source = np.zeros((256, 256), dtype=np.uint8)
        model_input = self._build_model_input(
            source,
            station_id=station_id,
            geometry_mode=geometry_mode,
            metadata=metadata,
        )
        if station_config is None:
            station_config = _default_station_config(
                model_input.image_uint8,
                station_id=model_input.station_id,
                geometry_mode=_geometry_mode_value(model_input.geometry_mode),
                reject_threshold=float(self.profile.reject_threshold),
            )
        torch_module = None
        try:
            import torch as torch_module  # type: ignore[assignment]
        except Exception:
            torch_module = None
        for _ in range(max(int(warmup_iterations), 0)):
            self.model.forward(model_input, station_config)
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.synchronize()
        durations_ms = []
        peak_memory_mb = 0.0
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()
        for _ in range(max(int(iterations), 1)):
            started = time.perf_counter()
            self.model.forward(model_input, station_config)
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.synchronize()
                peak_memory_mb = max(peak_memory_mb, float(torch_module.cuda.max_memory_allocated() / (1024.0 * 1024.0)))
            durations_ms.append((time.perf_counter() - started) * 1000.0)
        durations = np.asarray(durations_ms, dtype=np.float32)
        return LatencyReport(
            profile_id=self.profile.profile_id,
            backend_family=self.profile.backend_family,
            runtime_engine=self.profile.runtime_engine,
            accelerator="cuda" if torch_module is not None and torch_module.cuda.is_available() else "cpu",
            batch_size=1,
            image_shape=tuple(int(value) for value in model_input.image_uint8.shape),
            iterations=max(int(iterations), 1),
            warmup_iterations=max(int(warmup_iterations), 0),
            mean_ms=float(np.mean(durations)),
            p50_ms=float(np.percentile(durations, 50)),
            p95_ms=float(np.percentile(durations, 95)),
            throughput_per_second=float(1000.0 / max(float(np.mean(durations)), 1e-6)),
            peak_memory_mb=float(peak_memory_mb),
            target_ms=float(self.profile.latency_target_ms),
            meets_target=bool(float(np.percentile(durations, 95)) <= float(self.profile.latency_target_ms)),
            metadata={"profile_id": self.profile.profile_id},
        )

    def train(
        self,
        stage: str,
        *,
        manifest_path: Path | str,
        index_path: Path | str | None = None,
        run_root: Path | str | None = None,
        batch_size: int = 4,
        training_config: Optional[TrainingConfig] = None,
        checkpoint_path: Path | str | None = None,
        resume_from: Path | str | None = None,
    ):
        run_root = run_root or self.settings.run_root
        stage_key = str(stage).strip().lower().replace("-", "_")
        if stage_key == "pretrain":
            return run_pretraining_stage(
                manifest_path=manifest_path,
                index_path=index_path,
                variant=self.profile.native_variant if self.profile.native_variant in {"base", "lite"} else "base",
                run_root=run_root,
                batch_size=batch_size,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
            )
        if stage_key in {"domain_adapt", "domainadapt"}:
            return run_domain_adaptation_stage(
                manifest_path=manifest_path,
                index_path=index_path,
                variant=self.profile.native_variant if self.profile.native_variant in {"base", "lite"} else "base",
                run_root=run_root,
                batch_size=batch_size,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
            )
        if stage_key == "finetune":
            return run_finetune_stage(
                manifest_path=manifest_path,
                index_path=index_path,
                variant=self.profile.native_variant if self.profile.native_variant in {"base", "lite"} else "base",
                run_root=run_root,
                batch_size=batch_size,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
            )
        if stage_key == "resume":
            if checkpoint_path is None:
                raise ValueError("GreyModel.train(stage='resume') requires checkpoint_path.")
            return run_resume_stage(
                manifest_path=manifest_path,
                index_path=index_path,
                variant=self.profile.native_variant if self.profile.native_variant in {"base", "lite"} else "base",
                run_root=run_root,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
                training_config=training_config,
        )
        raise ValueError("Unsupported training stage %r." % stage)

    def fit(
        self,
        data: Path | str,
        *,
        model: str | Path | ModelProfile | Mapping[str, Any] | None = None,
        execution: str = "auto",
        run_root: Path | str | None = None,
        overrides: Optional[Mapping[str, Any]] = None,
        warm_start: Path | str | None = None,
        split_policy: str = "auto",
    ) -> AutoFitResult:
        return run_autofit(
            data=data,
            model=model or self.profile.profile_id,
            execution=execution,
            run_root=run_root or self.settings.run_root,
            overrides=overrides,
            warm_start=warm_start,
            split_policy=split_policy,
            registry_root=self.registry_root,
        )

    def val(
        self,
        *,
        manifest_path: Path | str,
        index_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        return benchmark_manifest(
            manifest_path,
            index_path=index_path,
            variant=self.profile.native_variant if self.profile.native_variant in {"base", "lite"} else "base",
            output_path=output_path,
            model_profile=self.profile.profile_id,
            model_registry_root=self.registry_root,
        )
