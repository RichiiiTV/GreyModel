from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .calibration import StationCalibration, StationCalibrator
from .data import load_dataset_index, load_dataset_manifest, load_station_configs_from_index, station_config_for_record
from .preprocessing import preprocess_image, stack_prepared_images
from .types import ModelInput, StationConfig, TensorBatch, station_id_to_int
from .utils import ensure_dir, save_array_artifact, write_json
from .version import __version__


def _torch_backend_from_model(model):
    backend = getattr(model, "backend", None)
    backend_model = getattr(backend, "model", None)
    if backend_model is not None:
        return backend_model
    direct_model = getattr(model, "model", None)
    if direct_model is not None and (
        hasattr(direct_model, "screen_forward")
        or hasattr(getattr(direct_model, "config", None), "num_stations")
    ):
        return direct_model
    return None


def _manual_integrated_gradients(torch_model, batch: TensorBatch, steps: int = 16):
    import torch

    baseline = torch.zeros_like(batch.image)
    total_grads = torch.zeros_like(batch.image)
    num_stations = max(int(getattr(getattr(torch_model, "config", None), "num_stations", 1) or 1), 1)
    station_ids = batch.station_id % num_stations
    for alpha in torch.linspace(0.0, 1.0, steps, device=batch.image.device):
        scaled = (baseline + alpha * (batch.image - baseline)).detach().requires_grad_(True)
        working_batch = TensorBatch(
            image=scaled,
            valid_mask=batch.valid_mask,
            station_id=station_ids,
            geometry_id=batch.geometry_id,
            metadata=batch.metadata,
        )
        output = torch_model(working_batch)
        target = output.accept_reject_logit.sum()
        torch_model.zero_grad(set_to_none=True)
        target.backward()
        total_grads += scaled.grad.detach()
    return (batch.image - baseline) * (total_grads / max(steps, 1))


def _captum_integrated_gradients(torch_model, batch: TensorBatch, steps: int = 16):
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return _manual_integrated_gradients(torch_model, batch, steps=steps), "manual_integrated_gradients"

    num_stations = max(int(getattr(getattr(torch_model, "config", None), "num_stations", 1) or 1), 1)
    station_ids = batch.station_id % num_stations
    geometry_ids = batch.geometry_id

    def _expand_batch_dim(tensor, batch_size: int):
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        repeats = [1] * tensor.ndim
        repeats[0] = int(np.ceil(float(batch_size) / float(tensor.shape[0])))
        expanded = tensor.repeat(*repeats)
        return expanded[:batch_size]

    def _forward(image_tensor):
        batch_size = int(image_tensor.shape[0])
        working_batch = TensorBatch(
            image=image_tensor,
            valid_mask=_expand_batch_dim(batch.valid_mask, batch_size),
            station_id=_expand_batch_dim(station_ids, batch_size),
            geometry_id=_expand_batch_dim(geometry_ids, batch_size),
            metadata=batch.metadata,
        )
        return torch_model(working_batch).accept_reject_logit

    ig = IntegratedGradients(_forward)
    baseline = batch.image * 0.0
    attribution = ig.attribute(batch.image, baselines=baseline, n_steps=max(steps, 4))
    return attribution, "captum_integrated_gradients"


def build_explanation_bundle(
    model,
    model_input: ModelInput,
    station_config: StationConfig,
    output_dir: Path | str,
    attribution_steps: int = 16,
) -> Dict[str, Path]:
    output_dir = ensure_dir(Path(output_dir))
    prepared = preprocess_image(model_input, station_config)
    batch = stack_prepared_images([prepared], as_torch=True)
    torch_model = _torch_backend_from_model(model)
    output = model.forward(model_input, station_config)

    if torch_model is not None:
        import torch

        torch_model.eval()
        try:
            device = next(torch_model.parameters()).device
        except StopIteration:
            device = batch.image.device
        batch = TensorBatch(
            image=batch.image.to(device),
            valid_mask=batch.valid_mask.to(device),
            station_id=batch.station_id.to(device),
            geometry_id=batch.geometry_id.to(device),
            metadata=batch.metadata,
        )
        with torch.enable_grad():
            attribution, attribution_method = _captum_integrated_gradients(torch_model, batch, steps=attribution_steps)
        attribution_np = attribution.detach().cpu().numpy()[0, 0]
    else:
        attribution_np = np.zeros_like(prepared.image, dtype=np.float32)
        attribution_method = "unavailable"

    heatmap = np.asarray(output.defect_heatmap, dtype=np.float32)
    valid_mask = prepared.valid_mask.astype(np.float32)
    local_heatmap = np.asarray(output.local_heatmap, dtype=np.float32) if output.local_heatmap is not None else heatmap
    global_heatmap = np.asarray(output.global_heatmap, dtype=np.float32) if output.global_heatmap is not None else heatmap
    top_tiles = np.asarray(output.top_tiles)
    station_key = station_id_to_int(station_config.station_id)

    calibrator = StationCalibrator(
        [StationCalibration(station_id=station_key, reject_threshold=station_config.reject_threshold, temperature=1.0)]
    )
    decision = calibrator.calibrate(
        station_id=station_key,
        reject_logit=float(np.asarray(output.accept_reject_logit).reshape(())),
        defect_logits={},
    )

    image_artifact = save_array_artifact(output_dir / "original_image", model_input.image_uint8)
    mask_artifact = save_array_artifact(output_dir / "valid_mask", valid_mask)
    attribution_artifact = save_array_artifact(output_dir / "attribution", attribution_np)
    heatmap_artifact = save_array_artifact(output_dir / "defect_heatmap", heatmap)
    local_heatmap_artifact = save_array_artifact(output_dir / "local_heatmap", local_heatmap)
    global_heatmap_artifact = save_array_artifact(output_dir / "global_heatmap", global_heatmap)
    top_tiles_path = write_json(output_dir / "top_tiles.json", {"top_tiles": top_tiles.tolist()})
    prediction_path = write_json(
        output_dir / "prediction.json",
        {
            "primary_label": "bad" if float(np.asarray(output.reject_score).reshape(())) >= float(decision.reject_threshold) else "good",
            "reject_score": float(np.asarray(output.reject_score).reshape(())),
            "accept_reject_logit": float(np.asarray(output.accept_reject_logit).reshape(())),
            "defect_family_probs": np.asarray(output.defect_family_probs, dtype=np.float32).reshape(-1).tolist(),
            "top_tiles": top_tiles.tolist(),
            "station_decision": {"reject": bool(decision.reject), "threshold": float(decision.reject_threshold)},
            "attribution_method": attribution_method,
            "model_metadata": dict(getattr(output, "metadata", {}) or {}),
            "model_version": __version__,
        },
    )
    bundle_path = write_json(
        output_dir / "bundle.json",
        {
            "image_path": image_artifact["npy"],
            "valid_mask_path": mask_artifact["npy"],
            "attribution_path": attribution_artifact["npy"],
            "heatmap_path": heatmap_artifact["npy"],
            "local_heatmap_path": local_heatmap_artifact["npy"],
            "global_heatmap_path": global_heatmap_artifact["npy"],
            "top_tiles_path": str(top_tiles_path),
            "prediction_path": str(prediction_path),
        },
    )
    return {
        "image_path": Path(image_artifact["npy"]),
        "valid_mask_path": Path(mask_artifact["npy"]),
        "attribution_path": Path(attribution_artifact["npy"]),
        "heatmap_path": Path(heatmap_artifact["npy"]),
        "local_heatmap_path": Path(local_heatmap_artifact["npy"]),
        "global_heatmap_path": Path(global_heatmap_artifact["npy"]),
        "top_tiles_path": top_tiles_path,
        "prediction_path": prediction_path,
        "bundle_path": bundle_path,
    }


def build_audit_report(
    model_factory,
    manifest_path: Path | str,
    output_dir: Path | str,
    limit: int = 5,
    failure_dir: Optional[Path | str] = None,
    continue_on_error: bool = True,
) -> Path:
    from .utils import load_uint8_grayscale

    output_dir = ensure_dir(Path(output_dir))
    failure_root = ensure_dir(Path(failure_dir)) if failure_dir is not None else ensure_dir(output_dir / "failures")
    records = load_dataset_manifest(manifest_path)[:limit]
    index = load_dataset_index(Path(manifest_path).with_name("dataset_index.json"))
    station_configs = load_station_configs_from_index(index)
    model = model_factory()
    bundles = []
    failures = []
    for record in records:
        try:
            image = load_uint8_grayscale(Path(record.image_path))
            model_input = ModelInput(
                image_uint8=image,
                station_id=record.station_id,
                geometry_mode=record.geometry_mode,
                metadata=record.capture_metadata,
            )
            station_config = station_config_for_record(record, station_configs)
            sample_dir = ensure_dir(output_dir / record.sample_id.replace("/", "_"))
            bundle = build_explanation_bundle(model, model_input, station_config, sample_dir)
            bundles.append({"sample_id": record.sample_id, **{key: str(value) for key, value in bundle.items()}})
        except Exception as exc:
            failure_payload = {
                "sample_id": record.sample_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "image_path": record.image_path,
            }
            failure_path = write_json(failure_root / ("%s.json" % record.sample_id.replace("/", "_")), failure_payload)
            failures.append({"sample_id": record.sample_id, "failure_path": str(failure_path), **failure_payload})
            if not continue_on_error:
                raise
    return write_json(output_dir / "audit_report.json", {"bundles": bundles, "failures": failures, "model_version": __version__})
