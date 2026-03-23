from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import io
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .preprocessing import preprocess_sample, stack_prepared_images
from .types import BoxAnnotation, DatasetIndex, DatasetRecord, GeometryMode, Sample, StationConfig
from .utils import (
    ensure_dir,
    first_nonempty,
    load_uint8_grayscale,
    normalize_uint8_image,
    read_json,
    read_jsonl,
    stable_int_hash,
    utc_timestamp,
    write_json,
    write_jsonl,
)


SUPPORTED_IMAGE_SUFFIXES = (".npy", ".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
POSITIVE_LABEL_FOLDERS = {"reject", "defect", "defects", "ng", "nok", "bad"}
NEGATIVE_LABEL_FOLDERS = {"accept", "ok", "clean", "good", "pass"}
PRODUCT_FAMILIES = {"syringe", "syringes", "vial", "vials"}
GRAYSCALE_IMAGE_MODES = {"1", "L", "LA", "I", "I;16", "F"}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for sharded sampling utilities.") from exc
    return torch


def _require_huggingface_datasets():
    try:
        import datasets as datasets_module
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets support requires the `datasets` package. Install `greymodel[framework]` or add `datasets`."
        ) from exc
    return datasets_module.load_dataset, getattr(datasets_module, "DownloadConfig", None)


def _resolve_huggingface_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token not in (None, ""):
        return str(explicit_token)
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            return str(env_value)
    return None


def _is_huggingface_rate_limit_error(exc: BaseException) -> bool:
    message = ("%s: %s" % (type(exc).__name__, exc)).lower()
    return (
        "429" in message
        or "too many requests" in message
        or "rate limit" in message
        or "ratelimit" in message
    )


def _load_huggingface_dataset(
    dataset_name: str,
    *,
    config_name: Optional[str] = None,
    split: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
    data_dir: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    max_retries: int = 4,
    retry_backoff_seconds: float = 5.0,
):
    load_dataset, download_config_cls = _require_huggingface_datasets()
    cache_dir_value = str(cache_dir) if cache_dir is not None else None
    resolved_token = _resolve_huggingface_token(token)
    resolved_max_retries = max(int(max_retries), 1)
    resolved_backoff = max(float(retry_backoff_seconds), 0.0)

    load_kwargs = {
        "path": dataset_name,
        "name": config_name,
        "split": split,
        "cache_dir": cache_dir_value,
        "data_dir": data_dir,
    }
    if resolved_token is not None:
        load_kwargs["token"] = resolved_token
    if download_config_cls is not None:
        load_kwargs["download_config"] = download_config_cls(
            cache_dir=cache_dir_value,
            local_files_only=bool(local_files_only),
            max_retries=resolved_max_retries,
            token=resolved_token,
        )

    for attempt_index in range(resolved_max_retries):
        try:
            return load_dataset(**load_kwargs)
        except Exception as exc:
            is_rate_limit = _is_huggingface_rate_limit_error(exc)
            is_last_attempt = attempt_index + 1 >= resolved_max_retries
            if not is_rate_limit or is_last_attempt:
                if is_rate_limit:
                    raise RuntimeError(
                        "Hugging Face returned HTTP 429 while loading %s after %d attempt(s). "
                        "Set HF_TOKEN or pass --token, reuse --cache-dir, or rerun with --local-files-only after the cache is warm."
                        % (dataset_name, attempt_index + 1)
                    ) from exc
                raise
            sleep_seconds = resolved_backoff * (2 ** attempt_index)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


def station_config_key(station_id: Any, geometry_mode: GeometryMode) -> str:
    return "%s::%s" % (station_id, geometry_mode.value)


def _sanitize_artifact_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or "dataset"


def _normalize_dataset_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    return aliases.get(normalized, normalized or "train")


def _sanitize_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value(item) for item in list(value)[:32]]
    if isinstance(value, Mapping):
        sanitized = {}
        for key, nested_value in list(value.items())[:32]:
            if isinstance(nested_value, (bytes, bytearray)):
                sanitized[str(key)] = {"num_bytes": len(nested_value)}
            else:
                sanitized[str(key)] = _sanitize_metadata_value(nested_value)
        return sanitized
    if isinstance(value, (bytes, bytearray)):
        return {"num_bytes": len(value)}
    return str(value)


def _row_column(row: Mapping[str, Any], column_name: Optional[str], default: Any = None) -> Any:
    if column_name in (None, ""):
        return default
    return row.get(str(column_name), default)


def _to_grayscale_from_multichannel(image: np.ndarray) -> np.ndarray:
    channels = image[..., :3].astype(np.float32)
    grayscale = 0.299 * channels[..., 0] + 0.587 * channels[..., 1] + 0.114 * channels[..., 2]
    return np.clip(np.rint(grayscale), 0.0, 255.0).astype(np.uint8)


def _coerce_huggingface_image_to_uint8(image_value: Any, strict_grayscale: bool = True) -> np.ndarray:
    if isinstance(image_value, np.ndarray):
        image = image_value
    elif hasattr(image_value, "convert") and hasattr(image_value, "mode"):
        if strict_grayscale and str(image_value.mode) not in GRAYSCALE_IMAGE_MODES:
            raise ValueError("Expected a grayscale Hugging Face image but found mode %s." % image_value.mode)
        image = np.asarray(image_value.convert("L"))
    elif isinstance(image_value, Mapping):
        if "array" in image_value:
            image = np.asarray(image_value["array"])
        elif image_value.get("path"):
            return load_uint8_grayscale(Path(str(image_value["path"])))
        elif image_value.get("bytes") is not None:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError("Pillow is required to decode byte-backed Hugging Face images.") from exc
            with Image.open(io.BytesIO(image_value["bytes"])) as pil_image:
                return _coerce_huggingface_image_to_uint8(pil_image, strict_grayscale=strict_grayscale)
        else:
            raise TypeError("Unsupported Hugging Face image payload: %s" % sorted(image_value.keys()))
    else:
        raise TypeError("Unsupported Hugging Face image type: %s" % type(image_value).__name__)

    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image[..., 0]
        elif image.shape[-1] in (3, 4):
            if strict_grayscale:
                raise ValueError("Expected a grayscale Hugging Face image but found %d channels." % image.shape[-1])
            image = _to_grayscale_from_multichannel(image)
        else:
            raise ValueError("Unsupported Hugging Face image shape: %s" % (image.shape,))
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image after conversion, got shape %s." % (image.shape,))
    return normalize_uint8_image(image)


def _resolve_huggingface_geometry_mode(
    row: Mapping[str, Any],
    image_shape: Tuple[int, int],
    geometry_mode: str,
    geometry_mode_column: Optional[str],
) -> GeometryMode:
    explicit = _row_column(row, geometry_mode_column)
    if explicit not in (None, ""):
        return GeometryMode.from_value(str(explicit))
    if geometry_mode != "auto":
        return GeometryMode.from_value(geometry_mode)
    return GeometryMode.SQUARE if int(image_shape[0]) == int(image_shape[1]) else GeometryMode.RECT


def _resolve_huggingface_accept_reject(row: Mapping[str, Any], accept_reject_column: Optional[str]) -> int:
    value = _row_column(row, accept_reject_column)
    if value in (None, "", []):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 0 if int(value) == 0 else 1
    normalized = str(value).strip().lower()
    if normalized in {"0", "false", "accept", "ok", "good", "clean", "negative"}:
        return 0
    if normalized in {"1", "true", "reject", "ng", "nok", "bad", "defect", "positive"}:
        return 1
    raise ValueError("Unsupported accept/reject value %r for Hugging Face row." % value)


def _resolve_huggingface_defect_tags(row: Mapping[str, Any], defect_tags_column: Optional[str]) -> Tuple[str, ...]:
    value = _row_column(row, defect_tags_column)
    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        normalized = value.strip().lower()
        return (normalized,) if normalized else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return (str(value).strip().lower(),)


def _build_explicit_split_payload(
    records: Sequence[DatasetRecord],
    output_path: Optional[Path | str] = None,
    source: str = "explicit",
    grouping_keys: Sequence[str] = (),
) -> dict:
    payload = {
        "seed": None,
        "ratios": None,
        "grouping_keys": list(grouping_keys),
        "assignments": {record.sample_id: record.split for record in records},
        "generated_at": utc_timestamp(),
        "source": source,
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def _resolve_huggingface_splits(
    dataset_name: str,
    config_name: Optional[str] = None,
    split_names: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path | str] = None,
    data_dir: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    max_retries: int = 4,
    retry_backoff_seconds: float = 5.0,
):
    if split_names:
        return [
            (
                str(split_name),
                _load_huggingface_dataset(
                    dataset_name,
                    config_name=config_name,
                    split=str(split_name),
                    cache_dir=cache_dir,
                    data_dir=data_dir,
                    token=token,
                    local_files_only=local_files_only,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                ),
            )
            for split_name in split_names
        ]
    dataset_bundle = _load_huggingface_dataset(
        dataset_name,
        config_name=config_name,
        cache_dir=cache_dir,
        data_dir=data_dir,
        token=token,
        local_files_only=local_files_only,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    if hasattr(dataset_bundle, "items"):
        return [(str(split_name), split_dataset) for split_name, split_dataset in dataset_bundle.items()]
    return [("train", dataset_bundle)]


def serialize_box(box: BoxAnnotation) -> dict:
    return {
        "xyxy": list(box.xyxy),
        "defect_tag": box.defect_tag,
        "confidence": float(box.confidence),
        "annotator": box.annotator,
        "is_hard_case": bool(box.is_hard_case),
    }


def deserialize_box(payload: Mapping[str, Any]) -> BoxAnnotation:
    return BoxAnnotation(
        xyxy=tuple(int(value) for value in payload["xyxy"]),
        defect_tag=str(payload.get("defect_tag", "unknown")),
        confidence=float(payload.get("confidence", 1.0)),
        annotator=str(payload.get("annotator", "unknown")),
        is_hard_case=bool(payload.get("is_hard_case", False)),
    )


def serialize_dataset_record(record: DatasetRecord) -> dict:
    return {
        "sample_id": record.sample_id,
        "image_path": record.image_path,
        "station_id": record.station_id,
        "product_family": record.product_family,
        "geometry_mode": record.geometry_mode.value,
        "accept_reject": int(record.accept_reject),
        "defect_tags": list(record.defect_tags),
        "boxes": [serialize_box(box) for box in record.boxes],
        "mask_path": record.mask_path,
        "split": record.split,
        "capture_metadata": dict(record.capture_metadata),
        "source_dataset": record.source_dataset,
        "review_state": record.review_state,
    }


def deserialize_dataset_record(payload: Mapping[str, Any]) -> DatasetRecord:
    return DatasetRecord(
        sample_id=str(payload["sample_id"]),
        image_path=str(payload["image_path"]),
        station_id=payload["station_id"],
        product_family=str(payload.get("product_family", "unknown")),
        geometry_mode=payload.get("geometry_mode", "rect"),
        accept_reject=int(payload.get("accept_reject", 0)),
        defect_tags=tuple(payload.get("defect_tags", ())),
        boxes=tuple(deserialize_box(box) for box in payload.get("boxes", ())),
        mask_path=payload.get("mask_path"),
        split=str(payload.get("split", "unspecified")),
        capture_metadata=dict(payload.get("capture_metadata", {})),
        source_dataset=str(payload.get("source_dataset", "unknown")),
        review_state=str(payload.get("review_state", "unreviewed")),
    )


def _infer_accept_reject(path_parts: Sequence[str], sidecar: Mapping[str, Any]) -> int:
    explicit = sidecar.get("accept_reject")
    if explicit is not None:
        return int(explicit)
    lowered = {part.lower() for part in path_parts}
    if lowered & POSITIVE_LABEL_FOLDERS:
        return 1
    if lowered & NEGATIVE_LABEL_FOLDERS:
        return 0
    return 0


def _infer_station_id(path_parts: Sequence[str], sidecar: Mapping[str, Any]) -> str:
    explicit = first_nonempty(sidecar, ("station_id", "station", "line_id"))
    if explicit is not None:
        return str(explicit)
    for part in reversed(path_parts):
        lowered = part.lower()
        if lowered.startswith("station"):
            return part
    return "station-default"


def _infer_product_family(path_parts: Sequence[str], sidecar: Mapping[str, Any]) -> str:
    explicit = first_nonempty(sidecar, ("product_family", "family"))
    if explicit is not None:
        return str(explicit)
    for part in reversed(path_parts):
        lowered = part.lower()
        if lowered in PRODUCT_FAMILIES:
            return "syringe" if lowered.startswith("syringe") else "vial"
    return "unknown"


def _infer_geometry_mode(path_parts: Sequence[str], sidecar: Mapping[str, Any], image_shape: Tuple[int, int]) -> GeometryMode:
    explicit = first_nonempty(sidecar, ("geometry_mode", "geometry"))
    if explicit is not None:
        return GeometryMode.from_value(str(explicit))
    lowered_parts = {part.lower() for part in path_parts}
    if "square" in lowered_parts:
        return GeometryMode.SQUARE
    if "rect" in lowered_parts or "rectangular" in lowered_parts:
        return GeometryMode.RECT
    if image_shape[0] == image_shape[1]:
        return GeometryMode.SQUARE
    return GeometryMode.RECT


def _infer_defect_tags(path_parts: Sequence[str], sidecar: Mapping[str, Any]) -> Tuple[str, ...]:
    explicit = first_nonempty(sidecar, ("defect_tags", "defects"))
    if explicit is not None:
        return tuple(str(tag) for tag in explicit)
    tags = []
    reserved = POSITIVE_LABEL_FOLDERS | NEGATIVE_LABEL_FOLDERS | PRODUCT_FAMILIES | {"square", "rect", "rectangular"}
    for part in path_parts:
        lowered = part.lower()
        if lowered in reserved or lowered.startswith("station"):
            continue
        if lowered in {"train", "val", "validation", "test", "dataset", "images"}:
            continue
        if lowered not in tags:
            tags.append(lowered)
    return tuple(tags)


def _read_sidecar(image_path: Path) -> Mapping[str, Any]:
    sidecar_path = image_path.with_suffix(".json")
    if sidecar_path.exists():
        return read_json(sidecar_path)
    return {}


def _record_from_path(image_path: Path, root_dir: Path, source_dataset: str) -> DatasetRecord:
    image = load_uint8_grayscale(image_path)
    sidecar = _read_sidecar(image_path)
    relative_parts = image_path.relative_to(root_dir).parts[:-1]
    station_id = _infer_station_id(relative_parts, sidecar)
    product_family = _infer_product_family(relative_parts, sidecar)
    geometry_mode = _infer_geometry_mode(relative_parts, sidecar, image.shape)
    accept_reject = _infer_accept_reject(relative_parts, sidecar)
    defect_tags = _infer_defect_tags(relative_parts, sidecar)
    sample_id = str(image_path.relative_to(root_dir)).replace("\\", "/")
    box_payloads = sidecar.get("boxes", ())
    boxes = tuple(deserialize_box(payload) for payload in box_payloads)
    capture_metadata = dict(sidecar.get("capture_metadata", {}))
    capture_metadata.setdefault("image_shape", list(image.shape))
    for key in ("capture_day", "batch_id", "camera_id", "defect_scale"):
        if key in sidecar and key not in capture_metadata:
            capture_metadata[key] = sidecar[key]
    return DatasetRecord(
        sample_id=sample_id,
        image_path=str(image_path.resolve()),
        station_id=station_id,
        product_family=product_family,
        geometry_mode=geometry_mode,
        accept_reject=accept_reject,
        defect_tags=tuple(sidecar.get("defect_tags", defect_tags)),
        boxes=boxes,
        mask_path=sidecar.get("mask_path"),
        split=str(sidecar.get("split", "unspecified")),
        capture_metadata=capture_metadata,
        source_dataset=str(sidecar.get("source_dataset", source_dataset)),
        review_state=str(sidecar.get("review_state", "unreviewed")),
    )


def scan_folder_dataset(root_dir: Path | str, source_dataset: str = "folder_import") -> List[DatasetRecord]:
    root_path = Path(root_dir)
    records: List[DatasetRecord] = []
    for image_path in sorted(root_path.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        records.append(_record_from_path(image_path, root_path, source_dataset=source_dataset))
    return records


def ingest_dataset_manifest(root_dir: Path | str, output_dir: Optional[Path | str] = None, source_dataset: str = "folder_import") -> DatasetIndex:
    return build_dataset_manifest(root_dir=root_dir, output_dir=output_dir, source_dataset=source_dataset)


def save_dataset_manifest(records: Sequence[DatasetRecord], manifest_path: Path | str) -> Path:
    manifest_path = Path(manifest_path)
    write_jsonl(manifest_path, [serialize_dataset_record(record) for record in records])
    return manifest_path


def load_dataset_manifest(manifest_path: Path | str) -> List[DatasetRecord]:
    return [deserialize_dataset_record(record) for record in read_jsonl(Path(manifest_path))]


def _station_config_from_record_group(records: Sequence[DatasetRecord]) -> StationConfig:
    max_height = 0
    max_width = 0
    station_id = records[0].station_id
    geometry_mode = records[0].geometry_mode
    for record in records:
        shape = record.capture_metadata.get("image_shape")
        if shape is None:
            shape = load_uint8_grayscale(Path(record.image_path)).shape
        height, width = int(shape[0]), int(shape[1])
        max_height = max(max_height, height)
        max_width = max(max_width, width)
    if geometry_mode is GeometryMode.SQUARE:
        side = max(max_height, max_width)
        canvas_shape = (side, side)
    else:
        canvas_shape = (max_height, max_width)
    return StationConfig(
        canvas_shape=canvas_shape,
        station_id=station_id,
        geometry_mode=geometry_mode,
        pad_value=0,
        normalization_mean=127.5,
        normalization_std=50.0,
        tile_size=(64, 64),
        tile_stride=(32, 32),
        adapter_id=station_config_key(station_id, geometry_mode),
        reject_threshold=0.5,
        metadata={"source": "inferred"},
    )


def infer_station_configs(records: Sequence[DatasetRecord]) -> Dict[str, StationConfig]:
    grouped: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[station_config_key(record.station_id, record.geometry_mode)].append(record)
    return {key: _station_config_from_record_group(group) for key, group in grouped.items()}


def serialize_station_config(config: StationConfig) -> dict:
    return {
        "canvas_shape": list(config.canvas_shape),
        "station_id": config.station_id,
        "geometry_mode": config.geometry_mode.value if config.geometry_mode is not None else None,
        "pad_value": config.pad_value,
        "normalization_mean": config.normalization_mean,
        "normalization_std": config.normalization_std,
        "tile_size": list(config.tile_size_2d),
        "tile_stride": list(config.tile_stride_2d),
        "adapter_id": config.adapter_id,
        "reject_threshold": config.reject_threshold,
        "defect_thresholds": dict(config.defect_thresholds),
        "metadata": dict(config.metadata),
    }


def deserialize_station_config(payload: Mapping[str, Any]) -> StationConfig:
    return StationConfig(
        canvas_shape=tuple(payload["canvas_shape"]),
        station_id=payload.get("station_id", 0),
        geometry_mode=payload.get("geometry_mode"),
        pad_value=int(payload.get("pad_value", 0)),
        normalization_mean=float(payload.get("normalization_mean", 127.5)),
        normalization_std=float(payload.get("normalization_std", 50.0)),
        tile_size=tuple(payload.get("tile_size", (64, 64))),
        tile_stride=tuple(payload.get("tile_stride", (32, 32))),
        adapter_id=payload.get("adapter_id", 0),
        reject_threshold=float(payload.get("reject_threshold", 0.5)),
        defect_thresholds=dict(payload.get("defect_thresholds", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def build_dataset_ontology(records: Sequence[DatasetRecord], output_path: Optional[Path | str] = None, version: str = "1.0") -> dict:
    ontology = {
        "ontology_version": version,
        "defect_tags": sorted({tag for record in records for tag in record.defect_tags}),
        "product_families": sorted({record.product_family for record in records}),
        "geometry_modes": sorted({record.geometry_mode.value for record in records}),
        "generated_at": utc_timestamp(),
    }
    if output_path is not None:
        write_json(Path(output_path), ontology)
    return ontology


def _split_bucket(group_identifier: str, ratios: Tuple[float, float, float], seed: int) -> str:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")
    bucket = stable_int_hash("%s::%s" % (seed, group_identifier)) / float(2 ** 32)
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _group_identifier(record: DatasetRecord, grouping_keys: Sequence[str]) -> str:
    values = []
    for key in grouping_keys:
        if key == "station_id":
            values.append(str(record.station_id))
            continue
        if key == "geometry_mode":
            values.append(record.geometry_mode.value)
            continue
        values.append(str(record.capture_metadata.get(key, "missing")))
    return "|".join(values)


def build_dataset_splits(
    manifest_path: Path | str,
    output_path: Optional[Path | str] = None,
    seed: int = 17,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    grouping_keys: Sequence[str] = ("station_id", "capture_day", "batch_id", "camera_id"),
) -> dict:
    manifest_path = Path(manifest_path)
    records = load_dataset_manifest(manifest_path)
    assignments = {}
    updated_records = []
    for record in records:
        group_id = _group_identifier(record, grouping_keys)
        split = _split_bucket(group_id, ratios=ratios, seed=seed)
        assignments[record.sample_id] = split
        updated_records.append(
            DatasetRecord(
                sample_id=record.sample_id,
                image_path=record.image_path,
                station_id=record.station_id,
                product_family=record.product_family,
                geometry_mode=record.geometry_mode,
                accept_reject=record.accept_reject,
                defect_tags=record.defect_tags,
                boxes=record.boxes,
                mask_path=record.mask_path,
                split=split,
                capture_metadata=record.capture_metadata,
                source_dataset=record.source_dataset,
                review_state=record.review_state,
            )
        )
    save_dataset_manifest(updated_records, manifest_path)
    payload = {
        "seed": seed,
        "ratios": list(ratios),
        "grouping_keys": list(grouping_keys),
        "assignments": assignments,
        "generated_at": utc_timestamp(),
    }
    output_path = Path(output_path) if output_path is not None else manifest_path.with_name("splits.json")
    write_json(output_path, payload)
    return payload


def infer_defect_scale(record: DatasetRecord) -> str:
    explicit = record.capture_metadata.get("defect_scale")
    if explicit:
        return str(explicit)
    if not record.boxes:
        return "clean" if record.accept_reject == 0 else "unknown"
    max_area = max(box.area for box in record.boxes)
    if max_area <= 25:
        return "tiny"
    if max_area <= 1024:
        return "medium"
    return "global"


def build_hard_negative_subset(
    manifest_path: Path | str,
    output_path: Optional[Path | str] = None,
    predictions_path: Optional[Path | str] = None,
    score_threshold: float = 0.5,
) -> Path:
    records = load_dataset_manifest(manifest_path)
    scores: Dict[str, float] = {}
    if predictions_path is not None and Path(predictions_path).exists():
        for row in read_jsonl(Path(predictions_path)):
            scores[str(row["sample_id"])] = float(row.get("reject_score", 0.0))

    selected = []
    for record in records:
        if record.accept_reject != 0:
            continue
        score = scores.get(record.sample_id, 0.0)
        if score >= score_threshold or record.review_state in {"hard_negative", "review"}:
            selected.append(serialize_dataset_record(record))
    output_path = Path(output_path) if output_path is not None else Path(manifest_path).with_name("hard_negatives.jsonl")
    write_jsonl(output_path, selected)
    return output_path


def validate_dataset_records(records: Sequence[DatasetRecord]) -> dict:
    errors: List[str] = []
    warnings: List[str] = []
    seen_ids = set()
    for record in records:
        if record.sample_id in seen_ids:
            errors.append("Duplicate sample_id: %s" % record.sample_id)
        seen_ids.add(record.sample_id)
        image_path = Path(record.image_path)
        if not image_path.exists():
            errors.append("Missing image: %s" % image_path)
            continue
        try:
            image = load_uint8_grayscale(image_path)
        except Exception as exc:
            errors.append("Failed to load %s: %s" % (image_path, exc))
            continue
        if image.dtype != np.uint8:
            errors.append("Non-uint8 image: %s" % image_path)
        if image.ndim != 2:
            errors.append("Non-grayscale image: %s" % image_path)
        for box in record.boxes:
            x1, y1, x2, y2 = box.xyxy
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                errors.append("Box outside image bounds for %s" % record.sample_id)
        if record.accept_reject == 0 and record.defect_tags:
            warnings.append("Clean sample has defect tags: %s" % record.sample_id)
    return {
        "num_records": len(records),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def validate_dataset_manifest(manifest_path: Path | str) -> dict:
    records = load_dataset_manifest(manifest_path)
    report = validate_dataset_records(records)
    report["manifest_path"] = str(manifest_path)
    return report


def save_dataset_index(index: DatasetIndex, path: Path | str) -> Path:
    payload = asdict(index)
    return write_json(Path(path), payload)


def load_dataset_index(path: Path | str) -> DatasetIndex:
    payload = read_json(Path(path))
    return DatasetIndex(
        manifest_version=payload["manifest_version"],
        ontology_version=payload["ontology_version"],
        root_dir=payload["root_dir"],
        manifest_path=payload["manifest_path"],
        splits_path=payload["splits_path"],
        ontology_path=payload["ontology_path"],
        hard_negatives_path=payload["hard_negatives_path"],
        index_path=payload.get("index_path"),
        split_seed=int(payload.get("split_seed", 17)),
        grouping_keys=tuple(payload.get("grouping_keys", ("station_id", "capture_day", "batch_id", "camera_id"))),
        split_assignments=dict(payload.get("split_assignments", {})),
        hard_negative_ids=tuple(payload.get("hard_negative_ids", ())),
        review_subset_ids=tuple(payload.get("review_subset_ids", ())),
        station_configs=dict(payload.get("station_configs", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def build_dataset_manifest(
    root_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    source_dataset: str = "folder_import",
    seed: int = 17,
) -> DatasetIndex:
    root_dir = Path(root_dir)
    output_dir = ensure_dir(Path(output_dir) if output_dir is not None else root_dir / "_greymodel")
    records = scan_folder_dataset(root_dir, source_dataset=source_dataset)
    station_configs = infer_station_configs(records)
    manifest_path = output_dir / "manifest.jsonl"
    ontology_path = output_dir / "ontology.json"
    splits_path = output_dir / "splits.json"
    hard_negatives_path = output_dir / "hard_negatives.jsonl"
    index_path = output_dir / "dataset_index.json"
    save_dataset_manifest(records, manifest_path)
    ontology = build_dataset_ontology(records, output_path=ontology_path)
    split_payload = build_dataset_splits(manifest_path=manifest_path, output_path=splits_path, seed=seed)
    write_jsonl(hard_negatives_path, [])
    index = DatasetIndex(
        manifest_version="1.0",
        ontology_version=str(ontology["ontology_version"]),
        root_dir=str(root_dir.resolve()),
        manifest_path=str(manifest_path.resolve()),
        splits_path=str(splits_path.resolve()),
        ontology_path=str(ontology_path.resolve()),
        hard_negatives_path=str(hard_negatives_path.resolve()),
        index_path=str(index_path.resolve()),
        split_seed=seed,
        grouping_keys=tuple(split_payload["grouping_keys"]),
        split_assignments=dict(split_payload["assignments"]),
        hard_negative_ids=(),
        review_subset_ids=(),
        station_configs={key: serialize_station_config(config) for key, config in station_configs.items()},
        metadata={"created_at": utc_timestamp(), "num_records": len(records)},
    )
    save_dataset_index(index, index_path)
    return index


def build_huggingface_dataset_manifest(
    dataset_name: str,
    output_dir: Path | str,
    config_name: Optional[str] = None,
    split_names: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    image_column: str = "image",
    station_id: Any = "hf-public",
    product_family: str = "unknown",
    geometry_mode: str = "auto",
    source_dataset: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
    max_records: Optional[int] = None,
    accept_reject_column: Optional[str] = None,
    defect_tags_column: Optional[str] = None,
    station_id_column: Optional[str] = None,
    product_family_column: Optional[str] = None,
    geometry_mode_column: Optional[str] = None,
    metadata_columns: Sequence[str] = (),
    strict_grayscale: bool = True,
    token: Optional[str] = None,
    local_files_only: bool = False,
    max_retries: int = 4,
    retry_backoff_seconds: float = 5.0,
) -> DatasetIndex:
    output_dir = ensure_dir(Path(output_dir))
    materialized_root = ensure_dir(output_dir / "images")
    source_name = str(source_dataset or dataset_name)
    records: List[DatasetRecord] = []
    resolved_splits = _resolve_huggingface_splits(
        dataset_name=dataset_name,
        config_name=config_name,
        split_names=split_names,
        cache_dir=cache_dir,
        data_dir=data_dir,
        token=token,
        local_files_only=local_files_only,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    if not resolved_splits:
        raise ValueError("No Hugging Face splits were resolved for %s." % dataset_name)

    remaining_records = None if max_records is None else max(int(max_records), 0)
    for original_split_name, split_dataset in resolved_splits:
        if remaining_records is not None and remaining_records <= 0:
            break
        normalized_split = _normalize_dataset_split_name(original_split_name)
        split_dir_name = _sanitize_artifact_component(original_split_name or normalized_split)
        split_image_root = ensure_dir(materialized_root / split_dir_name)
        for row_index, row in enumerate(split_dataset):
            if remaining_records is not None and remaining_records <= 0:
                break
            if image_column not in row:
                raise KeyError("Column %r is not present in Hugging Face dataset %s." % (image_column, dataset_name))
            image = _coerce_huggingface_image_to_uint8(row[image_column], strict_grayscale=strict_grayscale)
            image_path = split_image_root / ("%08d.npy" % row_index)
            np.save(image_path, image)
            auto_station_column = station_id_column if station_id_column is not None else ("station_id" if "station_id" in row else None)
            auto_product_column = product_family_column if product_family_column is not None else ("product_family" if "product_family" in row else None)
            auto_geometry_column = geometry_mode_column if geometry_mode_column is not None else ("geometry_mode" if "geometry_mode" in row else None)
            resolved_geometry_mode = _resolve_huggingface_geometry_mode(
                row,
                image.shape,
                geometry_mode=geometry_mode,
                geometry_mode_column=auto_geometry_column,
            )
            resolved_station_id = _row_column(row, auto_station_column, station_id)
            if auto_station_column is None and geometry_mode == "auto":
                resolved_station_id = "%s-%s" % (resolved_station_id, resolved_geometry_mode.value)
            resolved_product_family = str(_row_column(row, auto_product_column, product_family))
            base_capture_metadata = _sanitize_metadata_value(row.get("capture_metadata", {}))
            if not isinstance(base_capture_metadata, Mapping):
                base_capture_metadata = {}
            capture_metadata = {
                **dict(base_capture_metadata),
                "image_shape": list(image.shape),
                "hf_dataset_name": dataset_name,
                "hf_config_name": config_name,
                "hf_original_split": str(original_split_name),
                "hf_row_index": int(row_index),
            }
            for column_name in metadata_columns:
                if column_name in row:
                    capture_metadata[str(column_name)] = _sanitize_metadata_value(row[column_name])
            records.append(
                DatasetRecord(
                    sample_id=str(row.get("sample_id", "%s/%08d.npy" % (split_dir_name, row_index))),
                    image_path=str(image_path.resolve()),
                    station_id=resolved_station_id,
                    product_family=resolved_product_family,
                    geometry_mode=resolved_geometry_mode,
                    accept_reject=_resolve_huggingface_accept_reject(row, accept_reject_column),
                    defect_tags=_resolve_huggingface_defect_tags(row, defect_tags_column),
                    boxes=(),
                    mask_path=None,
                    split=normalized_split,
                    capture_metadata=capture_metadata,
                    source_dataset=source_name,
                    review_state="unreviewed",
                )
            )
            if remaining_records is not None:
                remaining_records -= 1
    if not records:
        raise ValueError("No records were materialized from Hugging Face dataset %s." % dataset_name)

    station_configs = infer_station_configs(records)
    manifest_path = output_dir / "manifest.jsonl"
    ontology_path = output_dir / "ontology.json"
    splits_path = output_dir / "splits.json"
    hard_negatives_path = output_dir / "hard_negatives.jsonl"
    index_path = output_dir / "dataset_index.json"
    save_dataset_manifest(records, manifest_path)
    ontology = build_dataset_ontology(records, output_path=ontology_path)
    split_payload = _build_explicit_split_payload(records, output_path=splits_path, source="huggingface")
    write_jsonl(hard_negatives_path, [])
    index = DatasetIndex(
        manifest_version="1.0",
        ontology_version=str(ontology["ontology_version"]),
        root_dir=str(materialized_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        splits_path=str(splits_path.resolve()),
        ontology_path=str(ontology_path.resolve()),
        hard_negatives_path=str(hard_negatives_path.resolve()),
        index_path=str(index_path.resolve()),
        split_seed=0,
        grouping_keys=(),
        split_assignments=dict(split_payload["assignments"]),
        hard_negative_ids=(),
        review_subset_ids=(),
        station_configs={key: serialize_station_config(config) for key, config in station_configs.items()},
        metadata={
            "created_at": utc_timestamp(),
            "num_records": len(records),
            "source": "huggingface",
            "dataset_name": dataset_name,
            "config_name": config_name,
            "data_dir": data_dir,
            "requested_splits": list(split_names or ()),
            "image_column": image_column,
            "strict_grayscale": bool(strict_grayscale),
            "max_records": max_records,
            "metadata_columns": list(metadata_columns),
            "local_files_only": bool(local_files_only),
            "max_retries": int(max_retries),
        },
    )
    save_dataset_index(index, index_path)
    return index


build_hf_dataset_manifest = build_huggingface_dataset_manifest
import_huggingface_dataset = build_huggingface_dataset_manifest


def load_station_configs_from_index(index: DatasetIndex | Path | str) -> Dict[str, StationConfig]:
    if not isinstance(index, DatasetIndex):
        index = load_dataset_index(index)
    return {key: deserialize_station_config(value) for key, value in index.station_configs.items()}


def station_config_for_record(record: DatasetRecord, station_configs: Mapping[str, StationConfig]) -> StationConfig:
    key = station_config_key(record.station_id, record.geometry_mode)
    if key not in station_configs:
        raise KeyError("Missing StationConfig for %s." % key)
    return station_configs[key]


class GreyInspectionDataset:
    def __init__(self, samples: Sequence[Sample], station_configs: Sequence[StationConfig]) -> None:
        self.samples = list(samples)
        self.station_configs = {config.station_id: config for config in station_configs}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        config = self.station_configs[sample.station_id]
        return {
            "sample": sample,
            "prepared": preprocess_sample(sample, config),
            "station_config": config,
        }


class ManifestInspectionDataset:
    def __init__(
        self,
        manifest_path: Path | str,
        index_path: Optional[Path | str] = None,
        split: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.records = load_dataset_manifest(self.manifest_path)
        if split is not None:
            self.records = [record for record in self.records if record.split == split]
        if index_path is None:
            candidate = self.manifest_path.with_name("dataset_index.json")
            self.index = load_dataset_index(candidate) if candidate.exists() else None
        else:
            self.index = load_dataset_index(index_path)
        self.station_configs = load_station_configs_from_index(self.index) if self.index is not None else infer_station_configs(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def station_balanced_indices(self) -> List[int]:
        return station_balanced_record_order(self.records)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        record = self.records[index]
        image = load_uint8_grayscale(Path(record.image_path))
        sample = record.to_sample(image)
        station_config = station_config_for_record(record, self.station_configs)
        return {
            "record": record,
            "sample": sample,
            "prepared": preprocess_sample(sample, station_config),
            "station_config": station_config,
        }


class StationBalancedManifestSampler:
    def __init__(
        self,
        records: Sequence[DatasetRecord],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = False,
        seed: int = 17,
        drop_last: bool = False,
    ) -> None:
        self.records = list(records)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        base_length = len(self.records)
        if self.num_replicas == 1:
            return base_length
        if self.drop_last:
            return base_length // self.num_replicas
        return int(np.ceil(base_length / float(self.num_replicas)))

    def __iter__(self) -> Iterator[int]:
        indices = station_balanced_record_order(self.records)
        if self.shuffle and indices:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.num_replicas == 1:
            return iter(indices)
        if self.drop_last:
            usable = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:usable]
        else:
            while len(indices) % self.num_replicas != 0 and indices:
                indices.append(indices[len(indices) % len(indices)])
        return iter(indices[self.rank :: self.num_replicas])


class DistributedShardedSampler:
    def __init__(
        self,
        indices: Sequence[int],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 17,
        drop_last: bool = False,
    ) -> None:
        self.indices = list(indices)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        base_length = len(self.indices)
        if self.num_replicas == 1:
            return base_length
        if self.drop_last:
            return base_length // self.num_replicas
        return int(np.ceil(base_length / float(self.num_replicas)))

    def __iter__(self) -> Iterator[int]:
        indices = list(self.indices)
        if self.shuffle and indices:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.num_replicas == 1:
            return iter(indices)
        if self.drop_last:
            usable = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:usable]
        else:
            while len(indices) % self.num_replicas != 0 and indices:
                indices.append(indices[len(indices) % len(indices)])
        return iter(indices[self.rank :: self.num_replicas])


def register_synthetic_recipe(
    index_path: Path | str,
    recipe_name: str,
    recipe_payload: Mapping[str, Any],
) -> DatasetIndex:
    path = Path(index_path)
    index = load_dataset_index(path)
    metadata = dict(index.metadata)
    recipes = list(metadata.get("synthetic_recipes", ()))
    recipes.append(
        {
            "name": str(recipe_name),
            "payload": dict(recipe_payload),
            "registered_at": utc_timestamp(),
        }
    )
    metadata["synthetic_recipes"] = recipes
    updated = DatasetIndex(
        manifest_version=index.manifest_version,
        ontology_version=index.ontology_version,
        root_dir=index.root_dir,
        manifest_path=index.manifest_path,
        splits_path=index.splits_path,
        ontology_path=index.ontology_path,
        hard_negatives_path=index.hard_negatives_path,
        index_path=index.index_path or str(path.resolve()),
        split_seed=index.split_seed,
        grouping_keys=index.grouping_keys,
        split_assignments=index.split_assignments,
        hard_negative_ids=index.hard_negative_ids,
        review_subset_ids=index.review_subset_ids,
        station_configs=index.station_configs,
        metadata=metadata,
    )
    save_dataset_index(updated, path)
    return updated


def group_samples_by_station(samples: Iterable[Sample]) -> Dict[int, List[Sample]]:
    grouped: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.station_id].append(sample)
    return dict(grouped)


def station_balanced_record_order(records: Sequence[DatasetRecord]) -> List[int]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record.station_id)].append(index)
    max_length = max((len(indices) for indices in grouped.values()), default=0)
    ordered = []
    for offset in range(max_length):
        for station_key in sorted(grouped):
            indices = grouped[station_key]
            if offset < len(indices):
                ordered.append(indices[offset])
    return ordered


def collate_batch(items, as_torch: bool = True) -> Mapping[str, object]:
    prepared = [item["prepared"] for item in items]
    samples = [item["sample"] for item in items]
    model_input = stack_prepared_images(prepared, as_torch=as_torch)
    return {
        "model_input": model_input,
        "accept_reject": [sample.accept_reject for sample in samples],
        "defect_tags": [sample.defect_tags for sample in samples],
        "samples": samples,
        "records": [item.get("record") for item in items],
    }
