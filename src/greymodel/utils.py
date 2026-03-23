from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zlib

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_int_hash(value: str) -> int:
    return int(zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Object of type %s is not JSON serializable." % type(value).__name__)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=json_default))
            handle.write("\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default))
        handle.write("\n")
    return path


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError("Unsupported PGM format in %s." % path)

        def _read_non_comment() -> bytes:
            line = handle.readline()
            while line.startswith(b"#"):
                line = handle.readline()
            return line

        size_tokens = _read_non_comment().split()
        if len(size_tokens) != 2:
            raise ValueError("Invalid PGM image size in %s." % path)
        width, height = int(size_tokens[0]), int(size_tokens[1])
        max_value = int(_read_non_comment())
        if max_value > 255:
            raise ValueError("Only 8-bit PGM is supported.")
        buffer = handle.read(width * height)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
        return image.copy()


def write_pgm(path: Path, image_uint8: np.ndarray) -> Path:
    if image_uint8.dtype != np.uint8:
        raise TypeError("write_pgm expects uint8 data.")
    if image_uint8.ndim != 2:
        raise ValueError("write_pgm expects a 2D grayscale array.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "P5\n{} {}\n255\n".format(image_uint8.shape[1], image_uint8.shape[0]).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(image_uint8.tobytes())
    return path


def normalize_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale array.")
    working = image.astype(np.float32)
    working -= float(working.min(initial=0.0))
    max_value = float(working.max(initial=0.0))
    if max_value > 0:
        working = working / max_value
    return np.clip(working * 255.0, 0.0, 255.0).astype(np.uint8)


def load_uint8_grayscale(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path)
    elif suffix == ".pgm":
        image = _read_pgm(path)
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to read %s. Supported dependency-free formats are .npy and .pgm."
                % suffix
            ) from exc
        with Image.open(path) as image_file:
            image = np.asarray(image_file.convert("L"))
    if image.ndim != 2:
        raise ValueError("Expected grayscale image at %s." % path)
    if image.dtype != np.uint8:
        image = normalize_uint8_image(image)
    return image


def save_array_artifact(path_without_suffix: Path, array: np.ndarray) -> dict[str, str]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    npy_path = path_without_suffix.with_suffix(".npy")
    np.save(npy_path, array)
    result = {"npy": str(npy_path)}
    if array.ndim == 2:
        try:
            pgm_path = path_without_suffix.with_suffix(".pgm")
            write_pgm(pgm_path, normalize_uint8_image(array))
            result["pgm"] = str(pgm_path)
        except Exception:
            pass
    return result


def copy_text_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def listify(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", []):
            return mapping[key]
    return default
