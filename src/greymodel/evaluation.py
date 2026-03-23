from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .api import BaseModel, LiteModel
from .data import (
    DatasetRecord,
    infer_defect_scale,
    load_dataset_index,
    load_dataset_manifest,
    load_station_configs_from_index,
    station_config_for_record,
)
from .types import ModelInput, PredictionRecord
from .utils import read_json, read_jsonl, write_json, write_jsonl


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def binary_confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "far": _safe_divide(fp, fp + tn),
        "frr": _safe_divide(fn, fn + tp),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
    }


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    comparisons = []
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                comparisons.append(1.0)
            elif positive == negative:
                comparisons.append(0.5)
            else:
                comparisons.append(0.0)
    return float(np.mean(comparisons))


def precision_recall_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    thresholds = sorted({float(value) for value in y_score.tolist()}, reverse=True)
    if not thresholds:
        thresholds = [0.5]
    return [binary_confusion(y_true, y_score, threshold=value) for value in thresholds]


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 21) -> list[dict]:
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    return [binary_confusion(y_true, y_score, threshold=float(value)) for value in thresholds]


def calibration_curve(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        start, stop = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (y_score >= start) & (y_score <= stop)
        else:
            mask = (y_score >= start) & (y_score < stop)
        if not np.any(mask):
            rows.append({"bin": index, "start": float(start), "stop": float(stop), "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        rows.append(
            {
                "bin": index,
                "start": float(start),
                "stop": float(stop),
                "count": int(mask.sum()),
                "confidence": float(np.mean(y_score[mask])),
                "accuracy": float(np.mean(y_true[mask])),
            }
        )
    return rows


def _defect_family_probs(output, defect_families: Sequence[str]) -> Dict[str, float]:
    values = np.asarray(output.defect_family_probs, dtype=np.float32).reshape(-1)
    return {
        defect_families[index]: float(values[index]) for index in range(min(len(defect_families), len(values)))
    }


def predict_dataset(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    num_defect_families: Optional[int] = None,
) -> list[PredictionRecord]:
    records = load_dataset_manifest(manifest_path)
    if index_path is None:
        candidate = Path(manifest_path).with_name("dataset_index.json")
        index = load_dataset_index(candidate) if candidate.exists() else None
    else:
        index = load_dataset_index(index_path)
    station_configs = load_station_configs_from_index(index) if index is not None else {}
    ontology = read_json(Path(index.ontology_path)) if index is not None and Path(index.ontology_path).exists() else {}
    defect_families = tuple(ontology.get("defect_tags", []))
    if num_defect_families is None:
        num_defect_families = max(len(defect_families), 1)
    model = BaseModel(num_defect_families=num_defect_families, defect_families=defect_families) if variant == "base" else LiteModel(num_defect_families=num_defect_families, defect_families=defect_families)
    predictions = []
    from .utils import load_uint8_grayscale

    for record in records:
        image = load_uint8_grayscale(Path(record.image_path))
        model_input = ModelInput(
            image_uint8=image,
            station_id=record.station_id,
            geometry_mode=record.geometry_mode,
            metadata=record.capture_metadata,
        )
        if not station_configs:
            raise ValueError("Station configs are required for dataset prediction.")
        station_config = station_config_for_record(record, station_configs)
        output = model.forward(model_input, station_config)
        reject_score = float(np.asarray(output.reject_score).reshape(()))
        predictions.append(
            PredictionRecord(
                sample_id=record.sample_id,
                station_id=record.station_id,
                accept_reject=record.accept_reject,
                reject_score=reject_score,
                predicted_label=int(reject_score >= station_config.reject_threshold),
                defect_probs=_defect_family_probs(output, defect_families),
                split=record.split,
                defect_scale=infer_defect_scale(record),
                metadata={"product_family": record.product_family},
            )
        )
    return predictions


def save_predictions(predictions: Sequence[PredictionRecord], output_path: Path | str) -> Path:
    return write_jsonl(Path(output_path), [asdict(prediction) for prediction in predictions])


def load_predictions(predictions_path: Path | str) -> list[PredictionRecord]:
    return [PredictionRecord(**row) for row in read_jsonl(Path(predictions_path))]


def evaluate_predictions(records: Sequence[DatasetRecord], predictions: Sequence[PredictionRecord]) -> dict:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    paired_records = [record for record in records if record.sample_id in by_id]
    y_true = np.asarray([record.accept_reject for record in paired_records], dtype=np.int64)
    y_score = np.asarray([by_id[record.sample_id].reject_score for record in paired_records], dtype=np.float32)

    overall = binary_confusion(y_true, y_score)
    overall["auroc"] = binary_auroc(y_true, y_score)
    report = {
        "overall": overall,
        "threshold_sweep": threshold_sweep(y_true, y_score),
        "pr_curve": precision_recall_curve_points(y_true, y_score),
        "calibration_curve": calibration_curve(y_true, y_score),
        "per_station": {},
        "per_defect_scale": {},
        "per_defect_family": {},
    }

    station_groups: Dict[str, list[int]] = {}
    scale_groups: Dict[str, list[int]] = {}
    family_groups: Dict[str, list[int]] = {}
    for index, record in enumerate(paired_records):
        station_groups.setdefault(str(record.station_id), []).append(index)
        scale_groups.setdefault(infer_defect_scale(record), []).append(index)
        for tag in record.defect_tags:
            family_groups.setdefault(str(tag), []).append(index)

    for station, indices in station_groups.items():
        report["per_station"][station] = binary_confusion(y_true[indices], y_score[indices])
    for scale, indices in scale_groups.items():
        report["per_defect_scale"][scale] = binary_confusion(y_true[indices], y_score[indices])
    for family, indices in family_groups.items():
        report["per_defect_family"][family] = binary_confusion(y_true[indices], y_score[indices])
    return report


def benchmark_manifest(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    output_path: Optional[Path | str] = None,
) -> dict:
    records = load_dataset_manifest(manifest_path)
    predictions = predict_dataset(manifest_path, index_path=index_path, variant=variant)
    report = evaluate_predictions(records, predictions)
    if output_path is not None:
        write_json(Path(output_path), report)
    return report


def build_calibration_report(
    manifest_path: Path | str,
    predictions_path: Optional[Path | str] = None,
    index_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> dict:
    records = load_dataset_manifest(manifest_path)
    if predictions_path is None:
        predictions = predict_dataset(manifest_path, index_path=index_path)
    else:
        predictions = load_predictions(predictions_path)
    by_station: Dict[str, list[PredictionRecord]] = {}
    for prediction in predictions:
        by_station.setdefault(str(prediction.station_id), []).append(prediction)
    report = {"stations": {}, "num_records": len(records)}
    for station, station_predictions in by_station.items():
        scores = np.asarray([row.reject_score for row in station_predictions], dtype=np.float32)
        negatives = np.asarray([row.reject_score for row in station_predictions if row.accept_reject == 0], dtype=np.float32)
        threshold = float(np.quantile(negatives, 0.95)) if negatives.size else 0.5
        report["stations"][station] = {
            "temperature": 1.0,
            "recommended_reject_threshold": threshold,
            "score_mean": float(scores.mean()) if scores.size else 0.0,
            "score_std": float(scores.std()) if scores.size else 0.0,
        }
    if output_path is not None:
        write_json(Path(output_path), report)
    return report
