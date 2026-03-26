from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

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
from .explainability import build_explanation_bundle
from .types import HierarchicalPredictionRecord, ModelInput, PredictionEvidence, PredictionRecord
from .utils import load_uint8_grayscale, read_json, read_jsonl, write_json, write_jsonl


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
        "threshold": float(threshold),
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


def _variant_model(variant: str, num_defect_families: int, defect_families: Sequence[str]):
    if variant == "lite":
        return LiteModel(num_defect_families=num_defect_families, defect_families=defect_families)
    return BaseModel(num_defect_families=num_defect_families, defect_families=defect_families)


def _build_prediction_record(
    record: DatasetRecord,
    output,
    station_config,
    defect_families: Sequence[str],
    evidence: PredictionEvidence | None = None,
) -> PredictionRecord:
    reject_score = float(np.asarray(output.reject_score).reshape(()))
    predicted_label = int(reject_score >= float(station_config.reject_threshold))
    probs = _defect_family_probs(output, defect_families)
    station_decision = {
        "reject": bool(predicted_label),
        "threshold": float(station_config.reject_threshold),
        "station_id": str(record.station_id),
    }
    if output.station_temperature is not None:
        station_decision["temperature"] = float(np.asarray(output.station_temperature).reshape(()))
    return PredictionRecord(
        sample_id=record.sample_id,
        station_id=record.station_id,
        accept_reject=record.accept_reject,
        reject_score=reject_score,
        predicted_label=predicted_label,
        primary_label="bad" if predicted_label else "good",
        primary_score=reject_score,
        top_defect_family=max(probs.items(), key=lambda item: item[1])[0] if probs else None,
        defect_family_probs=probs,
        evidence=evidence
        or PredictionEvidence(
            station_decision=station_decision,
            metadata={
                "top_tile_indices": np.asarray(output.top_tile_indices).reshape(-1).tolist()
                if output.top_tile_indices is not None
                else [],
                "top_tile_boxes": np.asarray(output.top_tile_boxes).tolist()
                if output.top_tile_boxes is not None
                else [],
            },
        ),
        split=record.split,
        defect_scale=infer_defect_scale(record),
        metadata={"product_family": record.product_family, **dict(record.capture_metadata or {})},
    )


def predict_records(
    records: Sequence[DatasetRecord],
    *,
    index_path: Optional[Path | str] = None,
    station_configs: Optional[Mapping[str, Any]] = None,
    variant: str = "base",
    num_defect_families: Optional[int] = None,
    defect_families: Sequence[str] = (),
    evidence_root: Optional[Path | str] = None,
    evidence_policy: str = "none",
    on_error: Optional[Callable[[DatasetRecord, BaseException], None]] = None,
    continue_on_error: bool = False,
) -> list[PredictionRecord]:
    if station_configs is None:
        if index_path is None:
            raise ValueError("predict_records requires dataset index metadata or explicit station configs.")
        index = load_dataset_index(index_path)
        station_configs = load_station_configs_from_index(index)
        if not defect_families:
            ontology_path = Path(index.ontology_path)
            ontology = read_json(ontology_path) if ontology_path.exists() else {}
            defect_families = tuple(ontology.get("defect_tags", []))
    else:
        defect_families = tuple(defect_families)
    if num_defect_families is None:
        num_defect_families = max(len(defect_families), 1)
    model = _variant_model(variant, num_defect_families=num_defect_families, defect_families=defect_families)
    predictions: list[PredictionRecord] = []
    evidence_policy = str(evidence_policy or "none").lower()

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
            output = model.forward(model_input, station_config)
            provisional = _build_prediction_record(record, output, station_config, defect_families)
            write_bundle = evidence_policy == "all" or (evidence_policy == "bad" and provisional.primary_label == "bad")
            if write_bundle and evidence_root is not None:
                sample_dir = Path(evidence_root) / record.sample_id.replace("/", "_")
                bundle = build_explanation_bundle(model, model_input, station_config, sample_dir)
                evidence = PredictionEvidence(
                    heatmap_path=str(bundle["heatmap_path"]),
                    top_tiles_path=str(bundle["top_tiles_path"]),
                    sample_dir=str(sample_dir),
                    explanation_bundle_path=str(bundle["bundle_path"]),
                    station_decision=provisional.evidence.station_decision,
                    metadata={"prediction_path": str(bundle["prediction_path"])},
                )
                provisional = _build_prediction_record(record, output, station_config, defect_families, evidence=evidence)
            predictions.append(provisional)
        except BaseException as exc:
            if on_error is not None:
                on_error(record, exc)
                if continue_on_error:
                    continue
            raise
    return predictions


def predict_dataset(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    num_defect_families: Optional[int] = None,
    evidence_root: Optional[Path | str] = None,
    evidence_policy: str = "none",
    on_error: Optional[Callable[[DatasetRecord, BaseException], None]] = None,
    continue_on_error: bool = False,
) -> list[PredictionRecord]:
    records = load_dataset_manifest(manifest_path)
    if index_path is None:
        candidate = Path(manifest_path).with_name("dataset_index.json")
        index_path = candidate if candidate.exists() else None
    if index_path is None:
        raise ValueError("Station configs are required for dataset prediction.")
    return predict_records(
        records,
        index_path=index_path,
        variant=variant,
        num_defect_families=num_defect_families,
        evidence_root=evidence_root,
        evidence_policy=evidence_policy,
        on_error=on_error,
        continue_on_error=continue_on_error,
    )


def predict_hierarchical_dataset(
    manifest_path: Path | str,
    index_path: Optional[Path | str] = None,
    variant: str = "base",
    num_defect_families: Optional[int] = None,
    evidence_root: Optional[Path | str] = None,
    evidence_policy: str = "none",
    on_error: Optional[Callable[[DatasetRecord, BaseException], None]] = None,
    continue_on_error: bool = False,
) -> list[HierarchicalPredictionRecord]:
    predictions = predict_dataset(
        manifest_path,
        index_path=index_path,
        variant=variant,
        num_defect_families=num_defect_families,
        evidence_root=evidence_root,
        evidence_policy=evidence_policy,
        on_error=on_error,
        continue_on_error=continue_on_error,
    )
    return [
        HierarchicalPredictionRecord(
            sample_id=prediction.sample_id,
            station_id=prediction.station_id,
            accept_reject=prediction.accept_reject,
            primary_label=prediction.primary_label,
            primary_score=float(prediction.primary_score),
            predicted_label=prediction.predicted_label,
            reject_score=prediction.reject_score,
            top_defect_family=prediction.top_defect_family,
            defect_family_probs=prediction.defect_family_probs,
            split=prediction.split,
            defect_scale=prediction.defect_scale,
            evidence=prediction.evidence,
            metadata=prediction.metadata,
        )
        for prediction in predictions
    ]


def save_predictions(predictions: Sequence[PredictionRecord], output_path: Path | str) -> Path:
    return write_jsonl(Path(output_path), [asdict(prediction) for prediction in predictions])


def load_predictions(predictions_path: Path | str) -> list[PredictionRecord]:
    return [PredictionRecord(**row) for row in read_jsonl(Path(predictions_path))]


def _defect_family_report_bad_only(records: Sequence[DatasetRecord], predictions: Sequence[PredictionRecord]) -> dict:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    paired = [(record, by_id[record.sample_id]) for record in records if record.sample_id in by_id]
    paired = [(record, prediction) for record, prediction in paired if int(record.accept_reject) == 1 and tuple(record.defect_tags)]
    if not paired:
        return {"num_bad_records": 0, "top1_accuracy": 0.0, "per_family": {}}
    top1_hits = 0
    per_family: Dict[str, Dict[str, float]] = {}
    for record, prediction in paired:
        predicted_family = prediction.top_defect_family
        if predicted_family in record.defect_tags:
            top1_hits += 1
        for family in record.defect_tags:
            row = per_family.setdefault(family, {"count": 0, "top1_hits": 0, "mean_true_family_score": 0.0})
            row["count"] += 1
            if predicted_family == family:
                row["top1_hits"] += 1
            row["mean_true_family_score"] += float(prediction.defect_family_probs.get(family, 0.0))
    for family, row in per_family.items():
        count = max(int(row["count"]), 1)
        row["top1_accuracy"] = float(row["top1_hits"]) / float(count)
        row["mean_true_family_score"] = float(row["mean_true_family_score"]) / float(count)
    return {
        "num_bad_records": len(paired),
        "top1_accuracy": float(top1_hits) / float(len(paired)),
        "per_family": per_family,
    }


def evaluate_predictions(records: Sequence[DatasetRecord], predictions: Sequence[PredictionRecord]) -> dict:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    paired_records = [record for record in records if record.sample_id in by_id]
    if not paired_records:
        return {
            "overall": binary_confusion(np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)),
            "threshold_sweep": [],
            "pr_curve": [],
            "calibration_curve": [],
            "per_station": {},
            "per_defect_scale": {},
            "per_defect_family": {},
            "defect_family_bad_only": {"num_bad_records": 0, "top1_accuracy": 0.0, "per_family": {}},
        }
    y_true = np.asarray([record.accept_reject for record in paired_records], dtype=np.int64)
    y_score = np.asarray([by_id[record.sample_id].primary_score for record in paired_records], dtype=np.float32)

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
        "defect_family_bad_only": _defect_family_report_bad_only(paired_records, [by_id[record.sample_id] for record in paired_records]),
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
        scores = np.asarray([row.primary_score for row in station_predictions], dtype=np.float32)
        negatives = np.asarray([row.primary_score for row in station_predictions if row.accept_reject == 0], dtype=np.float32)
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
