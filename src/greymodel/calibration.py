from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class StationCalibration:
    station_id: int
    reject_threshold: float = 0.5
    temperature: float = 1.0
    defect_thresholds: Mapping[str, float] = None


@dataclass(frozen=True)
class CalibratedStationDecision:
    station_id: int
    reject_score: float
    reject_threshold: float
    reject: bool
    temperature: float
    defect_scores: Mapping[str, float]
    defect_flags: Mapping[str, bool]


class StationCalibrator:
    def __init__(self, calibrations: Iterable[StationCalibration]) -> None:
        self._calibrations: Dict[int, StationCalibration] = {
            calibration.station_id: calibration for calibration in calibrations
        }

    def get(self, station_id: int) -> Optional[StationCalibration]:
        return self._calibrations.get(station_id)

    @staticmethod
    def _sigmoid(logit: float) -> float:
        if logit >= 0:
            exp_term = pow(2.718281828459045, -logit)
            return 1.0 / (1.0 + exp_term)
        exp_term = pow(2.718281828459045, logit)
        return exp_term / (1.0 + exp_term)

    def calibrate(
        self,
        station_id: int,
        reject_logit: float,
        defect_logits: Mapping[str, float],
    ) -> CalibratedStationDecision:
        calibration = self._calibrations.get(station_id, StationCalibration(station_id=station_id))
        temperature = max(calibration.temperature, 1e-6)
        reject_score = self._sigmoid(reject_logit / temperature)
        reject = reject_score >= calibration.reject_threshold
        thresholds = calibration.defect_thresholds or {}
        defect_scores = {
            defect_name: self._sigmoid(defect_logit / temperature)
            for defect_name, defect_logit in defect_logits.items()
        }
        defect_flags = {
            defect_name: score >= thresholds.get(defect_name, 0.5)
            for defect_name, score in defect_scores.items()
        }
        return CalibratedStationDecision(
            station_id=station_id,
            reject_score=reject_score,
            reject_threshold=calibration.reject_threshold,
            reject=reject,
            temperature=temperature,
            defect_scores=defect_scores,
            defect_flags=defect_flags,
        )
