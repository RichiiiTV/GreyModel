from __future__ import annotations

import numpy as np

from greymodel import StationCalibration, StationCalibrator, inject_defect, inject_particle, inject_scratch, inject_streak


def test_station_calibrator_applies_per_station_thresholds() -> None:
    calibrator = StationCalibrator(
        [
            StationCalibration(
                station_id=3,
                reject_threshold=0.7,
                temperature=1.0,
                defect_thresholds={"scratch": 0.6},
            )
        ]
    )

    decision = calibrator.calibrate(3, reject_logit=2.0, defect_logits={"scratch": 0.5})

    assert decision.station_id == 3
    assert decision.reject is True
    assert "scratch" in decision.defect_scores


def test_synthetic_defect_injection_helpers_return_modified_images() -> None:
    image = np.full((16, 16), 120, dtype=np.uint8)

    particle_image, particle = inject_particle(image, center=(8, 8), radius=2, intensity_delta=40)
    scratch_image, scratch = inject_scratch(image, start=(0, 0), end=(15, 15), thickness=1, intensity_delta=30)
    streak_image, streak = inject_streak(image, axis=1, intensity_delta=-25, width=4, position=4)

    assert particle.kind == "particle"
    assert scratch.kind == "scratch"
    assert streak.kind == "streak"
    assert particle_image.shape == image.shape
    assert scratch_image.shape == image.shape
    assert streak_image.shape == image.shape
    assert not np.array_equal(particle_image, image)
    assert not np.array_equal(scratch_image, image)
    assert not np.array_equal(streak_image, image)


def test_generic_synthetic_defect_dispatch_supports_known_kinds() -> None:
    image = np.full((16, 16), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)

    for kind in ("particle", "scratch", "streak"):
        modified, injection = inject_defect(image, rng, kind=kind)
        assert modified.shape == image.shape
        assert injection.kind == kind
