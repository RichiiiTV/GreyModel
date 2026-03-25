from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class PretrainDatasetPreset:
    name: str
    dataset_name: str
    config_name: Optional[str] = None
    data_dir: Optional[str] = None
    source_dataset: Optional[str] = None
    allow_rgb_conversion: bool = False
    notes: str = ""


PRETRAIN_DATASET_REGISTRY: Dict[str, PretrainDatasetPreset] = {
    "ds_dagm": PretrainDatasetPreset(
        name="ds_dagm",
        dataset_name="DefectSpectrum/Defect_Spectrum",
        data_dir="DS-DAGM/image",
        source_dataset="DefectSpectrum/Defect_Spectrum:DS-DAGM",
        allow_rgb_conversion=False,
        notes="Industrial grayscale defect textures. Best public grayscale starter for defect-centric pretraining.",
    ),
    "defect_spectrum_full": PretrainDatasetPreset(
        name="defect_spectrum_full",
        dataset_name="DefectSpectrum/Defect_Spectrum",
        source_dataset="DefectSpectrum/Defect_Spectrum:full",
        allow_rgb_conversion=True,
        notes="Broader public defect corpus. Imported as 8-bit grayscale and recommended only with patch-based pretraining.",
    ),
    "mvtec_ad_gray": PretrainDatasetPreset(
        name="mvtec_ad_gray",
        dataset_name="Voxel51/mvtec-ad",
        source_dataset="Voxel51/mvtec-ad:grayscale",
        allow_rgb_conversion=True,
        notes="Public anomaly-detection dataset converted to grayscale during import.",
    ),
}


def list_pretrain_dataset_presets() -> Mapping[str, PretrainDatasetPreset]:
    return dict(PRETRAIN_DATASET_REGISTRY)


def get_pretrain_dataset_preset(name: str) -> PretrainDatasetPreset:
    normalized = str(name).strip().lower()
    if normalized not in PRETRAIN_DATASET_REGISTRY:
        raise KeyError("Unknown pretrain dataset preset %r." % name)
    return PRETRAIN_DATASET_REGISTRY[normalized]
