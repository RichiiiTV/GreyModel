from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

from .preprocessing import preprocess_sample, stack_prepared_images
from .types import Sample, StationConfig


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


def group_samples_by_station(samples: Iterable[Sample]) -> Dict[int, List[Sample]]:
    grouped: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.station_id].append(sample)
    return dict(grouped)


def collate_batch(items, as_torch: bool = True) -> Mapping[str, object]:
    prepared = [item["prepared"] for item in items]
    samples = [item["sample"] for item in items]
    model_input = stack_prepared_images(prepared, as_torch=as_torch)
    return {
        "model_input": model_input,
        "accept_reject": [sample.accept_reject for sample in samples],
        "defect_tags": [sample.defect_tags for sample in samples],
        "samples": samples,
    }
