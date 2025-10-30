from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.cauker.cauker_generator import CauKerGenerator
from src.synthetic_generation.generator_params import CauKerGeneratorParams


class CauKerGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for CauKerGenerator that handles batch generation and formatting.
    """

    def __init__(self, params: CauKerGeneratorParams):
        super().__init__(params)
        self.params: CauKerGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        params = super()._sample_parameters(batch_size)
        # Resolve num_channels if range is given: sample once per batch for consistency
        desired_channels = self.params.num_channels
        if isinstance(desired_channels, tuple) and len(desired_channels) == 2:
            low, high = desired_channels
            if low > high:
                low, high = high, low
            num_channels = int(self.rng.integers(low, high + 1))
        elif isinstance(desired_channels, list):
            num_channels = int(self.rng.choice(desired_channels))
        else:
            num_channels = int(desired_channels)

        params.update(
            {
                "length": self.params.length,
                "num_channels": num_channels,
                "max_parents": self.params.max_parents,
                "num_nodes": self.params.num_nodes,
            }
        )
        return params

    def generate_batch(
        self, batch_size: int, seed: Optional[int] = None
    ) -> TimeSeriesContainer:
        # Establish a base seed to ensure different series use different seeds
        base_seed = seed if seed is not None else self.params.global_seed
        self._set_random_seeds(base_seed)

        sampled = self._sample_parameters(batch_size)

        batch_params = CauKerGeneratorParams(
            global_seed=self.params.global_seed,
            length=sampled["length"],
            frequency=None,
            start=None,
            num_channels=sampled["num_channels"],
            max_parents=sampled["max_parents"],
            num_nodes=sampled["num_nodes"],
        )
        generator = CauKerGenerator(batch_params)

        values = []
        for i in range(batch_size):
            series_seed = base_seed + i
            series = generator.generate_time_series(series_seed)
            values.append(series)

        return TimeSeriesContainer(
            values=np.array(values, dtype=np.float32),
            start=sampled["start"],
            frequency=sampled["frequency"],
        )
