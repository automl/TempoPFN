from typing import Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import (
    OrnsteinUhlenbeckProcessGeneratorParams,
)
from src.synthetic_generation.ornstein_uhlenbeck_process.ou_generator import (
    OrnsteinUhlenbeckProcessGenerator,
)


class OrnsteinUhlenbeckProcessGeneratorWrapper(GeneratorWrapper):
    """Wrapper for the regime-switching OU generator."""

    def __init__(self, params: OrnsteinUhlenbeckProcessGeneratorParams):
        super().__init__(params)
        self.generator = OrnsteinUhlenbeckProcessGenerator(params)

    def generate_batch(
        self, batch_size: int, seed: Optional[int] = None
    ) -> TimeSeriesContainer:
        if seed is not None:
            self._set_random_seeds(seed)

        sampled_params = self._sample_parameters(batch_size)

        values = []
        for i in range(batch_size):
            series_seed = (seed + i) if seed is not None else None
            series = self.generator.generate_time_series(series_seed)
            values.append(series)

        return TimeSeriesContainer(
            values=np.array(values),
            start=sampled_params["start"],
            frequency=sampled_params["frequency"],
        )
