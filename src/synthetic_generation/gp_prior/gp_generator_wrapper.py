from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import GPGeneratorParams
from src.synthetic_generation.gp_prior.gp_generator import GPGenerator


class GPGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: GPGeneratorParams):
        super().__init__(params)
        self.params: GPGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        params = super()._sample_parameters(batch_size)

        params.update(
            {
                "length": self.params.length,
                "max_kernels": self.params.max_kernels,
                "likelihood_noise_level": self.params.likelihood_noise_level,
                "noise_level": self.params.noise_level,
                "use_original_gp": self.params.use_original_gp,
                "gaussians_periodic": self.params.gaussians_periodic,
                "peak_spike_ratio": self.params.peak_spike_ratio,
                "subfreq_ratio": self.params.subfreq_ratio,
                "periods_per_freq": self.params.periods_per_freq,
                "gaussian_sampling_ratio": self.params.gaussian_sampling_ratio,
                "kernel_periods": self.params.kernel_periods,
                "max_period_ratio": self.params.max_period_ratio,
                "kernel_bank": self.params.kernel_bank,
            }
        )
        return params

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TimeSeriesContainer:
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters(batch_size)

        generator = GPGenerator(
            params=self.params,
            length=params["length"],
            random_seed=seed,
        )

        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i
            values = generator.generate_time_series(random_seed=batch_seed)

            batch_values.append(values)

        return TimeSeriesContainer(
            values=np.array(batch_values),
            start=params["start"],
            frequency=params["frequency"],
        )
