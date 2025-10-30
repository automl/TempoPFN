from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.audio_generators.multi_scale_fractal_generator import (
    MultiScaleFractalAudioGenerator,
)
from src.synthetic_generation.generator_params import MultiScaleFractalAudioParams


class MultiScaleFractalAudioWrapper(GeneratorWrapper):
    def __init__(self, params: MultiScaleFractalAudioParams):
        super().__init__(params)
        self.params: MultiScaleFractalAudioParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                "server_duration": self.params.server_duration,
                "sample_rate": self.params.sample_rate,
                "normalize_output": self.params.normalize_output,
                "base_noise_mul_range": self.params.base_noise_mul_range,
                "num_scales_range": self.params.num_scales_range,
                "scale_freq_base_range": self.params.scale_freq_base_range,
                "q_factor_range": self.params.q_factor_range,
                "per_scale_attenuation_range": self.params.per_scale_attenuation_range,
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

        generator = MultiScaleFractalAudioGenerator(
            length=params["length"],
            server_duration=params["server_duration"],
            sample_rate=params["sample_rate"],
            normalize_output=params["normalize_output"],
            base_noise_mul_range=params["base_noise_mul_range"],
            num_scales_range=params["num_scales_range"],
            scale_freq_base_range=params["scale_freq_base_range"],
            q_factor_range=params["q_factor_range"],
            per_scale_attenuation_range=params["per_scale_attenuation_range"],
            random_seed=seed,
        )

        def _derive_series_seed(base_seed: int, index: int) -> int:
            mixed = (
                (base_seed & 0x7FFFFFFF)
                ^ ((index * 0x9E3779B1) & 0x7FFFFFFF)
                ^ (hash(self.__class__.__name__) & 0x7FFFFFFF)
            )
            return int(mixed)

        batch_values = []
        for i in range(batch_size):
            series_seed = None if seed is None else _derive_series_seed(seed, i)
            values = generator.generate_time_series(random_seed=series_seed)
            batch_values.append(values)

        return TimeSeriesContainer(
            values=np.array(batch_values),
            start=params["start"],
            frequency=params["frequency"],
        )
