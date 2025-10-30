from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.audio_generators.financial_volatility_generator import (
    FinancialVolatilityAudioGenerator,
)
from src.synthetic_generation.generator_params import FinancialVolatilityAudioParams


class FinancialVolatilityAudioWrapper(GeneratorWrapper):
    def __init__(self, params: FinancialVolatilityAudioParams):
        super().__init__(params)
        self.params: FinancialVolatilityAudioParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                "server_duration": self.params.server_duration,
                "sample_rate": self.params.sample_rate,
                "normalize_output": self.params.normalize_output,
                # Trend LFO
                "trend_lfo_freq_range": self.params.trend_lfo_freq_range,
                "trend_lfo_mul_range": self.params.trend_lfo_mul_range,
                # Volatility clustering
                "volatility_carrier_freq_range": self.params.volatility_carrier_freq_range,
                "follower_freq_range": self.params.follower_freq_range,
                "volatility_range": self.params.volatility_range,
                # Jumps
                "jump_metro_time_range": self.params.jump_metro_time_range,
                "jump_env_start_range": self.params.jump_env_start_range,
                "jump_env_decay_time_range": self.params.jump_env_decay_time_range,
                "jump_freq_range": self.params.jump_freq_range,
                "jump_direction_up_probability": self.params.jump_direction_up_probability,
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

        generator = FinancialVolatilityAudioGenerator(
            length=params["length"],
            server_duration=params["server_duration"],
            sample_rate=params["sample_rate"],
            normalize_output=params["normalize_output"],
            trend_lfo_freq_range=params["trend_lfo_freq_range"],
            trend_lfo_mul_range=params["trend_lfo_mul_range"],
            volatility_carrier_freq_range=params["volatility_carrier_freq_range"],
            follower_freq_range=params["follower_freq_range"],
            volatility_range=params["volatility_range"],
            jump_metro_time_range=params["jump_metro_time_range"],
            jump_env_start_range=params["jump_env_start_range"],
            jump_env_decay_time_range=params["jump_env_decay_time_range"],
            jump_freq_range=params["jump_freq_range"],
            jump_direction_up_probability=params["jump_direction_up_probability"],
            random_seed=seed,
        )

        def _derive_series_seed(base_seed: int, index: int) -> int:
            # Mix base seed with index and class hash to decorrelate adjacent seeds
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
