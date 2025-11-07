from typing import Any

import numpy as np
from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.audio_generators.stochastic_rhythm_generator import (
    StochasticRhythmAudioGenerator,
)
from src.synthetic_generation.generator_params import StochasticRhythmAudioParams


class StochasticRhythmAudioWrapper(GeneratorWrapper):
    def __init__(self, params: StochasticRhythmAudioParams):
        super().__init__(params)
        self.params: StochasticRhythmAudioParams = params

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                "server_duration": self.params.server_duration,
                "sample_rate": self.params.sample_rate,
                "normalize_output": self.params.normalize_output,
                "base_tempo_hz_range": self.params.base_tempo_hz_range,
                "num_layers_range": self.params.num_layers_range,
                "subdivisions": self.params.subdivisions,
                "attack_range": self.params.attack_range,
                "decay_range": self.params.decay_range,
                "tone_freq_range": self.params.tone_freq_range,
                "tone_mul_range": self.params.tone_mul_range,
            }
        )
        return params

    def generate_batch(
        self,
        batch_size: int,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> TimeSeriesContainer:
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters(batch_size)

        generator = StochasticRhythmAudioGenerator(
            length=params["length"],
            server_duration=params["server_duration"],
            sample_rate=params["sample_rate"],
            normalize_output=params["normalize_output"],
            base_tempo_hz_range=params["base_tempo_hz_range"],
            num_layers_range=params["num_layers_range"],
            subdivisions=params["subdivisions"],
            attack_range=params["attack_range"],
            decay_range=params["decay_range"],
            tone_freq_range=params["tone_freq_range"],
            tone_mul_range=params["tone_mul_range"],
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
