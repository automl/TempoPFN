from typing import Any

import numpy as np
from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.audio_generators.network_topology_generator import (
    NetworkTopologyAudioGenerator,
)
from src.synthetic_generation.generator_params import NetworkTopologyAudioParams


class NetworkTopologyAudioWrapper(GeneratorWrapper):
    def __init__(self, params: NetworkTopologyAudioParams):
        super().__init__(params)
        self.params: NetworkTopologyAudioParams = params

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                "server_duration": self.params.server_duration,
                "sample_rate": self.params.sample_rate,
                "normalize_output": self.params.normalize_output,
                "traffic_lfo_freq_range": self.params.traffic_lfo_freq_range,
                "traffic_lfo_mul_range": self.params.traffic_lfo_mul_range,
                "burst_rate_hz_range": self.params.burst_rate_hz_range,
                "burst_duration_range": self.params.burst_duration_range,
                "burst_mul_range": self.params.burst_mul_range,
                "congestion_period_range": self.params.congestion_period_range,
                "congestion_depth_range": self.params.congestion_depth_range,
                "congestion_release_time_range": self.params.congestion_release_time_range,
                "overhead_lfo_freq_range": self.params.overhead_lfo_freq_range,
                "overhead_mul_range": self.params.overhead_mul_range,
                "attack_period_range": self.params.attack_period_range,
                "attack_env_points": self.params.attack_env_points,
                "attack_mul_range": self.params.attack_mul_range,
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

        generator = NetworkTopologyAudioGenerator(
            length=params["length"],
            server_duration=params["server_duration"],
            sample_rate=params["sample_rate"],
            normalize_output=params["normalize_output"],
            traffic_lfo_freq_range=params["traffic_lfo_freq_range"],
            traffic_lfo_mul_range=params["traffic_lfo_mul_range"],
            burst_rate_hz_range=params["burst_rate_hz_range"],
            burst_duration_range=params["burst_duration_range"],
            burst_mul_range=params["burst_mul_range"],
            congestion_period_range=params["congestion_period_range"],
            congestion_depth_range=params["congestion_depth_range"],
            congestion_release_time_range=params["congestion_release_time_range"],
            overhead_lfo_freq_range=params["overhead_lfo_freq_range"],
            overhead_mul_range=params["overhead_mul_range"],
            attack_period_range=params["attack_period_range"],
            attack_env_points=params["attack_env_points"],
            attack_mul_range=params["attack_mul_range"],
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
