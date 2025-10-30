from typing import Optional

import numpy as np
from pyo import Biquad, BrownNoise, Mix

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.audio_generators.utils import (
    normalize_waveform,
    run_offline_pyo,
)


class MultiScaleFractalAudioGenerator(AbstractTimeSeriesGenerator):
    """
    Generate multi-scale fractal-like patterns by filtering noise at
    multiple frequency bands with varying Q and attenuation per scale.
    """

    def __init__(
        self,
        length: int,
        server_duration: float,
        sample_rate: int,
        normalize_output: bool,
        base_noise_mul_range: tuple[float, float],
        num_scales_range: tuple[int, int],
        scale_freq_base_range: tuple[float, float],
        q_factor_range: tuple[float, float],
        per_scale_attenuation_range: tuple[float, float],
        random_seed: Optional[int] = None,
    ):
        self.length = length
        self.server_duration = server_duration
        self.sample_rate = sample_rate
        self.normalize_output = normalize_output

        self.base_noise_mul_range = base_noise_mul_range
        self.num_scales_range = num_scales_range
        self.scale_freq_base_range = scale_freq_base_range
        self.q_factor_range = q_factor_range
        self.per_scale_attenuation_range = per_scale_attenuation_range

        self.rng = np.random.default_rng(random_seed)

    def _build_synth(self):
        base_mul = self.rng.uniform(*self.base_noise_mul_range)
        base = BrownNoise(mul=base_mul)

        num_scales = int(
            self.rng.integers(self.num_scales_range[0], self.num_scales_range[1] + 1)
        )

        scales = []
        for i in range(num_scales):
            scale_freq = self.rng.uniform(*self.scale_freq_base_range) * (0.5**i)
            q_factor = self.rng.uniform(*self.q_factor_range)
            per_scale_att = self.rng.uniform(*self.per_scale_attenuation_range)
            filtered = Biquad(base, freq=scale_freq, q=q_factor, type=0)
            scales.append(filtered * (per_scale_att**i))

        return Mix(scales, voices=1)

    def generate_time_series(self, random_seed: Optional[int] = None) -> np.ndarray:
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        waveform = run_offline_pyo(
            synth_builder=self._build_synth,
            server_duration=self.server_duration,
            sample_rate=self.sample_rate,
            length=self.length,
        )
        if self.normalize_output:
            waveform = normalize_waveform(waveform)
        return waveform
