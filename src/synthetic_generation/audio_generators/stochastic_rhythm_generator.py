from typing import Optional

import numpy as np
from pyo import Metro, Mix, Sine, TrigExpseg

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.audio_generators.utils import (
    normalize_waveform,
    run_offline_pyo,
)


class StochasticRhythmAudioGenerator(AbstractTimeSeriesGenerator):
    """
    Generate rhythmic patterns with layered triggers, per-layer envelopes
    and tones. Parameters are sampled per series for diversity.
    """

    def __init__(
        self,
        length: int,
        server_duration: float,
        sample_rate: int,
        normalize_output: bool,
        base_tempo_hz_range: tuple[float, float],
        num_layers_range: tuple[int, int],
        subdivisions: tuple[int, ...],
        attack_range: tuple[float, float],
        decay_range: tuple[float, float],
        tone_freq_range: tuple[float, float],
        tone_mul_range: tuple[float, float],
        random_seed: Optional[int] = None,
    ):
        self.length = length
        self.server_duration = server_duration
        self.sample_rate = sample_rate
        self.normalize_output = normalize_output

        self.base_tempo_hz_range = base_tempo_hz_range
        self.num_layers_range = num_layers_range
        self.subdivisions = subdivisions
        self.attack_range = attack_range
        self.decay_range = decay_range
        self.tone_freq_range = tone_freq_range
        self.tone_mul_range = tone_mul_range

        self.rng = np.random.default_rng(random_seed)

    def _build_synth(self):
        base_tempo = self.rng.uniform(*self.base_tempo_hz_range)
        num_layers = int(
            self.rng.integers(self.num_layers_range[0], self.num_layers_range[1] + 1)
        )

        layers = []
        for _ in range(num_layers):
            subdivision = self.subdivisions[
                int(self.rng.integers(0, len(self.subdivisions)))
            ]
            rhythm_freq = base_tempo * subdivision
            trigger = Metro(time=1.0 / rhythm_freq).play()

            attack = self.rng.uniform(*self.attack_range)
            decay = self.rng.uniform(*self.decay_range)
            env = TrigExpseg(trigger, list=[(0.0, 1.0), (attack, 0.8), (decay, 0.0)])

            tone_freq = self.rng.uniform(*self.tone_freq_range)
            tone_mul = self.rng.uniform(*self.tone_mul_range)
            tone = Sine(freq=tone_freq, mul=env * tone_mul)
            layers.append(tone)

        return Mix(layers, voices=1)

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
