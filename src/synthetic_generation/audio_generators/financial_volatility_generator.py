import numpy as np
from pyo import LFO, BrownNoise, Follower, Metro, Mix, Sine, TrigExpseg
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.audio_generators.utils import (
    normalize_waveform,
    run_offline_pyo,
)


class FinancialVolatilityAudioGenerator(AbstractTimeSeriesGenerator):
    """
    Generate synthetic univariate time series that mimics financial market
    behavior with volatility clustering and occasional jumps.
    """

    def __init__(
        self,
        length: int,
        server_duration: float,
        sample_rate: int,
        normalize_output: bool,
        # Trend LFO
        trend_lfo_freq_range: tuple[float, float],
        trend_lfo_mul_range: tuple[float, float],
        # Volatility clustering
        volatility_carrier_freq_range: tuple[float, float],
        follower_freq_range: tuple[float, float],
        volatility_range: tuple[float, float],
        # Jumps
        jump_metro_time_range: tuple[float, float],
        jump_env_start_range: tuple[float, float],
        jump_env_decay_time_range: tuple[float, float],
        jump_freq_range: tuple[float, float],
        jump_direction_up_probability: float,
        random_seed: int | None = None,
    ):
        self.length = length
        self.server_duration = server_duration
        self.sample_rate = sample_rate
        self.normalize_output = normalize_output

        self.trend_lfo_freq_range = trend_lfo_freq_range
        self.trend_lfo_mul_range = trend_lfo_mul_range
        self.volatility_carrier_freq_range = volatility_carrier_freq_range
        self.follower_freq_range = follower_freq_range
        self.volatility_range = volatility_range
        self.jump_metro_time_range = jump_metro_time_range
        self.jump_env_start_range = jump_env_start_range
        self.jump_env_decay_time_range = jump_env_decay_time_range
        self.jump_freq_range = jump_freq_range
        self.jump_direction_up_probability = jump_direction_up_probability

        self.rng = np.random.default_rng(random_seed)

    def _build_synth(self):
        # Trend
        trend_freq = self.rng.uniform(*self.trend_lfo_freq_range)
        trend_mul = self.rng.uniform(*self.trend_lfo_mul_range)
        trend = LFO(freq=trend_freq, type=0, mul=trend_mul)

        # Volatility clustering
        carrier_freq = self.rng.uniform(*self.volatility_carrier_freq_range)
        follower_freq = self.rng.uniform(*self.follower_freq_range)
        volatility_min, volatility_max = self.volatility_range
        volatility_osc = Sine(freq=carrier_freq)
        volatility = Follower(volatility_osc, freq=follower_freq).range(volatility_min, volatility_max)
        market_noise = BrownNoise(mul=volatility)

        # Jumps
        jump_time = self.rng.uniform(*self.jump_metro_time_range)
        jump_env_start = self.rng.uniform(*self.jump_env_start_range)
        jump_env_decay = self.rng.uniform(*self.jump_env_decay_time_range)
        jump_freq = self.rng.uniform(*self.jump_freq_range)
        direction = 1.0 if self.rng.random() < self.jump_direction_up_probability else -1.0

        jump_trigger = Metro(time=jump_time).play()
        jump_env = TrigExpseg(jump_trigger, list=[(0.0, jump_env_start), (jump_env_decay, 0.0)])
        jumps = Sine(freq=jump_freq, mul=jump_env * direction)

        return Mix([trend, market_noise, jumps], voices=1)

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
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
