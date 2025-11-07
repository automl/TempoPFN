import numpy as np
from pyo import LFO, BrownNoise, Metro, Mix, Noise, TrigExpseg
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.audio_generators.utils import (
    normalize_waveform,
    run_offline_pyo,
)


class NetworkTopologyAudioGenerator(AbstractTimeSeriesGenerator):
    """
    Simulate network traffic with base flow, packet bursts, periodic congestion,
    protocol overhead, and DDoS-like attacks. Parameters are sampled per series.
    """

    def __init__(
        self,
        length: int,
        server_duration: float,
        sample_rate: int,
        normalize_output: bool,
        traffic_lfo_freq_range: tuple[float, float],
        traffic_lfo_mul_range: tuple[float, float],
        burst_rate_hz_range: tuple[float, float],
        burst_duration_range: tuple[float, float],
        burst_mul_range: tuple[float, float],
        congestion_period_range: tuple[float, float],
        congestion_depth_range: tuple[float, float],
        congestion_release_time_range: tuple[float, float],
        overhead_lfo_freq_range: tuple[float, float],
        overhead_mul_range: tuple[float, float],
        attack_period_range: tuple[float, float],
        attack_env_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        attack_mul_range: tuple[float, float],
        random_seed: int | None = None,
    ):
        self.length = length
        self.server_duration = server_duration
        self.sample_rate = sample_rate
        self.normalize_output = normalize_output

        self.traffic_lfo_freq_range = traffic_lfo_freq_range
        self.traffic_lfo_mul_range = traffic_lfo_mul_range
        self.burst_rate_hz_range = burst_rate_hz_range
        self.burst_duration_range = burst_duration_range
        self.burst_mul_range = burst_mul_range
        self.congestion_period_range = congestion_period_range
        self.congestion_depth_range = congestion_depth_range
        self.congestion_release_time_range = congestion_release_time_range
        self.overhead_lfo_freq_range = overhead_lfo_freq_range
        self.overhead_mul_range = overhead_mul_range
        self.attack_period_range = attack_period_range
        self.attack_env_points = attack_env_points
        self.attack_mul_range = attack_mul_range

        self.rng = np.random.default_rng(random_seed)

    def _build_synth(self):
        # Base traffic flow
        traffic_freq = self.rng.uniform(*self.traffic_lfo_freq_range)
        traffic_mul = self.rng.uniform(*self.traffic_lfo_mul_range)
        traffic_base = LFO(freq=traffic_freq, type=0, mul=traffic_mul)

        # Packet bursts
        burst_rate = self.rng.uniform(*self.burst_rate_hz_range)
        burst_trigger = Metro(time=1.0 / burst_rate).play()
        burst_duration = self.rng.uniform(*self.burst_duration_range)
        burst_env = TrigExpseg(burst_trigger, list=[(0.0, 0.8), (burst_duration, 0.0)])
        burst_mul = self.rng.uniform(*self.burst_mul_range)
        bursts = Noise(mul=burst_env * burst_mul)

        # Periodic congestion (negative amplitude dip)
        congestion_period = self.rng.uniform(*self.congestion_period_range)
        congestion_trigger = Metro(time=congestion_period).play()
        congestion_depth = self.rng.uniform(*self.congestion_depth_range)  # negative
        congestion_release = self.rng.uniform(*self.congestion_release_time_range)
        congestion_env = TrigExpseg(
            congestion_trigger,
            list=[(0.0, congestion_depth), (congestion_release, 0.0)],
        )

        # Protocol overhead
        overhead_freq = self.rng.uniform(*self.overhead_lfo_freq_range)
        overhead_mul = self.rng.uniform(*self.overhead_mul_range)
        overhead = LFO(freq=overhead_freq, type=1, mul=overhead_mul)

        # DDoS-like attacks
        attack_period = self.rng.uniform(*self.attack_period_range)
        attack_trigger = Metro(time=attack_period).play()
        attack_env = TrigExpseg(attack_trigger, list=list(self.attack_env_points))
        attack_mul = self.rng.uniform(*self.attack_mul_range)
        attacks = BrownNoise(mul=attack_env * attack_mul)

        return Mix([traffic_base, bursts, congestion_env, overhead, attacks], voices=1)

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
