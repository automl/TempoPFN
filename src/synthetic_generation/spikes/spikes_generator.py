from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.generator_params import SpikesGeneratorParams, SpikeShape


class SpikesGenerator(AbstractTimeSeriesGenerator):
    """Generates spike-based time series with V-shaped or chopped spikes."""

    def __init__(self, params: SpikesGeneratorParams):
        self.params = params
        np.random.seed(params.global_seed)

    def generate_time_series(self, random_seed: Optional[int] = None) -> np.ndarray:
        """Generate a time series with baseline and random spikes."""
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize signal
        signal = np.full(
            self.params.length,
            self._sample_scalar(self.params.baseline),
            dtype=np.float64,
        )
        series_params = self._sample_series_parameters()

        if series_params["spike_count"] > 0:
            positions = self._generate_spike_positions(
                series_params["spike_count"], series_params["burst_mode"]
            )
            for pos in positions:
                spike = self._generate_single_spike(
                    series_params["amplitude"],
                    series_params["angle_deg"],
                    series_params["spike_shapes"],
                    series_params["spikes_above_baseline"],
                )
                self._inject_spike(signal, spike, pos)

        if series_params["add_noise"] and self.params.noise_std > 0:
            signal += self._generate_colored_noise()

        return signal

    def _sample_series_parameters(self) -> Dict:
        """Sample consistent parameters for the entire series."""
        series_type = np.random.choice(
            list(self.params.series_type_probabilities.keys()),
            p=list(self.params.series_type_probabilities.values()),
        )

        spike_shapes = []
        if series_type == "v_only":
            shape = (
                SpikeShape.V_SHAPE
                if np.random.random() < 0.5
                else SpikeShape.INVERTED_V
            )
            spike_shapes = [shape]  # Only one shape type for consistency
        elif series_type == "chopped_only":
            shape = (
                SpikeShape.CHOPPED_V
                if np.random.random() < 0.5
                else SpikeShape.CHOPPED_INVERTED_V
            )
            spike_shapes = [shape]
        else:  # mixed
            above = np.random.random() < self.params.spikes_above_baseline_probability
            shape1 = SpikeShape.V_SHAPE if above else SpikeShape.INVERTED_V
            shape2 = SpikeShape.CHOPPED_V if above else SpikeShape.CHOPPED_INVERTED_V
            spike_shapes = [shape1, shape2]

        # Determine mode and corresponding spike count
        burst_mode = np.random.random() < self.params.burst_mode_probability
        if burst_mode:
            spike_count_range = self.params.spike_count_burst
        else:
            spike_count_range = self.params.spike_count_uniform

        return {
            "add_noise": np.random.random() < self.params.noise_probability,
            "burst_mode": burst_mode,
            "spike_count": self._sample_scalar(spike_count_range, is_int=True),
            "amplitude": self._sample_scalar(self.params.spike_amplitude),
            "angle_deg": np.random.uniform(*self.params.spike_angle_range),
            "spikes_above_baseline": np.random.random()
            < self.params.spikes_above_baseline_probability,
            "spike_shapes": spike_shapes,
        }

    def _sample_scalar(
        self, value: Union[float, int, Tuple], is_int: bool = False
    ) -> float:
        """Sample a scalar from a value or range."""
        if isinstance(value, tuple):
            return np.random.randint(*value) if is_int else np.random.uniform(*value)
        return float(value)

    def _generate_colored_noise(self) -> np.ndarray:
        """Generate colored noise with brown/pink characteristics."""
        white_noise = np.random.normal(0, 1, self.params.length)
        fft_noise = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(self.params.length)
        freqs[0] = freqs[1]  # Avoid DC division by zero

        filter_response = 1.0 / (np.abs(freqs) ** (self.params.brown_noise_alpha / 2.0))
        filter_response[np.abs(freqs) > self.params.noise_cutoff_freq] *= np.exp(
            -(
                (
                    (
                        np.abs(freqs)[np.abs(freqs) > self.params.noise_cutoff_freq]
                        - self.params.noise_cutoff_freq
                    )
                    / self.params.noise_cutoff_freq
                )
                ** 2
            )
        )

        colored_noise = np.real(np.fft.ifft(fft_noise * filter_response))
        return colored_noise / np.std(colored_noise) * self.params.noise_std

    def _generate_spike_positions(
        self, spike_count: int, burst_mode: bool
    ) -> List[int]:
        """Generate spike positions with minimum separation."""
        if spike_count == 0:
            return []

        # Adjust spike count based on available space
        min_separation = self.params.max_spike_width + self.params.min_spike_margin
        margin = self.params.max_spike_width
        usable_length = self.params.length - 2 * margin
        max_spikes = max(1, usable_length // min_separation)
        spike_count = min(spike_count, max_spikes)

        if burst_mode:
            burst_width = max(
                spike_count * min_separation,
                int(
                    np.random.uniform(*self.params.burst_width_fraction)
                    * self.params.length
                ),
            )
            burst_width = min(burst_width, usable_length)
            burst_start = np.random.randint(
                margin, self.params.length - burst_width - margin + 1
            )
            positions = self._distribute_positions_burst_mode(
                spike_count, burst_start, burst_start + burst_width
            )
        else:
            positions = self._distribute_positions_spread_mode(
                spike_count, margin, self.params.length - margin
            )

        return self._enforce_separation(positions, min_separation, margin)

    def _distribute_positions_spread_mode(
        self, count: int, start: int, end: int
    ) -> List[int]:
        """
        Distribute positions with perfectly consistent spacing between spikes
        while using smaller edge margins.

        Strategy:
        - Let S be the inter-spike spacing and M be the edge margin.
        - We enforce M = r * S, where r = edge_margin_ratio in params.
        - Total usable span T = end - start must satisfy: T = (count - 1) * S + 2 * M
          => S = T / (count - 1 + 2r)
        - Positions: p_i = start + M + i * S for i in [0, count-1]

        If the spacing S would violate the minimum separation implied by spike
        geometry, we fall back to a safe placement.
        """
        if count <= 1:
            return [(start + end) // 2]

        # Minimum separation required so spikes cannot overlap once injected
        min_separation = self.params.max_spike_width + self.params.min_spike_margin

        total_space = end - start
        ratio = max(0.0, float(getattr(self.params, "edge_margin_ratio", 0.0)))

        # Compute ideal spacing S given desired margin ratio
        denominator = (count - 1) + 2.0 * ratio
        if denominator <= 0:
            return self._distribute_positions_fallback(
                count, start, end, min_separation
            )

        base_spacing = total_space / denominator

        # Ensure spacing respects the minimum separation
        if base_spacing < min_separation:
            return self._distribute_positions_fallback(
                count, start, end, min_separation
            )

        edge_margin = ratio * base_spacing

        positions = [
            int(round(start + edge_margin + i * base_spacing)) for i in range(count)
        ]

        # Clamp within bounds and ensure strictly increasing order
        positions = sorted(max(start, min(end, p)) for p in positions)

        return positions

    def _distribute_positions_burst_mode(
        self, count: int, start: int, end: int
    ) -> List[int]:
        """Original burst mode distribution logic."""
        if count <= 1:
            return [(start + end) // 2]

        # For burst mode, use the original logic with some randomness
        min_separation = self.params.max_spike_width + self.params.min_spike_margin
        available_space = end - start

        if available_space < (count - 1) * min_separation:
            return self._distribute_positions_fallback(
                count, start, end, min_separation
            )

        # Distribute positions with some randomness for burst mode
        positions = []
        if count == 1:
            positions.append((start + end) // 2)
        else:
            interval = (end - start) / (count - 1)
            for i in range(count):
                base_pos = start + i * interval
                # Add some jitter for burst mode
                jitter_range = min(interval * 0.2, min_separation * 0.3)
                jitter = np.random.uniform(-jitter_range, jitter_range)
                pos = int(base_pos + jitter)
                pos = max(start, min(end, pos))
                positions.append(pos)

        return positions

    def _distribute_positions_fallback(
        self, count: int, start: int, end: int, min_separation: int
    ) -> List[int]:
        """Fallback method when there's not enough space for optimal distribution."""
        positions = []
        current_pos = start

        for i in range(count):
            if current_pos <= end:
                positions.append(current_pos)
                current_pos += min_separation
            else:
                break

        return positions

    def _enforce_separation(
        self, positions: List[int], min_separation: int, margin: int
    ) -> List[int]:
        """Ensure minimum separation between spike positions."""
        if len(positions) <= 1:
            return positions

        positions = sorted(positions)
        adjusted = [max(margin, positions[0])]

        for pos in positions[1:]:
            next_pos = max(pos, adjusted[-1] + min_separation)
            if next_pos <= self.params.length - margin:
                adjusted.append(next_pos)
            else:
                # If we can't fit this spike, stop adding more
                break

        return adjusted

    def _generate_single_spike(
        self,
        amplitude: float,
        angle_deg: float,
        spike_shapes: List[SpikeShape],
        spikes_above_baseline: bool,
    ) -> np.ndarray:
        """Generate a single spike with specified shape and angle."""
        shape = np.random.choice(spike_shapes)
        slope = np.tan(np.radians(angle_deg))
        rise_time = np.clip(
            int(np.round(amplitude / slope)),
            self.params.min_spike_width // 2,
            self.params.max_spike_width // 2,
        )
        fall_time = rise_time
        plateau_duration = (
            np.random.randint(*self.params.plateau_duration)
            if shape in (SpikeShape.CHOPPED_V, SpikeShape.CHOPPED_INVERTED_V)
            else 0
        )
        final_amplitude = amplitude if spikes_above_baseline else -amplitude

        spike_length = rise_time + plateau_duration + fall_time
        spike = np.zeros(spike_length)

        # Rise phase
        spike[:rise_time] = np.linspace(0, final_amplitude, rise_time, endpoint=False)

        # Plateau phase
        if plateau_duration:
            spike[rise_time : rise_time + plateau_duration] = final_amplitude

        # Fall phase
        spike[rise_time + plateau_duration :] = np.linspace(
            final_amplitude, 0, fall_time, endpoint=False
        )

        return spike

    def _inject_spike(
        self, signal: np.ndarray, spike: np.ndarray, position: int
    ) -> None:
        """Inject a spike into the signal at the given position."""
        half_length = len(spike) // 2
        start = max(0, position - half_length)
        end = min(len(signal), position + half_length + len(spike) % 2)
        spike_start = max(0, half_length - position)
        spike_end = spike_start + (end - start)

        if end > start and spike_end > spike_start:
            signal[start:end] += spike[spike_start:spike_end]
