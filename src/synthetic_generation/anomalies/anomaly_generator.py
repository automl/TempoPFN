import numpy as np
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.generator_params import (
    AnomalyGeneratorParams,
    AnomalyType,
    MagnitudePattern,
)


class AnomalyGenerator(AbstractTimeSeriesGenerator):
    """
    Generator for synthetic time series with realistic spike anomalies.

    Creates clean constant baseline signals with periodic spike patterns that
    resemble real-world time series behavior, including clustering and magnitude patterns.
    """

    def __init__(self, params: AnomalyGeneratorParams):
        """
        Initialize the AnomalyGenerator.

        Parameters
        ----------
        params : AnomalyGeneratorParams
            Configuration parameters for anomaly generation.
        """
        self.params = params

    def _determine_spike_direction(self) -> AnomalyType:
        """
        Determine if this series will have only up or only down spikes.

        Returns
        -------
        AnomalyType
            Either SPIKE_UP or SPIKE_DOWN for the entire series.
        """
        if np.random.random() < self.params.spike_direction_probability:
            return AnomalyType.SPIKE_UP
        else:
            return AnomalyType.SPIKE_DOWN

    def _generate_spike_positions(self) -> list[list[int]]:
        """
        Generate spike positions:
        - Always create uniformly spaced single spikes (base schedule)
        - With 25% probability: add clusters (1-3 extra spikes) near a fraction of base spikes
        - With 25% probability: add single random spikes across the series

        Returns
        -------
        List[List[int]]
            List of spike events, where each event is a list of positions
            (single spike = [pos], cluster = [pos, pos+offset, ...]).
        """
        # Base uniform schedule (no jitter/variance)
        base_period = np.random.randint(*self.params.base_period_range)
        start_position = base_period // 2
        base_positions = list(range(start_position, self.params.length, base_period))

        # Start with single-spike events at base positions
        spike_events: list[list[int]] = [[pos] for pos in base_positions]

        if not base_positions:
            return spike_events

        # Decide series type
        series_draw = np.random.random()

        # 25%: augment with clusters near some base spikes
        if series_draw < self.params.cluster_series_probability:
            num_base_events = len(base_positions)
            num_to_augment = max(1, int(round(self.params.cluster_event_fraction * num_base_events)))
            num_to_augment = min(num_to_augment, num_base_events)

            chosen_indices = (
                np.random.choice(num_base_events, size=num_to_augment, replace=False)
                if num_to_augment > 0
                else np.array([], dtype=int)
            )

            for idx in chosen_indices:
                base_pos = base_positions[int(idx)]
                # Number of additional spikes (1..3) per selected event
                num_additional = np.random.randint(*self.params.cluster_additional_spikes_range)
                if num_additional <= 0:
                    continue

                # Draw offsets around base spike and exclude zero to avoid duplicates
                offsets = np.random.randint(
                    self.params.cluster_offset_range[0],
                    self.params.cluster_offset_range[1],
                    size=num_additional,
                )
                offsets = [int(off) for off in offsets if off != 0]

                cluster_positions: set[int] = {base_pos}
                for off in offsets:
                    pos = base_pos + off
                    if 0 <= pos < self.params.length:
                        cluster_positions.add(pos)

                spike_events[int(idx)] = sorted(cluster_positions)

        # Next 25%: add random single spikes across the series
        elif series_draw < (self.params.cluster_series_probability + self.params.random_series_probability):
            num_base_events = len(base_positions)
            num_random = int(round(self.params.random_spike_fraction_of_base * num_base_events))
            if num_random > 0:
                all_indices = np.arange(self.params.length)
                base_array = np.array(base_positions, dtype=int)
                candidates = np.setdiff1d(all_indices, base_array, assume_unique=False)
                if candidates.size > 0:
                    choose_n = min(num_random, candidates.size)
                    rand_positions = np.random.choice(candidates, size=choose_n, replace=False)
                    for pos in rand_positions:
                        spike_events.append([int(pos)])

        # Else: 50% clean series (uniform singles only)

        return spike_events

    def _generate_spike_magnitudes(self, total_spikes: int) -> np.ndarray:
        """
        Generate spike magnitudes based on the configured pattern.

        Parameters
        ----------
        total_spikes : int
            Total number of individual spikes to generate magnitudes for.

        Returns
        -------
        np.ndarray
            Array of spike magnitudes.
        """
        base_magnitude = np.random.uniform(*self.params.base_magnitude_range)
        magnitudes = np.zeros(total_spikes)

        if self.params.magnitude_pattern == MagnitudePattern.CONSTANT:
            # All spikes have similar magnitude with small noise
            magnitudes = np.full(total_spikes, base_magnitude)
            noise = np.random.normal(0, self.params.magnitude_noise * base_magnitude, total_spikes)
            magnitudes += noise

        elif self.params.magnitude_pattern == MagnitudePattern.INCREASING:
            # Magnitude increases over time
            trend = np.linspace(
                0,
                self.params.magnitude_trend_strength * base_magnitude * total_spikes,
                total_spikes,
            )
            magnitudes = base_magnitude + trend

        elif self.params.magnitude_pattern == MagnitudePattern.DECREASING:
            # Magnitude decreases over time
            trend = np.linspace(
                0,
                -self.params.magnitude_trend_strength * base_magnitude * total_spikes,
                total_spikes,
            )
            magnitudes = base_magnitude + trend

        elif self.params.magnitude_pattern == MagnitudePattern.CYCLICAL:
            # Cyclical magnitude pattern
            cycle_length = int(total_spikes * self.params.cyclical_period_ratio)
            if cycle_length == 0:
                cycle_length = max(1, total_spikes // 4)

            phase = np.linspace(0, 2 * np.pi * total_spikes / cycle_length, total_spikes)
            cyclical_component = 0.3 * base_magnitude * np.sin(phase)
            magnitudes = base_magnitude + cyclical_component

        elif self.params.magnitude_pattern == MagnitudePattern.RANDOM_BOUNDED:
            # Random with correlation between consecutive spikes
            magnitudes[0] = base_magnitude

            for i in range(1, total_spikes):
                # Correlated random walk
                prev_magnitude = magnitudes[i - 1]
                random_component = np.random.normal(0, 0.2 * base_magnitude)

                magnitudes[i] = (
                    self.params.magnitude_correlation * prev_magnitude
                    + (1 - self.params.magnitude_correlation) * base_magnitude
                    + random_component
                )

        # Add noise to all patterns
        noise = np.random.normal(0, self.params.magnitude_noise * base_magnitude, total_spikes)
        magnitudes += noise

        # Ensure magnitudes are positive and within reasonable bounds
        min_magnitude = 0.1 * base_magnitude
        max_magnitude = 3.0 * base_magnitude
        magnitudes = np.clip(magnitudes, min_magnitude, max_magnitude)

        return magnitudes

    def _inject_spike_anomalies(self, signal: np.ndarray, spike_direction: AnomalyType) -> np.ndarray:
        """
        Inject spike anomalies into the clean signal using realistic patterns.

        Parameters
        ----------
        signal : np.ndarray
            Clean baseline signal to inject spikes into.
        spike_direction : AnomalyType
            Direction of spikes for this series (all up or all down).

        Returns
        -------
        np.ndarray
            Signal with injected spike anomalies.
        """
        anomalous_signal = signal.copy()

        # Generate spike positions based on pattern
        spike_events = self._generate_spike_positions()

        # Flatten spike events to get total number of individual spikes
        all_positions = []
        for event in spike_events:
            all_positions.extend(event)

        if not all_positions:
            return anomalous_signal

        # Generate magnitudes for all spikes
        magnitudes = self._generate_spike_magnitudes(len(all_positions))

        # Inject spikes
        for i, position in enumerate(all_positions):
            if position < len(anomalous_signal):
                magnitude = magnitudes[i]

                if spike_direction == AnomalyType.SPIKE_UP:
                    anomalous_signal[position] += magnitude
                else:  # SPIKE_DOWN
                    anomalous_signal[position] -= magnitude

        return anomalous_signal

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate a synthetic time series with realistic spike anomalies.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Generated time series of shape (length,) - clean baseline with periodic spikes.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate clean baseline signal (constant level)
        baseline_level = np.random.uniform(*self.params.base_level_range)
        signal = np.full(self.params.length, baseline_level)

        # Determine spike direction for this series (all up or all down)
        spike_direction = self._determine_spike_direction()

        # Inject spike anomalies with realistic patterns
        anomalous_signal = self._inject_spike_anomalies(signal, spike_direction)

        return anomalous_signal
