from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.generator_params import (
    StepGeneratorParams,
    StepPatternType,
    SubseriesConfig,
)


class StepGenerator(AbstractTimeSeriesGenerator):
    """
    Generator for step function time series.
    Creates realistic step functions with optional seasonality, trend, and noise.
    """

    def __init__(self, params: StepGeneratorParams):
        """
        Initialize the StepGenerator.

        Parameters
        ----------
        params : StepGeneratorParams
            Parameters controlling the step function generation.
        """
        self.params = params
        self.rng = np.random.default_rng(params.global_seed)

    def _select_subseries_configs(self) -> List[Tuple[SubseriesConfig, int]]:
        """
        Select which subseries patterns to use and their lengths.

        Returns
        -------
        List[Tuple[SubseriesConfig, int]]
            List of (config, length) tuples for each subseries.
        """
        # Determine number of subseries
        num_subseries = self.rng.integers(
            self.params.min_subseries, self.params.max_subseries + 1
        )

        # Calculate weights for pattern selection
        configs = self.params.subseries_configs
        weights = np.array([config.weight for config in configs])
        weights = weights / weights.sum()

        # Select patterns
        selected_configs = []
        remaining_length = self.params.length

        for i in range(num_subseries):
            # Select pattern
            config_idx = self.rng.choice(len(configs), p=weights)
            config = configs[config_idx]

            # Determine length for this subseries
            if i == num_subseries - 1:
                # Last subseries gets remaining length
                length = remaining_length
            else:
                # Sample length from range, but ensure we don't exceed remaining
                min_length = min(
                    config.length_range[0], remaining_length // (num_subseries - i)
                )
                max_length = min(
                    config.length_range[1],
                    remaining_length - (num_subseries - i - 1) * 50,
                )
                max_length = max(min_length, max_length)

                length = self.rng.integers(min_length, max_length + 1)
                remaining_length -= length

            selected_configs.append((config, length))

        return selected_configs

    def _generate_changepoints_for_pattern(
        self, config: SubseriesConfig, length: int
    ) -> np.ndarray:
        """
        Generate changepoints for a specific pattern type.

        Parameters
        ----------
        config : SubseriesConfig
            Configuration for this subseries
        length : int
            Length of the subseries

        Returns
        -------
        np.ndarray
            Array of changepoint positions
        """
        num_changepoints = self.rng.integers(
            config.num_changepoints_range[0], config.num_changepoints_range[1] + 1
        )

        if num_changepoints == 0:
            return np.array([])

        # Ensure minimum spacing between changepoints
        min_spacing = max(1, length // (num_changepoints * 2))

        if config.pattern_type == StepPatternType.STABLE:
            # Few changepoints, mostly at the beginning or end
            if num_changepoints > 0:
                changepoints = self.rng.choice(
                    np.arange(length // 4, 3 * length // 4),
                    size=min(num_changepoints, length // 2),
                    replace=False,
                )
            else:
                changepoints = np.array([])

        elif config.pattern_type in [
            StepPatternType.GRADUAL_INCREASE,
            StepPatternType.GRADUAL_DECREASE,
        ]:
            # More evenly distributed
            changepoints = np.linspace(
                length // 10, 9 * length // 10, num_changepoints
            ).astype(int)
            # Add some randomness
            noise = self.rng.integers(
                -min_spacing, min_spacing + 1, size=num_changepoints
            )
            changepoints = np.clip(changepoints + noise, 0, length - 1)

        elif config.pattern_type in [
            StepPatternType.SPIKE_UP,
            StepPatternType.SPIKE_DOWN,
        ]:
            # Concentrated in the first third, then spread out
            first_third = length // 3
            num_first_third = max(1, num_changepoints // 2)
            num_rest = num_changepoints - num_first_third

            if num_first_third > 0:
                changepoints_first = np.linspace(
                    length // 20, first_third, num_first_third
                ).astype(int)
            else:
                changepoints_first = np.array([])

            if num_rest > 0:
                changepoints_rest = np.linspace(
                    first_third + 1, 9 * length // 10, num_rest
                ).astype(int)
            else:
                changepoints_rest = np.array([])

            changepoints = np.concatenate([changepoints_first, changepoints_rest])

        elif config.pattern_type == StepPatternType.OSCILLATING:
            # Regular spacing
            changepoints = np.linspace(
                length // 10, 9 * length // 10, num_changepoints
            ).astype(int)

        else:  # RANDOM_WALK
            # Random distribution
            changepoints = self.rng.choice(
                np.arange(length // 10, 9 * length // 10),
                size=min(num_changepoints, length // 2),
                replace=False,
            )

        return np.sort(changepoints)

    def _generate_step_sizes_for_pattern(
        self, config: SubseriesConfig, num_changepoints: int
    ) -> np.ndarray:
        """
        Generate step sizes for a specific pattern type.

        Parameters
        ----------
        config : SubseriesConfig
            Configuration for this subseries
        num_changepoints : int
            Number of changepoints

        Returns
        -------
        np.ndarray
            Array of step sizes
        """
        if num_changepoints == 0:
            return np.array([])

        # Generate base step sizes
        step_sizes = self.rng.uniform(
            config.step_size_range[0], config.step_size_range[1], num_changepoints
        )

        if config.pattern_type == StepPatternType.STABLE:
            # Very small steps
            return step_sizes * 0.1

        elif config.pattern_type == StepPatternType.GRADUAL_INCREASE:
            # All positive steps with optional decay
            step_sizes = np.abs(step_sizes)
            if config.step_size_decay != 1.0:
                decay_factors = np.power(
                    config.step_size_decay, np.arange(num_changepoints)
                )
                step_sizes = step_sizes * decay_factors
            return step_sizes

        elif config.pattern_type == StepPatternType.GRADUAL_DECREASE:
            # All negative steps with optional decay
            step_sizes = -np.abs(step_sizes)
            if config.step_size_decay != 1.0:
                decay_factors = np.power(
                    config.step_size_decay, np.arange(num_changepoints)
                )
                step_sizes = step_sizes * decay_factors
            return step_sizes

        elif config.pattern_type == StepPatternType.SPIKE_UP:
            # Large positive steps early, then smaller negative steps
            step_sizes = np.abs(step_sizes)
            mid_point = num_changepoints // 2
            step_sizes[mid_point:] = -step_sizes[mid_point:] * 0.5

            # Apply decay
            if config.step_size_decay != 1.0:
                decay_factors = np.power(
                    config.step_size_decay, np.arange(num_changepoints)
                )
                step_sizes = step_sizes * decay_factors
            return step_sizes

        elif config.pattern_type == StepPatternType.SPIKE_DOWN:
            # Large negative steps early, then smaller positive steps
            step_sizes = -np.abs(step_sizes)
            mid_point = num_changepoints // 2
            step_sizes[mid_point:] = -step_sizes[mid_point:] * 0.5

            # Apply decay
            if config.step_size_decay != 1.0:
                decay_factors = np.power(
                    config.step_size_decay, np.arange(num_changepoints)
                )
                step_sizes = step_sizes * decay_factors
            return step_sizes

        elif config.pattern_type == StepPatternType.OSCILLATING:
            # Alternating positive and negative steps
            step_sizes = np.abs(step_sizes)
            step_sizes[1::2] *= -1  # Make every other step negative
            return step_sizes

        else:  # RANDOM_WALK
            return step_sizes

    def _generate_subseries(
        self, config: SubseriesConfig, length: int, start_level: float
    ) -> np.ndarray:
        """
        Generate a single subseries with the specified pattern.

        Parameters
        ----------
        config : SubseriesConfig
            Configuration for this subseries
        length : int
            Length of the subseries
        start_level : float
            Starting level for this subseries

        Returns
        -------
        np.ndarray
            Generated subseries
        """
        # Generate changepoints and step sizes
        changepoints = self._generate_changepoints_for_pattern(config, length)
        step_sizes = self._generate_step_sizes_for_pattern(config, len(changepoints))

        # Initialize subseries with start level
        subseries = np.full(length, start_level)

        # Apply steps
        current_level = start_level
        for changepoint, step_size in zip(changepoints, step_sizes):
            current_level += step_size
            subseries[changepoint:] = current_level

        # Apply level drift if specified
        if config.level_drift_range[0] != 0 or config.level_drift_range[1] != 0:
            drift = self.rng.uniform(
                config.level_drift_range[0], config.level_drift_range[1]
            )
            drift_array = np.linspace(0, drift, length)
            subseries += drift_array

        return subseries

    def _create_combined_step_function(self) -> np.ndarray:
        """
        Create a combined step function from multiple subseries.

        Returns
        -------
        np.ndarray
            Combined step function
        """
        # Select subseries configurations
        subseries_configs = self._select_subseries_configs()

        # Generate base level
        base_level = self.rng.uniform(
            self.params.base_level_range[0], self.params.base_level_range[1]
        )

        # Generate subseries
        combined_series = []
        current_level = base_level

        for config, length in subseries_configs:
            # Generate subseries
            subseries = self._generate_subseries(config, length, current_level)

            # Ensure level continuity if required
            if (
                self.params.maintain_level_continuity
                and len(combined_series) > 0
                and len(subseries) > 0
            ):
                level_diff = subseries[0] - current_level
                if abs(level_diff) > self.params.max_level_jump_between_subseries:
                    # Adjust subseries to maintain continuity
                    adjustment = (
                        level_diff
                        - np.sign(level_diff)
                        * self.params.max_level_jump_between_subseries
                    )
                    subseries -= adjustment

            combined_series.append(subseries)
            current_level = subseries[-1]

        # Concatenate all subseries
        combined_series = np.concatenate(combined_series)

        # Apply transition smoothing if enabled
        if self.params.enable_smooth_transitions and len(subseries_configs) > 1:
            # Find transition points
            transition_points = []
            cumulative_length = 0
            for config, length in subseries_configs[:-1]:  # Exclude last
                cumulative_length += length
                transition_points.append(cumulative_length)

            # Smooth transitions
            for transition_point in transition_points:
                start_idx = max(
                    0, transition_point - self.params.transition_length // 2
                )
                end_idx = min(
                    len(combined_series),
                    transition_point + self.params.transition_length // 2,
                )

                if end_idx - start_idx > 2:
                    # Apply light Gaussian smoothing only to transition regions
                    combined_series[start_idx:end_idx] = gaussian_filter1d(
                        combined_series[start_idx:end_idx],
                        sigma=1.0,  # Very light smoothing
                    )

        # Ensure exact length
        if len(combined_series) > self.params.length:
            combined_series = combined_series[: self.params.length]
        elif len(combined_series) < self.params.length:
            # Pad with the last value
            padding = np.full(
                self.params.length - len(combined_series), combined_series[-1]
            )
            combined_series = np.concatenate([combined_series, padding])

        return combined_series

    def generate_time_series(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single step function time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Generated time series of shape (length,).
        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        # Create the main step function
        step_function = self._create_combined_step_function()

        # Add noise
        if self.params.noise_level_range[0] > 0 or self.params.noise_level_range[1] > 0:
            noise_level = self.rng.uniform(
                self.params.noise_level_range[0], self.params.noise_level_range[1]
            )
            noise = self.rng.normal(0, noise_level, size=len(step_function))
            step_function += noise

        # Add seasonality using simple sine waves if enabled
        if self.params.add_seasonality:
            # Daily seasonality
            if self.params.daily_seasonality_amplitude_range[1] > 0:
                daily_amplitude = self.rng.uniform(
                    self.params.daily_seasonality_amplitude_range[0],
                    self.params.daily_seasonality_amplitude_range[1],
                )
                daily_period = 288  # 5-minute intervals in a day
                t = np.arange(len(step_function))
                daily_seasonality = daily_amplitude * np.sin(
                    2 * np.pi * t / daily_period
                )
                step_function += daily_seasonality

            # Weekly seasonality
            if self.params.weekly_seasonality_amplitude_range[1] > 0:
                weekly_amplitude = self.rng.uniform(
                    self.params.weekly_seasonality_amplitude_range[0],
                    self.params.weekly_seasonality_amplitude_range[1],
                )
                weekly_period = 288 * 7  # 7 days
                t = np.arange(len(step_function))
                weekly_seasonality = weekly_amplitude * np.sin(
                    2 * np.pi * t / weekly_period
                )
                step_function += weekly_seasonality

        # Add trend if enabled
        if self.params.add_trend:
            slope = self.rng.uniform(
                self.params.trend_slope_range[0], self.params.trend_slope_range[1]
            )
            trend = slope * np.arange(len(step_function))
            step_function += trend

        # Scale the signal
        scale_factor = self.rng.uniform(
            self.params.scale_range[0], self.params.scale_range[1]
        )
        step_function *= scale_factor

        # Inject anomalies if enabled
        if self.params.inject_anomalies:
            anomaly_indicators = (
                self.rng.random(len(step_function)) < self.params.anomaly_probability
            )
            anomaly_magnitudes = self.rng.uniform(
                self.params.anomaly_magnitude_range[0],
                self.params.anomaly_magnitude_range[1],
                size=len(step_function),
            )
            step_function[anomaly_indicators] += anomaly_magnitudes[anomaly_indicators]

        return step_function
