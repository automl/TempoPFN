import numpy as np

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator


class SawToothGenerator(AbstractTimeSeriesGenerator):
    """
    Generate synthetic univariate time series using sawtooth waves with configurable parameters.

    Each series is a sawtooth wave with random amplitude, frequency, phase. The sawtooth direction
    is randomly flipped 50% of the time to create both upward-ramping and downward-ramping patterns.
    The generator emphasizes straight line components with minimal wiggly seasonality for cleaner patterns.
    """

    def __init__(
        self,
        length: int = 2048,
        periods: tuple[int, int] = (3, 6),
        amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.5, 3.0),
        phase_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.0, 1.0),
        trend_slope_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (-0.001, 0.001),
        seasonality_amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
            0.0,
            0.02,
        ),
        add_trend: bool = True,
        add_seasonality: bool = True,
        random_seed: int | None = None,
    ):
        """
        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 2048).
        periods : tuple, optional
            (min_periods, max_periods) for number of sawtooth periods in the series (default: (3, 6)).
        amplitude_range : tuple, optional
            (min_amplitude, max_amplitude) or
            ((min_min, min_max), (max_min, max_max)) for sawtooth wave amplitude
            (default: (0.5, 3.0)).
        phase_range : tuple, optional
            (min_phase, max_phase) or
            ((min_min, min_max), (max_min, max_max)) for sawtooth wave phase as
            fraction of period (default: (0.0, 1.0)).
        trend_slope_range : tuple, optional
            (min_slope, max_slope) or
            ((min_min, min_max), (max_min, max_max)) for linear trend slope,
            emphasizing straight line components (default: (-0.001, 0.001)).
        seasonality_amplitude_range : tuple, optional
            (min_amplitude, max_amplitude) or
            ((min_min, min_max), (max_min, max_max)) for minimal seasonal
            component amplitude to reduce wiggly lines (default: (0.0, 0.02)).
        add_trend : bool, optional
            Whether to add linear trend component (default: True).
        add_seasonality : bool, optional
            Whether to add minimal seasonal component (default: True).
        random_seed : int, optional
            Seed for the random number generator.
        """
        self.length = length
        self.periods = periods
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.trend_slope_range = trend_slope_range
        self.seasonality_amplitude_range = seasonality_amplitude_range
        self.add_trend = add_trend
        self.add_seasonality = add_seasonality
        self.rng = np.random.default_rng(random_seed)

    def _sample_range_parameter(self, param_range):
        """Sample a range parameter that could be a fixed tuple or a tuple of ranges."""
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Check if it's a range of ranges: ((min_min, min_max), (max_min, max_max))
            if isinstance(param_range[0], tuple) and isinstance(param_range[1], tuple):
                min_val = self.rng.uniform(param_range[0][0], param_range[0][1])
                max_val = self.rng.uniform(param_range[1][0], param_range[1][1])
                # Ensure min_val <= max_val
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                return (min_val, max_val)
            else:
                # Fixed range
                return param_range
        else:
            raise ValueError(f"Invalid range parameter format: {param_range}")

    def _generate_sawtooth(
        self,
        time_idx: np.ndarray,
        period: float,
        amplitude: float,
        phase: float,
        flip: bool = False,
    ) -> np.ndarray:
        """Generate a sawtooth wave using period instead of frequency, optionally flipped."""
        # Convert time indices to actual time (assuming unit time steps)
        time = time_idx.astype(float)

        # Calculate frequency from period
        frequency = 1.0 / period

        # Calculate cycles with phase shift
        cycles = frequency * time + phase

        # Generate sawtooth wave: linear rise from 0 to 1, then drop back to 0
        if flip:
            # Flipped sawtooth: linear drop from 1 to 0, then jump back to 1
            sawtooth = amplitude * (1.0 - (cycles - np.floor(cycles)))
        else:
            # Normal sawtooth: linear rise from 0 to 1, then drop back to 0
            sawtooth = amplitude * (cycles - np.floor(cycles))

        return sawtooth

    def _generate_trend(self, time_idx: np.ndarray, slope: float) -> np.ndarray:
        """Generate linear trend component."""
        return slope * time_idx.astype(float)

    def _generate_seasonality(self, time_idx: np.ndarray, amplitude: float, period: float) -> np.ndarray:
        """Generate seasonal component using sine wave."""
        time = time_idx.astype(float)
        return amplitude * np.sin(2 * np.pi * time / period)

    def generate_time_series(self, random_seed: int | None = None) -> dict[str, np.ndarray]:
        """
        Generate a single univariate sawtooth wave time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible generation.

        Returns
        -------
        np.ndarray
            Shape: [seq_len]
        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        # Sample sawtooth wave parameters
        sampled_amplitude_range = self._sample_range_parameter(self.amplitude_range)
        sampled_phase_range = self._sample_range_parameter(self.phase_range)

        amplitude = self.rng.uniform(sampled_amplitude_range[0], sampled_amplitude_range[1])
        phase = self.rng.uniform(sampled_phase_range[0], sampled_phase_range[1])

        # Sample number of periods and calculate period length
        num_periods = self.rng.uniform(self.periods[0], self.periods[1])
        sawtooth_period = self.length / num_periods

        # Calculate seasonality period (use longer period for minimal seasonality)
        seasonality_period = self.length / self.rng.uniform(2.0, 4.0)  # 2-4 seasonality cycles

        # Randomly decide whether to flip the sawtooth wave (50% chance)
        flip_sawtooth = self.rng.random() < 0.5

        # Generate time indices
        time_idx = np.arange(self.length)

        # Generate base sawtooth wave
        values = self._generate_sawtooth(time_idx, sawtooth_period, amplitude, phase, flip_sawtooth)

        # Add trend if enabled
        if self.add_trend:
            sampled_trend_range = self._sample_range_parameter(self.trend_slope_range)
            trend_slope = self.rng.uniform(sampled_trend_range[0], sampled_trend_range[1])
            trend = self._generate_trend(time_idx, trend_slope)
            values += trend

        # Add minimal seasonality if enabled
        if self.add_seasonality:
            sampled_seasonality_amplitude_range = self._sample_range_parameter(self.seasonality_amplitude_range)

            seasonality_amplitude = self.rng.uniform(
                sampled_seasonality_amplitude_range[0],
                sampled_seasonality_amplitude_range[1],
            )

            if seasonality_amplitude > 0:  # Only add seasonality if amplitude > 0
                seasonality = self._generate_seasonality(time_idx, seasonality_amplitude, seasonality_period)
                values += seasonality

        return values
