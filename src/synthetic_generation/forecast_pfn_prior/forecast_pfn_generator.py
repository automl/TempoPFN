import logging

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from src.data.frequency import FREQUENCY_MAPPING, Frequency
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.forecast_pfn_prior.series_config import (
    ComponentNoise,
    ComponentScale,
    SeriesConfig,
)
from src.synthetic_generation.forecast_pfn_prior.utils import (
    get_random_walk_series,
    get_transition_coefficients,
    sample_scale,
    shift_axis,
    weibull_noise,
)
from src.synthetic_generation.generator_params import ForecastPFNGeneratorParams
from src.synthetic_generation.utils import generate_spikes


class ForecastPFNGenerator(AbstractTimeSeriesGenerator):
    """
    Time series generator based on the ForecastPFN prior.

    This generator creates synthetic time series with configurable trends, seasonality,
    and noise patterns. It includes built-in filtering to avoid extreme values and
    supports various univariate augmentations.
    """

    def __init__(
        self,
        params: ForecastPFNGeneratorParams,
        length: int = 1024,
        random_seed: int | None = None,
        max_absolute_spread: float = 300.0,
        max_absolute_value: float = 300.0,
        max_retries: int = 100,
    ) -> None:
        """
        Initialize the ForecastPFNGenerator.

        Parameters
        ----------
        params : ForecastPFNGeneratorParams
            Configuration parameters for the generator
        length : int, default=1024
            Length of the generated time series
        random_seed : Optional[int], default=None
            Random seed for reproducibility
        max_absolute_spread : float, default=300.0
            Maximum allowed spread (max - min) for generated series
        max_absolute_value : float, default=300.0
            Maximum allowed absolute value for any point in the series
        max_retries : int, default=100
            Maximum number of retry attempts if generation fails or produces extreme values
        """
        self.params = params
        self.length = length
        self.rng = np.random.default_rng(random_seed)
        self.frequency = params.frequency

        # Filtering parameters
        self.max_absolute_spread = max_absolute_spread
        self.max_absolute_value = max_absolute_value
        self.max_retries = max_retries

    def _calculate_scaled_exp_base(self, timescale: float) -> float:
        """
        Calculate an exponential base that is scaled according to the series length
        to prevent extreme values at the end of long sequences.

        Parameters
        ----------
        timescale : float
            The timescale factor for the frequency

        Returns
        -------
        float
            Scaled exponential base that keeps final values within reasonable bounds
        """
        if not self.params.trend_exp:
            return 1.0

        # Estimate maximum days in the series based on length and frequency
        # For most frequencies, each step represents timescale days
        max_days = self.length * timescale

        # Sample a raw exponential base with the original logic
        raw_exp_base = self.rng.normal(1, 0.005 / timescale)

        # Define reasonable bounds for the final exponential multiplier
        # Allow growth/decay up to 10x in either direction
        max_growth_factor = 10.0
        min_decay_factor = 0.1

        # Calculate what the maximum absolute exponent could be
        # considering the offset range of (-0.1, 0.5)
        # Worst case is when |1 - offset| * max_days is maximized
        max_abs_exponent = 1.1 * max_days  # Conservative estimate

        if raw_exp_base > 1.0:
            # For growth, ensure base^max_abs_exponent <= max_growth_factor
            max_allowed_base = max_growth_factor ** (1.0 / max_abs_exponent)
            scaled_base = min(raw_exp_base, max_allowed_base)
        elif raw_exp_base < 1.0:
            # For decay, ensure base^max_abs_exponent >= min_decay_factor
            min_allowed_base = min_decay_factor ** (1.0 / max_abs_exponent)
            scaled_base = max(raw_exp_base, min_allowed_base)
        else:
            scaled_base = raw_exp_base

        # Apply the original bounds as a final safety check
        return max(0.0001, min(1.01, scaled_base))

    def _is_series_acceptable(self, values: np.ndarray) -> bool:
        """
        Check if a generated time series meets the spread and value criteria.

        Parameters
        ----------
        values : np.ndarray
            Time series values to check

        Returns
        -------
        bool
            True if series is acceptable, False if it should be regenerated
        """
        if len(values) == 0:
            return False

        # Check for non-finite values
        if not np.all(np.isfinite(values)):
            return False

        min_val = np.min(values)
        max_val = np.max(values)
        spread = max_val - min_val

        # Check absolute spread threshold
        if spread > self.max_absolute_spread:
            return False

        # Check absolute value thresholds
        if abs(min_val) > self.max_absolute_value or abs(max_val) > self.max_absolute_value:
            return False

        return True

    def _generate_damping(self, input_size: int, p: list = None) -> np.ndarray:
        """Generate damping effect for a univariate time series."""
        if p is None:
            p = [0.4, 0.5, 0.1]

        spacing = self.rng.choice(["equal", "regular", "random"], p=p)
        t = np.arange(0, input_size, 1).astype(float)

        if spacing == "random":
            num_steps = self.rng.integers(1, 3)
            damping_intervals = np.sort(self.rng.choice(t[: -int(input_size * 0.1)], num_steps, replace=False))
            damping_factors = self.rng.uniform(0.1, 2, num_steps + 1)
        elif spacing == "equal":
            num_steps = self.rng.integers(3, 7)
            damping_intervals = np.linspace(0, input_size, num_steps + 2)[1:-1]
            damping_factors = np.array(
                [self.rng.uniform(0.4, 0.8) if (i % 2) == 0 else self.rng.uniform(1, 2) for i in range(num_steps + 1)]
            )
        else:
            custom_lengths = self.rng.integers(1, input_size // 2, 2)
            damping_intervals = []
            current_time = 0
            while current_time < input_size:
                for length in custom_lengths:
                    current_time += length
                    if current_time <= input_size:
                        damping_intervals.append(current_time)
                    else:
                        break
            damping_intervals = np.array(damping_intervals)
            num_steps = len(damping_intervals)
            damping_factors = np.array(
                [self.rng.uniform(0.4, 0.8) if (i % 2) == 0 else self.rng.uniform(1, 2) for i in range(num_steps + 1)]
            )

        damping = np.piecewise(
            t,
            [t < damping_intervals[0]]
            + [(t >= damping_intervals[i]) & (t < damping_intervals[i + 1]) for i in range(num_steps - 1)]
            + [t >= damping_intervals[-1]],
            damping_factors.tolist(),
        )
        return damping

    def _apply_time_warping(self, values: np.ndarray, warp_strength: float = 0.1) -> np.ndarray:
        """Apply time warping augmentation to univariate series."""
        length = len(values)
        # Create smooth random warping function
        n_knots = max(3, length // 20)  # Adaptive number of knots
        knot_positions = np.linspace(0, length - 1, n_knots)
        warp_offsets = self.rng.normal(0, warp_strength * length, n_knots)
        warp_offsets[0] = warp_offsets[-1] = 0  # Keep endpoints fixed

        # Interpolate to get smooth warping
        original_indices = np.arange(length)
        warped_indices = np.interp(original_indices, knot_positions, knot_positions + warp_offsets)
        warped_indices = np.clip(warped_indices, 0, length - 1)

        # Interpolate values at warped positions
        return np.interp(warped_indices, original_indices, values)

    def _apply_magnitude_scaling(self, values: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """Apply random magnitude scaling to different segments of the series."""
        length = len(values)
        num_segments = self.rng.integers(1, 4)
        segment_boundaries = np.sort(self.rng.choice(length, num_segments - 1, replace=False))
        segment_boundaries = np.concatenate([[0], segment_boundaries, [length]])

        scaled_values = values.copy()
        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
            scaled_values[start:end] *= scale_factor

        return scaled_values

    def _apply_univariate_augmentations(self, values: np.ndarray) -> np.ndarray:
        """Apply univariate-specific augmentations to a single time series."""
        augmented_values = np.asarray(values).copy()

        # Apply time warping with some probability
        if hasattr(self.params, "time_warp_prob") and self.rng.random() < self.params.time_warp_prob:
            warp_strength = getattr(self.params, "time_warp_strength", 0.05)
            augmented_values = self._apply_time_warping(augmented_values, warp_strength)

        # Apply magnitude scaling with some probability
        if hasattr(self.params, "magnitude_scale_prob") and self.rng.random() < self.params.magnitude_scale_prob:
            scale_range = getattr(self.params, "magnitude_scale_range", (0.9, 1.1))
            augmented_values = self._apply_magnitude_scaling(augmented_values, scale_range)

        # Apply damping augmentation
        if hasattr(self.params, "damping_prob") and self.rng.random() < self.params.damping_prob:
            damping = self._generate_damping(len(augmented_values))
            augmented_values = augmented_values * damping

        # Apply spike augmentation
        if hasattr(self.params, "spike_prob") and self.rng.random() < self.params.spike_prob:
            spikes = generate_spikes(len(augmented_values))
            spikes = spikes.numpy()  # Convert torch tensor to numpy array
            if spikes.max() < 0:
                augmented_values = augmented_values * spikes
            else:
                augmented_values = augmented_values + spikes + 1

        # Replace with pure spike signal (rare event)
        if hasattr(self.params, "pure_spike_prob") and self.rng.random() < self.params.pure_spike_prob:
            spikes = generate_spikes(len(augmented_values))
            augmented_values = spikes.numpy()  # Convert torch tensor to numpy array

        return augmented_values

    def generate_time_series(
        self,
        start: np.datetime64,
        random_seed: int | None = None,
        apply_augmentations: bool = True,
        frequency: Frequency | None = None,
    ) -> np.ndarray:
        """
        Generate a time series with built-in filtering and retry logic.

        If the generated series has extreme spreads, it will retry with different
        random seeds up to max_retries times.

        Parameters
        ----------
        start : np.datetime64
            Start date for the time series
        random_seed : Optional[int], default=None
            Random seed for reproducibility
        apply_augmentations : bool, default=True
            Whether to apply univariate augmentations
        frequency : Optional[Frequency], default=None
            Frequency for the time series. If None, uses self.frequency

        Returns
        -------
        np.ndarray
            Generated time series values of shape (length,)

        Raises
        ------
        RuntimeError
            If all generation attempts fail
        """
        original_seed = random_seed
        values = None  # Initialize to avoid UnboundLocalError

        for attempt in range(self.max_retries + 1):
            # Use different seed for each retry
            current_seed = original_seed
            if attempt > 0:
                # Generate a new seed based on the original seed and attempt number
                if original_seed is not None:
                    current_seed = original_seed + attempt * 123  # Large offset to ensure different sequences
                else:
                    current_seed = attempt * 987

            if current_seed is not None:
                self.rng = np.random.default_rng(current_seed)

            try:
                # Choose frequency per attempt: first attempt uses provided frequency (if any),
                # subsequent attempts randomly sample a supported frequency to avoid bad start/frequency combos
                supported_frequencies = [
                    Frequency.S,
                    Frequency.T1,
                    Frequency.T5,
                    Frequency.T10,
                    Frequency.T15,
                    Frequency.H,
                    Frequency.D,
                    Frequency.W,
                    Frequency.M,
                    Frequency.Q,
                    Frequency.A,
                ]
                if attempt == 0:
                    chosen_frequency = frequency if frequency is not None else self.frequency
                else:
                    # Sample a different frequency when possible
                    candidate_freqs = (
                        [f for f in supported_frequencies if f != (frequency or self.frequency)]
                        if (frequency or self.frequency) in supported_frequencies
                        else supported_frequencies
                    )
                    chosen_frequency = self.rng.choice(candidate_freqs)
                logging.debug(f"Attempt {attempt + 1}: Using frequency {chosen_frequency} (seed={current_seed})")

                values = self._generate_single_series(start, apply_augmentations, chosen_frequency)

                # Check if the generated series is acceptable
                values = np.asarray(values)
                if self._is_series_acceptable(values):
                    if attempt > 0:
                        logging.debug(f"Generated acceptable series on attempt {attempt + 1}")
                    return values
                else:
                    min_val = np.min(values) if len(values) > 0 else 0
                    max_val = np.max(values) if len(values) > 0 else 0
                    spread = max_val - min_val
                    logging.debug(
                        f"Attempt {attempt + 1}: Series rejected (spread={spread:.1f}, "
                        f"min={min_val:.1f}, max={max_val:.1f})"
                    )
                    continue

            except Exception as e:
                logging.debug(f"Attempt {attempt + 1}: Generation failed with error: {e}")
                # Keep the last successfully generated values (if any) for fallback
                continue

        # If all retries failed, raise an error as documented
        if values is None:
            raise RuntimeError(
                f"Failed to generate acceptable series after {self.max_retries + 1} attempts. "
                f"All generation attempts failed completely. Please try different parameters or increase max_retries."
            )
        else:
            # If we have values but they don't meet criteria, log warning and return them
            logging.warning(
                f"Failed to generate acceptable series after {self.max_retries + 1} attempts. "
                f"Returning last attempt (may have extreme values)."
            )
            return values

    def _generate_single_series(
        self,
        start: np.datetime64,
        apply_augmentations: bool = True,
        frequency: Frequency | None = None,
    ) -> np.ndarray:
        """
        Generate a single time series attempt (extracted from original generate_time_series).
        """
        # Use provided frequency or fall back to self.frequency
        effective_frequency = frequency if frequency is not None else self.frequency

        freq_key, subfreq, timescale = FREQUENCY_MAPPING.get(effective_frequency, ("D", "", 1))
        freq = f"{subfreq}{freq_key}" if subfreq else freq_key

        # Seasonal component weights based on frequency
        a, m, w, h, minute = 0.0, 0.0, 0.0, 0.0, 0.0
        if effective_frequency == Frequency.S:
            minute = self.rng.uniform(0.0, 1.0)
            h = self.rng.uniform(0.0, 0.2)
        elif effective_frequency in [
            Frequency.T1,
            Frequency.T5,
            Frequency.T10,
            Frequency.T15,
        ]:
            minute = self.rng.uniform(0.0, 1.0)
            h = self.rng.uniform(0.0, 0.2)
        elif effective_frequency == Frequency.H:
            minute = self.rng.uniform(0.0, 0.2)
            h = self.rng.uniform(0.0, 1.0)
        elif effective_frequency == Frequency.D:
            w = self.rng.uniform(0.0, 1.0)
            m = self.rng.uniform(0.0, 0.2)
        elif effective_frequency == Frequency.W:
            m = self.rng.uniform(0.0, 0.3)
            a = self.rng.uniform(0.0, 0.3)
        elif effective_frequency == Frequency.M:
            w = self.rng.uniform(0.0, 0.1)
            a = self.rng.uniform(0.0, 0.5)
        elif effective_frequency == Frequency.Q:
            a = self.rng.uniform(0.0, 1.0)
        elif effective_frequency == Frequency.A:
            w = self.rng.uniform(0.0, 0.2)
            a = self.rng.uniform(0.0, 1.0)
        else:
            raise NotImplementedError(f"Frequency {effective_frequency} not supported")

        scale_config = ComponentScale(
            base=1.0,
            linear=self.rng.normal(0, 0.01),
            exp=self._calculate_scaled_exp_base(timescale) if self.params.trend_exp else 1.0,
            a=a,
            m=m,
            w=w,
            minute=minute,
            h=h,
        )

        offset_config = ComponentScale(
            base=0,
            linear=self.rng.uniform(-0.1, 0.5),
            exp=self.rng.uniform(-0.1, 0.5),
            a=self.rng.uniform(0.0, 1.0),
            m=self.rng.uniform(0.0, 1.0),
            w=self.rng.uniform(0.0, 1.0),
        )

        noise_config = ComponentNoise(
            k=self.rng.uniform(1, 5),
            median=1,
            scale=sample_scale(
                low_ratio=self.params.scale_noise[0],
                moderate_ratio=self.params.scale_noise[1],
                rng=self.rng,
            ),
        )

        cfg = SeriesConfig(scale_config, offset_config, noise_config)
        options = {
            "trend_exp": self.params.trend_exp,
            "scale_noise": self.params.scale_noise,
            "harmonic_scale_ratio": self.params.harmonic_scale_ratio,
            "harmonic_rate": self.params.harmonic_rate,
            "period_factor": self.params.period_factor,
            "seasonal_only": self.params.seasonal_only,
            "trend_additional": self.params.trend_additional,
        }

        # Generate first series
        series1 = self._make_series(cfg, to_offset(freq), start, options, self.params.random_walk)

        # Generate second series for transition if enabled
        transition = self.rng.random() < self.params.transition_ratio
        if transition:
            cfg2 = SeriesConfig(
                ComponentScale(
                    base=1.0,
                    linear=self.rng.normal(0, 0.01),
                    exp=self._calculate_scaled_exp_base(timescale) if self.params.trend_exp else 1.0,
                    a=a,
                    m=m,
                    w=w,
                    minute=minute,
                    h=h,
                ),
                ComponentScale(
                    base=0,
                    linear=self.rng.uniform(-0.1, 0.5),
                    exp=self.rng.uniform(-0.1, 0.5),
                    a=self.rng.uniform(0.0, 1.0),
                    m=self.rng.uniform(0.0, 1.0),
                    w=self.rng.uniform(0.0, 1.0),
                ),
                ComponentNoise(
                    k=self.rng.uniform(1, 5),
                    median=1,
                    scale=sample_scale(
                        low_ratio=self.params.scale_noise[0],
                        moderate_ratio=self.params.scale_noise[1],
                        rng=self.rng,
                    ),
                ),
            )
            series2 = self._make_series(cfg2, to_offset(freq), start, options, self.params.random_walk)
            coeff = get_transition_coefficients(self.length)
            values = coeff * series1["values"] + (1 - coeff) * series2["values"]
        else:
            values = series1["values"]

        # Apply univariate augmentations if requested
        if apply_augmentations:
            # Ensure values is a numpy array before augmentation
            values = np.asarray(values)
            values = self._apply_univariate_augmentations(values)

        return values

    def _make_series(
        self,
        series: SeriesConfig,
        freq: pd.DateOffset,
        start: np.datetime64,
        options: dict,
        random_walk: bool,
    ) -> dict:
        start = freq.rollback(start)
        dates = pd.date_range(start=start, periods=self.length, freq=freq)
        scaled_noise_term = 0
        values_seasonal = {}

        if random_walk:
            values = get_random_walk_series(len(dates), rng=self.rng)
        elif options["seasonal_only"]:
            values_seasonal = self._make_series_seasonal(series, dates, options)
            values = values_seasonal["seasonal"]
        else:
            values_trend = self._make_series_trend(series, dates)
            values_seasonal = self._make_series_seasonal(series, dates, options)
            values = (
                values_trend + values_seasonal["seasonal"]
                if options["trend_additional"]
                else values_trend * values_seasonal["seasonal"]
            )

            weibull_noise_term = weibull_noise(
                k=series.noise_config.k,
                median=series.noise_config.median,
                length=len(values),
                rng=self.rng,
            )
            noise_expected_val = series.noise_config.median
            scaled_noise_term = series.noise_config.scale * (weibull_noise_term - noise_expected_val)
            values = values * (1 + scaled_noise_term)

        return {
            "values": values,
            "noise": 1 + scaled_noise_term,
            "dates": dates,
            "seasonal": values_seasonal.get("seasonal", np.ones_like(values)),
        }

    def _make_series_trend(self, series: SeriesConfig, dates: pd.DatetimeIndex) -> np.ndarray:
        values = np.full_like(dates, series.scale.base, dtype=np.float32)
        days = (dates - dates[0]).days
        if series.scale.linear is not None:
            values += shift_axis(days, series.offset.linear) * series.scale.linear
        if series.scale.exp is not None:
            values *= np.power(series.scale.exp, shift_axis(days, series.offset.exp))

        return values

    def _make_series_seasonal(self, series: SeriesConfig, dates: pd.DatetimeIndex, options: dict) -> dict:
        seasonal = 1
        harmonic_scale = self.rng.random() < options["harmonic_scale_ratio"]
        harmonic_rate = options["harmonic_rate"]
        period_factor = options["period_factor"]
        seasonal_components = {}
        if series.scale.minute is not None and series.scale.minute != 0:
            seasonal_components["minute"] = 1 + series.scale.minute * self._get_freq_component(
                dates.minute,
                int(np.ceil(10 * harmonic_rate)),
                60 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["minute"]
        if series.scale.h is not None and series.scale.h != 0:
            seasonal_components["h"] = 1 + series.scale.h * self._get_freq_component(
                dates.hour,
                int(np.ceil(10 * harmonic_rate)),
                24 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["h"]
        if series.scale.a is not None and series.scale.a != 0:
            seasonal_components["a"] = 1 + series.scale.a * self._get_freq_component(
                dates.month,
                int(np.ceil(6 * harmonic_rate)),
                12 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["a"]
        if series.scale.m is not None and series.scale.m != 0:
            seasonal_components["m"] = 1 + series.scale.m * self._get_freq_component(
                dates.day,
                int(np.ceil(10 * harmonic_rate)),
                30.5 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["m"]
        if series.scale.w is not None and series.scale.w != 0:
            seasonal_components["w"] = 1 + series.scale.w * self._get_freq_component(
                dates.dayofweek,
                int(np.ceil(4 * harmonic_rate)),
                7 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["w"]
        seasonal_components["seasonal"] = seasonal
        return seasonal_components

    def _get_freq_component(
        self,
        dates_feature: pd.Index,
        n_harmonics: int,
        n_total: float,
        harmonic_scale: bool = True,
    ) -> np.ndarray:
        harmonics = list(range(1, n_harmonics + 1))
        sin_coef = np.zeros(n_harmonics)
        cos_coef = np.zeros(n_harmonics)
        for idx, harmonic in enumerate(harmonics):
            h = 1 if not harmonic_scale else harmonic
            sin_coef[idx] = self.rng.normal(scale=1 / h)
            cos_coef[idx] = self.rng.normal(scale=1 / h)
        coef_sq_sum = np.sqrt(np.sum(np.square(sin_coef)) + np.sum(np.square(cos_coef)))
        sin_coef /= coef_sq_sum
        cos_coef /= coef_sq_sum
        return_val = 0
        for idx, harmonic in enumerate(harmonics):
            return_val += sin_coef[idx] * np.sin(2 * np.pi * harmonic * dates_feature / n_total)
            return_val += cos_coef[idx] * np.cos(2 * np.pi * harmonic * dates_feature / n_total)
        return return_val
