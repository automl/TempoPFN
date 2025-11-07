import numpy as np

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator


class SineWaveGenerator(AbstractTimeSeriesGenerator):
    """
    Generate synthetic univariate time series using sinusoidal patterns with configurable parameters.

    This generator creates diverse sinusoidal series with:
    - Multiple sinusoidal components (seasonalities)
    - Linear trends
    - Small additive noise
    - Time-varying parameters for realism

    The output maintains clear sinusoidal characteristics while adding realistic variations.
    """

    def __init__(
        self,
        length: int = 1024,
        # Core sinusoidal parameters
        num_components_range: tuple[int, int] = (1, 3),
        period_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (10, 200),
        amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.5, 3.0),
        phase_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0, 2 * np.pi),
        # Trend parameters
        trend_slope_range: tuple[float, float] = (-0.01, 0.01),
        base_level_range: tuple[float, float] = (0.0, 2.0),
        # Noise parameters
        noise_probability: float = 0.7,  # Probability of adding noise (70% of series have noise)
        noise_level_range: tuple[float, float] = (
            0.05,
            0.2,
        ),  # Small noise as fraction of amplitude (when noise is applied)
        # Time-varying parameters (subtle)
        enable_amplitude_modulation: bool = True,
        amplitude_modulation_strength: float = 0.1,  # Max 10% amplitude variation
        enable_frequency_modulation: bool = True,
        frequency_modulation_strength: float = 0.05,  # Max 5% frequency variation
        random_seed: int | None = None,
    ):
        """
        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 1024).
        num_components_range : tuple, optional
            Range for number of sinusoidal components to combine (default: (1, 3)).
        period_range : tuple, optional
            Period range for sinusoidal components (default: (10, 200)).
        amplitude_range : tuple, optional
            Amplitude range for sinusoidal components (default: (0.5, 3.0)).
        phase_range : tuple, optional
            Phase range for sinusoidal components (default: (0, 2*pi)).
        trend_slope_range : tuple, optional
            Range for linear trend slope (default: (-0.01, 0.01)).
        base_level_range : tuple, optional
            Range for base level offset (default: (0.0, 2.0)).
        noise_probability : float, optional
            Probability of adding noise to a series (default: 0.7).
        noise_level_range : tuple, optional
            Range for noise level as fraction of total amplitude when noise is applied (default: (0.05, 0.2)).
        enable_amplitude_modulation : bool, optional
            Whether to enable subtle amplitude modulation (default: True).
        amplitude_modulation_strength : float, optional
            Strength of amplitude modulation (default: 0.1).
        enable_frequency_modulation : bool, optional
            Whether to enable subtle frequency modulation (default: True).
        frequency_modulation_strength : float, optional
            Strength of frequency modulation (default: 0.05).
        random_seed : int, optional
            Seed for the random number generator.
        """
        self.length = length
        self.num_components_range = num_components_range
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.trend_slope_range = trend_slope_range
        self.base_level_range = base_level_range
        self.noise_probability = noise_probability
        self.noise_level_range = noise_level_range
        self.enable_amplitude_modulation = enable_amplitude_modulation
        self.amplitude_modulation_strength = amplitude_modulation_strength
        self.enable_frequency_modulation = enable_frequency_modulation
        self.frequency_modulation_strength = frequency_modulation_strength
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

    def _sample_scalar_parameter(self, param):
        """Sample a scalar parameter that could be a fixed value or a range."""
        if isinstance(param, (int, float)):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            return self.rng.uniform(param[0], param[1])
        else:
            raise ValueError(f"Invalid scalar parameter format: {param}")

    def _generate_sinusoidal_components(self, t_array: np.ndarray, components: list[dict]) -> np.ndarray:
        """Generate sinusoidal signal from multiple components."""
        signal = np.zeros_like(t_array)

        for comp in components:
            amplitude = comp["amplitude"]
            period = comp["period"]
            phase = comp["phase"]

            # Basic sinusoidal component
            base_signal = amplitude * np.sin(2 * np.pi * t_array / period + phase)

            # Apply subtle amplitude modulation if enabled
            if self.enable_amplitude_modulation:
                # Use a slow modulation (period is 5-10x the main period)
                mod_period = period * self.rng.uniform(5, 10)
                mod_phase = self.rng.uniform(0, 2 * np.pi)
                amp_modulation = 1 + self.amplitude_modulation_strength * np.sin(
                    2 * np.pi * t_array / mod_period + mod_phase
                )
                base_signal *= amp_modulation

            # Apply subtle frequency modulation if enabled
            if self.enable_frequency_modulation:
                # Frequency modulation creates slight warping in the sine wave
                mod_period = period * self.rng.uniform(8, 15)
                mod_phase = self.rng.uniform(0, 2 * np.pi)
                freq_modulation = self.frequency_modulation_strength * np.sin(
                    2 * np.pi * t_array / mod_period + mod_phase
                )
                # Apply frequency modulation by modifying the phase
                instantaneous_freq = 2 * np.pi / period * (1 + freq_modulation)
                modulated_phase = np.cumsum(instantaneous_freq) * (t_array[1] - t_array[0]) + phase
                base_signal = amplitude * np.sin(modulated_phase)

                # Apply amplitude modulation on top if both are enabled
                if self.enable_amplitude_modulation:
                    mod_period_amp = period * self.rng.uniform(5, 10)
                    mod_phase_amp = self.rng.uniform(0, 2 * np.pi)
                    amp_modulation = 1 + self.amplitude_modulation_strength * np.sin(
                        2 * np.pi * t_array / mod_period_amp + mod_phase_amp
                    )
                    base_signal *= amp_modulation

            signal += base_signal

        return signal

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate a single univariate sinusoidal time series with trends and noise.

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

        # Generate time array
        t_array = np.linspace(0, self.length - 1, self.length)

        # Sample number of sinusoidal components
        num_components = self.rng.integers(self.num_components_range[0], self.num_components_range[1] + 1)

        # Sample parameters for each component
        components = []
        total_amplitude = 0

        for _ in range(num_components):
            sampled_period_range = self._sample_range_parameter(self.period_range)
            sampled_amplitude_range = self._sample_range_parameter(self.amplitude_range)
            sampled_phase_range = self._sample_range_parameter(self.phase_range)

            period = self.rng.uniform(sampled_period_range[0], sampled_period_range[1])
            amplitude = self.rng.uniform(sampled_amplitude_range[0], sampled_amplitude_range[1])
            phase = self.rng.uniform(sampled_phase_range[0], sampled_phase_range[1])

            components.append({"period": period, "amplitude": amplitude, "phase": phase})
            total_amplitude += amplitude

        # Generate sinusoidal signal
        signal = self._generate_sinusoidal_components(t_array, components)

        # Add linear trend
        trend_slope = self.rng.uniform(self.trend_slope_range[0], self.trend_slope_range[1])
        trend = trend_slope * t_array

        # Add base level
        base_level = self.rng.uniform(self.base_level_range[0], self.base_level_range[1])

        # Combine signal, trend, and base level
        values = signal + trend + base_level

        # Add noise with specified probability (70% of series have noise, 30% are noise-free)
        if self.rng.random() < self.noise_probability:
            noise_level = self.rng.uniform(self.noise_level_range[0], self.noise_level_range[1])
            noise_std = noise_level * total_amplitude  # Noise proportional to total amplitude
            noise = self.rng.normal(0, noise_std, size=self.length)
            values += noise

        return values
