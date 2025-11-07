from collections.abc import Callable
from dataclasses import replace

import numpy as np
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.generator_params import (
    OrnsteinUhlenbeckProcessGeneratorParams,
    TrendConfig,
    TrendType,
)


class OrnsteinUhlenbeckProcessGenerator(AbstractTimeSeriesGenerator):
    """
    Regime-Switching Ornstein-Uhlenbeck (OU) process generator with time-varying parameters.

    Enhanced to support:
    - Time-varying mu (trends, drifts, structural changes)
    - Time-varying theta and sigma parameters
    - Multiple trend types (linear, exponential, logistic, etc.)
    - Structural breaks and regime-dependent trends
    - Global scaling and level adjustments for real-world applicability
    """

    def __init__(self, params: OrnsteinUhlenbeckProcessGeneratorParams):
        self.params = params
        self.rng = np.random.default_rng(params.global_seed)

    # == Regime switching ==
    def _generate_regime_sequence(self, transition_matrix: np.ndarray, num_steps: int) -> np.ndarray:
        regimes = np.zeros(num_steps, dtype=int)
        regimes[0] = int(self.rng.integers(0, transition_matrix.shape[0]))
        for i in range(1, num_steps):
            current = regimes[i - 1]
            regimes[i] = self.rng.choice(transition_matrix.shape[0], p=transition_matrix[current, :])
        return regimes

    # == Time-varying parameter generation ==
    def _create_trend_function(self, trend_config: TrendConfig, t_values: np.ndarray) -> Callable[[float], float]:
        """Create a trend function based on the specified trend type."""
        trend_type = trend_config.trend_type

        if trend_type == TrendType.NONE:
            return lambda t: 0.0

        elif trend_type == TrendType.LINEAR:
            slope = self.rng.uniform(*trend_config.linear_slope_range)
            return lambda t: slope * t

        elif trend_type == TrendType.EXPONENTIAL:
            rate = self.rng.uniform(*trend_config.exp_rate_range)
            asymptote = self.rng.uniform(*trend_config.exp_asymptote_range)
            return lambda t: asymptote * (1.0 - np.exp(-rate * t))

        elif trend_type == TrendType.LOGISTIC:
            growth_rate = self.rng.uniform(*trend_config.logistic_growth_rate_range)
            capacity = self.rng.uniform(*trend_config.logistic_capacity_range)
            midpoint_ratio = self.rng.uniform(*trend_config.logistic_midpoint_ratio_range)
            midpoint = midpoint_ratio * t_values[-1]
            return lambda t: capacity / (1.0 + np.exp(-growth_rate * (t - midpoint)))

        elif trend_type == TrendType.SINUSOIDAL:
            amplitude = self.rng.uniform(*trend_config.sin_amplitude_range)
            period_ratio = self.rng.uniform(*trend_config.sin_period_ratio_range)
            period = period_ratio * t_values[-1]
            phase = self.rng.uniform(*trend_config.sin_phase_range)
            return lambda t: amplitude * np.sin(2.0 * np.pi * t / period + phase)

        elif trend_type == TrendType.PIECEWISE_LINEAR:
            return self._create_piecewise_linear_trend(trend_config, t_values)

        elif trend_type == TrendType.POLYNOMIAL:
            degree = self.rng.integers(*trend_config.poly_degree_range)
            coeffs = self.rng.uniform(*trend_config.poly_coeff_range, size=degree + 1)
            return lambda t: sum(coeff * (t**i) for i, coeff in enumerate(coeffs))

        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

    def _create_piecewise_linear_trend(
        self, trend_config: TrendConfig, t_values: np.ndarray
    ) -> Callable[[float], float]:
        """Create a piecewise linear trend function."""
        num_segments = self.rng.integers(*trend_config.num_segments_range)
        total_time = t_values[-1]

        # Create breakpoints
        breakpoints = np.sort(self.rng.uniform(0, total_time, num_segments - 1))
        breakpoints = np.concatenate([[0], breakpoints, [total_time]])

        # Create slopes for each segment
        slopes = self.rng.uniform(*trend_config.segment_slope_range, size=num_segments)

        # Compute y-values at breakpoints to ensure continuity
        y_values = [0.0]  # Start at 0
        for i in range(num_segments):
            segment_length = breakpoints[i + 1] - breakpoints[i]
            y_values.append(y_values[-1] + slopes[i] * segment_length)

        def piecewise_trend(t: float) -> float:
            # Find which segment t belongs to
            segment_idx = np.searchsorted(breakpoints[1:], t)
            segment_idx = min(segment_idx, num_segments - 1)

            # Linear interpolation within the segment
            t_start = breakpoints[segment_idx]
            y_start = y_values[segment_idx]
            slope = slopes[segment_idx]

            return y_start + slope * (t - t_start)

        return piecewise_trend

    def _add_structural_changes(
        self, base_function: Callable[[float], float], t_values: np.ndarray
    ) -> Callable[[float], float]:
        """Add structural changes to a base trend function."""
        if not self.params.trend_config.enable_structural_changes:
            return base_function

        config = self.params.trend_config
        num_changes = self.rng.integers(*config.num_structural_changes_range)

        if num_changes == 0:
            return base_function

        # Generate change points ensuring minimum segment length
        total_time = t_values[-1]
        min_segment = config.min_segment_length * self.params.dt

        if num_changes * min_segment >= total_time:
            # Too many changes requested, reduce number
            num_changes = max(1, int(total_time / min_segment) - 1)

        change_times = np.sort(self.rng.uniform(min_segment, total_time - min_segment, num_changes))
        change_magnitudes = self.rng.uniform(*config.structural_change_magnitude_range, size=num_changes)

        def structural_trend(t: float) -> float:
            base_value = base_function(t)
            structural_adjustment = 0.0

            for change_time, magnitude in zip(change_times, change_magnitudes, strict=True):
                if t >= change_time:
                    # Smooth step function for structural change
                    transition_width = min_segment * 0.1  # 10% of minimum segment for smooth transition
                    if transition_width > 0:
                        smooth_step = 1.0 / (1.0 + np.exp(-10.0 * (t - change_time) / transition_width))
                    else:
                        smooth_step = 1.0 if t >= change_time else 0.0
                    structural_adjustment += magnitude * smooth_step

            return base_value + structural_adjustment

        return structural_trend

    def _sample_trend_type(self) -> TrendType:
        """Sample a trend type based on realistic probabilities."""
        trend_weights = {
            TrendType.NONE: 0.3,
            TrendType.LINEAR: 0.15,
            TrendType.EXPONENTIAL: 0.15,
            TrendType.LOGISTIC: 0.1,
            TrendType.SINUSOIDAL: 0.15,
            TrendType.PIECEWISE_LINEAR: 0.1,
            TrendType.POLYNOMIAL: 0.05,
        }

        trend_types = list(trend_weights.keys())
        weights = list(trend_weights.values())

        return self.rng.choice(trend_types, p=weights)

    # == Seasonality functions ==
    def _sample_seasonal_components(self) -> list:
        """Sample seasonal components inspired by ou.py."""
        if not self.params.enable_seasonality:
            return []

        num_components = self.rng.integers(*self.params.num_seasonal_components_range)
        components = []

        # Sample from available periods
        selected_periods = self.rng.choice(
            self.params.seasonal_periods,
            size=min(num_components, len(self.params.seasonal_periods)),
            replace=False,
        )

        for period in selected_periods:
            # Add jitter to period for realism
            jittered_period = period * (
                1.0
                + self.rng.uniform(
                    -self.params.seasonal_period_jitter,
                    self.params.seasonal_period_jitter,
                )
            )

            # Sample amplitude trend if evolution is enabled
            amplitude_trend = 0.0
            if self.params.enable_seasonal_evolution:
                amplitude_trend = self.rng.uniform(*self.params.seasonal_amplitude_trend_range)

            component = {
                "period": float(jittered_period),
                "amplitude": self.rng.uniform(*self.params.seasonal_amplitude_range),
                "phase": self.rng.uniform(*self.params.seasonal_phase_range),
                "amplitude_trend": amplitude_trend,  # For evolving seasonality
            }
            components.append(component)

        return components

    def _create_seasonal_function(self, components: list) -> Callable[[float], float]:
        """Create a seasonal function from components."""
        if not components:
            return lambda t: 0.0

        def seasonal_func(t: float) -> float:
            seasonal_value = 0.0
            for comp in components:
                # Base amplitude with optional time-varying evolution
                amplitude = comp["amplitude"]
                if comp.get("amplitude_trend", 0.0) != 0.0:
                    amplitude += comp["amplitude_trend"] * t

                seasonal_value += amplitude * np.sin(2.0 * np.pi * t / comp["period"] + comp["phase"])
            return seasonal_value

        return seasonal_func

    def _sample_seasonal_functions(self, regime_params: dict) -> dict[str, dict[str, Callable]]:
        """Create seasonal functions for each regime based on sampled components."""
        seasonal_functions = {"regime_0": {}, "regime_1": {}}

        for regime_key in ["regime_0", "regime_1"]:
            regime_data = regime_params[regime_key]

            # Create seasonal function for mu if components exist
            if "mu_seasonality" in regime_data:
                seasonal_functions[regime_key]["mu"] = self._create_seasonal_function(regime_data["mu_seasonality"])

            # Create seasonal function for sigma if components exist
            if "sigma_seasonality" in regime_data:
                seasonal_functions[regime_key]["sigma"] = self._create_seasonal_function(
                    regime_data["sigma_seasonality"]
                )

        return seasonal_functions

    # == Parameter handling ==
    class _ParameterManager:
        def __init__(
            self,
            params: dict,
            num_steps: int,
            trend_functions: dict[str, Callable] | None = None,
            seasonal_functions: dict[str, Callable] | None = None,
        ):
            self.num_steps = num_steps
            self.params: dict = {}
            self.trend_functions = trend_functions or {}
            self.seasonal_functions = seasonal_functions or {}

            for key, value in params.items():
                # Skip seasonal component lists - handle them separately
                if key.endswith("_seasonality"):
                    self.params[key] = value
                elif isinstance(value, (tuple, list)) and len(value) == 2 and not callable(value):
                    self.params[key] = np.linspace(value[0], value[1], num_steps)
                else:
                    self.params[key] = value

        def get(self, key: str, idx: int, t_value: float):
            value = self.params.get(key)
            if value is None:
                return None

            # Get base parameter value
            base_value = value
            if isinstance(value, np.ndarray):
                base_value = value[idx]
            elif callable(value):
                base_value = value(t_value)

            # Apply trend if available
            if key in self.trend_functions:
                trend_adjustment = self.trend_functions[key](t_value)
                base_value += trend_adjustment

            # Apply seasonality if available
            if key in self.seasonal_functions:
                seasonal_adjustment = self.seasonal_functions[key](t_value)
                base_value += seasonal_adjustment

            return base_value

    def _get_params_from_managers(self, idx: int, t_value: float):
        if self._regime_sequence is not None:
            current_regime = int(self._regime_sequence[idx])
            return self._param_managers[current_regime]
        return self._param_manager

    # == Single-step OU update ==
    def _step_ou(self, x_value: float, t_value: float, idx: int, dt: float, dW_value: float) -> float:
        manager = self._get_params_from_managers(idx, t_value)
        theta = float(manager.get("theta", idx, t_value))
        mu = float(manager.get("mu", idx, t_value))
        sigma = float(manager.get("sigma", idx, t_value))
        return float(x_value + theta * (mu - x_value) * dt + sigma * dW_value)

    # == Sampling primitives ==
    def _sample_regime_parameters(self) -> dict[str, dict[str, float]]:
        """Sample base regime parameters (before applying trends and seasonality)."""
        p = self.params
        regime0 = {
            "theta": self.rng.uniform(p.regime0_theta_range[0], p.regime0_theta_range[1]),
            "mu": self.rng.normal(p.regime0_mu_mean_std[0], p.regime0_mu_mean_std[1]),
            "sigma": float(
                self.rng.lognormal(
                    p.regime0_sigma_lognormal_params[0],
                    p.regime0_sigma_lognormal_params[1],
                )
            ),
            "x0": self.rng.normal(p.x0_mean_std[0], p.x0_mean_std[1]),
        }
        regime1 = {
            "theta": self.rng.uniform(p.regime1_theta_range[0], p.regime1_theta_range[1]),
            "mu": self.rng.normal(p.regime1_mu_mean_std[0], p.regime1_mu_mean_std[1]),
            "sigma": float(
                self.rng.lognormal(
                    p.regime1_sigma_lognormal_params[0],
                    p.regime1_sigma_lognormal_params[1],
                )
            ),
            "x0": self.rng.normal(p.x0_mean_std[0], p.x0_mean_std[1]),
        }

        # Add seasonal components if enabled
        if self.params.enable_seasonality:
            # Sample seasonal components for mu (mean) in each regime
            if self.rng.random() < self.params.mu_seasonality_probability:
                regime0["mu_seasonality"] = self._sample_seasonal_components()
                regime1["mu_seasonality"] = self._sample_seasonal_components()

            # Sample seasonal components for sigma (volatility) in each regime
            if self.rng.random() < self.params.sigma_seasonality_probability:
                regime0["sigma_seasonality"] = self._sample_seasonal_components()
                regime1["sigma_seasonality"] = self._sample_seasonal_components()

        return {"regime_0": regime0, "regime_1": regime1}

    def _sample_trend_functions(self, t_values: np.ndarray) -> dict[str, dict[str, Callable]]:
        """Sample trend functions for each parameter and regime."""
        trend_functions = {"regime_0": {}, "regime_1": {}}

        # Sample trend types for each parameter
        for param in ["mu", "theta", "sigma"]:
            prob_key = f"{param}_trend_probability"
            if hasattr(self.params, prob_key):
                trend_prob = getattr(self.params, prob_key)

                for regime in ["regime_0", "regime_1"]:
                    if self.rng.random() < trend_prob:
                        # Sample trend type and create trend config using the global config as base
                        trend_type = self._sample_trend_type()
                        print(f"Sampling trend type: {trend_type}")
                        trend_config = replace(self.params.trend_config, trend_type=trend_type)

                        # Create trend function
                        base_trend = self._create_trend_function(trend_config, t_values)

                        # Add structural changes if enabled
                        final_trend = self._add_structural_changes(base_trend, t_values)

                        trend_functions[regime][param] = final_trend

        return trend_functions

    def _generate_fractional_brownian_motion(self, num_steps: int, hurst: float, dt: float) -> np.ndarray:
        """Generate fractional Brownian motion for long-term memory effects."""
        if not (0 < hurst < 1):
            raise ValueError("Hurst exponent must be between 0 and 1")

        # Simple approximation using cumulative sum of correlated Gaussian noise
        # For more accurate fBm, consider using more sophisticated methods
        noise = self.rng.normal(0, 1, num_steps)

        # Apply fractional integration (simplified)
        if hurst != 0.5:
            # Create correlation structure
            correlations = np.zeros(num_steps)
            for k in range(num_steps):
                if k == 0:
                    correlations[k] = 1.0
                else:
                    correlations[k] = 0.5 * ((k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst) + (k - 1) ** (2 * hurst))

            # Apply convolution (simplified approach)
            correlated_noise = np.convolve(noise, correlations[: min(100, num_steps)], mode="same")
            return correlated_noise * np.sqrt(dt)

        return noise * np.sqrt(dt)

    def _sample_transition_matrix(self) -> np.ndarray:
        p00 = float(self.rng.uniform(self.params.p00_range[0], self.params.p00_range[1]))
        p11 = float(self.rng.uniform(self.params.p11_range[0], self.params.p11_range[1]))
        transition_matrix = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]], dtype=float)
        return transition_matrix

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """Generate a time series with enhanced realism through time-varying parameters."""
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        num_steps = int(self.params.length)
        dt = float(self.params.dt)
        t_values = np.linspace(0.0, dt * (num_steps - 1), num_steps)

        # Sample base regime parameters
        sampled_regime_params = self._sample_regime_parameters()

        # Sample trend functions for time-varying behavior
        trend_functions = self._sample_trend_functions(t_values)

        # Sample seasonal functions for each regime
        seasonal_functions = self._sample_seasonal_functions(sampled_regime_params)

        # Generate regime switching
        transition_matrix = self._sample_transition_matrix()
        self._regime_sequence = self._generate_regime_sequence(transition_matrix, num_steps)

        # Create parameter managers with trend and seasonal support
        self._param_managers = [
            OrnsteinUhlenbeckProcessGenerator._ParameterManager(
                sampled_regime_params["regime_0"],
                num_steps,
                trend_functions["regime_0"],
                seasonal_functions["regime_0"],
            ),
            OrnsteinUhlenbeckProcessGenerator._ParameterManager(
                sampled_regime_params["regime_1"],
                num_steps,
                trend_functions["regime_1"],
                seasonal_functions["regime_1"],
            ),
        ]
        self._param_manager = None

        # Generate driving noise (with optional long-term memory)
        if self.params.enable_long_memory:
            hurst = self.rng.uniform(*self.params.hurst_exponent_range)
            dW = self._generate_fractional_brownian_motion(num_steps - 1, hurst, dt)
        else:
            dW = self.rng.normal(0.0, np.sqrt(dt), size=num_steps - 1)

        # Initialize path
        initial_regime = int(self._regime_sequence[0])
        x0_value = float(self._param_managers[initial_regime].get("x0", 0, 0.0))
        path = np.zeros(num_steps, dtype=float)
        path[0] = x0_value

        # Generate the OU process path
        for idx in range(num_steps - 1):
            path[idx + 1] = self._step_ou(path[idx], t_values[idx], idx, dt, dW[idx])

        # Apply global transformations for real-world applicability
        path = self._apply_global_transformations(path)

        # Add measurement noise if specified
        if self.params.measurement_noise_std_range[1] > 0:
            noise_std = self.rng.uniform(*self.params.measurement_noise_std_range)
            measurement_noise = self.rng.normal(0, noise_std, size=num_steps)
            path += measurement_noise

        return path

    def _apply_global_transformations(self, path: np.ndarray) -> np.ndarray:
        """Apply global level and scale transformations to make series more realistic."""
        # Sample global parameters
        global_level = self.rng.uniform(*self.params.global_level_range)
        global_scale = self.rng.uniform(*self.params.global_scale_range)

        # Apply transformations
        transformed_path = global_level + global_scale * path

        return transformed_path
