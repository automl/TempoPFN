from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.data.frequency import Frequency


@dataclass
class GeneratorParams:
    """Base class for generator parameters."""

    global_seed: int = 42
    length: int = 2048
    frequency: list[Frequency] | None = None
    start: list[np.datetime64] | None = None

    def update(self, **kwargs):
        """Update parameters from keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class ForecastPFNGeneratorParams(GeneratorParams):
    """Parameters for the ForecastPFNGenerator."""

    trend_exp: bool = True
    scale_noise: tuple[float, float] = (0.6, 0.3)
    harmonic_scale_ratio: float = 0.5
    harmonic_rate: float = 1.0
    period_factor: float = 1.0
    seasonal_only: bool = False
    trend_additional: bool = True
    transition_ratio: float = 1.0  # Probability of applying transition between two series
    random_walk: bool = False

    # Multivariate augmentation parameters (applied in wrapper)
    mixup_prob: float = 0.1  # Probability of applying mixup augmentation
    mixup_series: int = 4  # Maximum number of series to mix in mixup
    damp_and_spike: bool = False  # Whether to apply damping and spike augmentations
    damping_noise_ratio: float = 0.05  # Ratio of batch to apply damping
    spike_noise_ratio: float = 0.05  # Ratio of batch to apply spike noise
    spike_signal_ratio: float = 0.05  # Probability of applying spike signal replacement
    spike_batch_ratio: float = 0.05  # Ratio of batch for spike signal replacement

    # Univariate augmentation parameters (applied in generator)
    time_warp_prob: float = 0.1  # Probability of applying time warping
    time_warp_strength: float = 0.05  # Strength of time warping effect
    magnitude_scale_prob: float = 0.2  # Probability of applying magnitude scaling
    magnitude_scale_range: tuple[float, float] = (
        0.9,
        1.1,
    )  # Range for magnitude scaling
    damping_prob: float = 0.1  # Probability of applying damping augmentation
    spike_prob: float = 0.15  # Probability of applying spike augmentation
    pure_spike_prob: float = 0.02  # Probability of replacing with pure spike signal

    # Built-in filtering parameters
    max_absolute_spread: float = 300.0  # Maximum allowed spread (max - min) for generated series
    max_absolute_value: float = 300.0
    max_retries: int = 10


@dataclass
class GPGeneratorParams(GeneratorParams):
    """
    Parameters for the Gaussian Process (GP) Prior synthetic data generator.
    """

    max_kernels: int = 6
    likelihood_noise_level: float = 0.1
    noise_level: str = "low"  # Options: ["random", "high", "moderate", "low"]
    use_original_gp: bool = False
    gaussians_periodic: bool = True
    peak_spike_ratio: float = 0.1
    subfreq_ratio: float = 0.2
    periods_per_freq: float = 0.5
    gaussian_sampling_ratio: float = 0.2
    max_period_ratio: float = 0.5
    kernel_periods: tuple[int, ...] = (4, 5, 7, 21, 24, 30, 60, 120)
    kernel_bank: dict[str, float] = field(
        default_factory=lambda: {
            "matern_kernel": 1.5,
            "linear_kernel": 1.0,
            "periodic_kernel": 5.0,
            "polynomial_kernel": 0.0,
            "spectral_mixture_kernel": 0.0,
        }
    )


@dataclass
class KernelGeneratorParams(GeneratorParams):
    """Parameters for the KernelSynthGenerator."""

    max_kernels: int = 5


@dataclass
class SineWaveGeneratorParams(GeneratorParams):
    """Parameters for the SineWaveGenerator - focused on diverse sinusoidal patterns."""

    # Core sinusoidal parameters
    num_components_range: tuple[int, int] = (1, 3)
    period_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (10.0, 200.0)
    amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.5, 3.0)
    phase_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.0, 2.0 * np.pi)

    # Trend parameters
    trend_slope_range: tuple[float, float] = (-0.01, 0.01)
    base_level_range: tuple[float, float] = (0.0, 2.0)

    # Noise parameters
    noise_probability: float = 0.7  # Probability of adding noise (70% of series have noise)
    noise_level_range: tuple[float, float] = (
        0.05,
        0.2,
    )  # Small noise as fraction of amplitude (when noise is applied)

    # Time-varying parameters (subtle modulation)
    enable_amplitude_modulation: bool = True
    amplitude_modulation_strength: float = 0.1  # Max 10% amplitude variation
    enable_frequency_modulation: bool = True
    frequency_modulation_strength: float = 0.05  # Max 5% frequency variation


@dataclass
class SawToothGeneratorParams(GeneratorParams):
    """Parameters for the SawToothGenerator."""

    periods: tuple[int, int] = (2, 7)  # Number of sawtooth periods in the series
    amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (0.5, 3.0)
    phase_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        0.0,
        1.0,
    )  # Phase shift as fraction of period
    trend_slope_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        -0.001,
        0.001,
    )  # Slightly stronger linear trend slope for more straight lines
    seasonality_amplitude_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        0.0,
        0.02,
    )  # Minimal seasonal component amplitude
    add_trend: bool = True  # Whether to add linear trend
    add_seasonality: bool = True  # Whether to add seasonal component


class StepPatternType(Enum):
    """Types of step patterns that can be generated."""

    STABLE = "stable"  # Flat line with minimal variation
    GRADUAL_INCREASE = "gradual_increase"  # Gradual upward steps
    GRADUAL_DECREASE = "gradual_decrease"  # Gradual downward steps
    SPIKE_UP = "spike_up"  # Sharp increase then gradual decrease
    SPIKE_DOWN = "spike_down"  # Sharp decrease then gradual increase
    OSCILLATING = "oscillating"  # Up and down pattern
    RANDOM_WALK = "random_walk"  # Random steps (current behavior)


@dataclass
class SubseriesConfig:
    """Configuration for a single subseries pattern."""

    pattern_type: StepPatternType
    length_range: tuple[int, int]  # Min and max length for this subseries
    num_changepoints_range: tuple[int, int]  # Number of changepoints in this subseries
    step_size_range: tuple[float, float]  # Step size range for this pattern
    level_drift_range: tuple[float, float] = (0.0, 0.0)  # Overall level drift
    step_size_decay: float = 1.0  # Decay factor for step sizes over time
    weight: float = 1.0  # Probability weight for selecting this pattern


@dataclass
class StepGeneratorParams(GeneratorParams):
    """Parameters for the StepGenerator with subseries support."""

    # Subseries configuration
    subseries_configs: list[SubseriesConfig] = field(
        default_factory=lambda: [
            # Stable beginning (20-30% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.STABLE,
                length_range=(200, 600),
                num_changepoints_range=(0, 3),
                step_size_range=(-1.0, 1.0),
                weight=0.8,
            ),
            # Gradual increase pattern (15-25% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.GRADUAL_INCREASE,
                length_range=(300, 700),
                num_changepoints_range=(5, 15),
                step_size_range=(1.0, 5.0),
                level_drift_range=(0.0, 0.1),
                weight=0.6,
            ),
            # Gradual decrease pattern (15-25% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.GRADUAL_DECREASE,
                length_range=(300, 700),
                num_changepoints_range=(5, 15),
                step_size_range=(-5.0, -1.0),
                level_drift_range=(-0.1, 0.0),
                weight=0.6,
            ),
            # Spike up pattern (10-20% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.SPIKE_UP,
                length_range=(200, 500),
                num_changepoints_range=(3, 8),
                step_size_range=(3.0, 10.0),
                step_size_decay=0.7,
                weight=0.4,
            ),
            # Spike down pattern (10-20% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.SPIKE_DOWN,
                length_range=(200, 500),
                num_changepoints_range=(3, 8),
                step_size_range=(-10.0, -3.0),
                step_size_decay=0.7,
                weight=0.4,
            ),
            # Oscillating pattern (10-15% of series)
            SubseriesConfig(
                pattern_type=StepPatternType.OSCILLATING,
                length_range=(400, 800),
                num_changepoints_range=(8, 20),
                step_size_range=(-4.0, 4.0),
                weight=0.3,
            ),
            # Random walk pattern (fallback)
            SubseriesConfig(
                pattern_type=StepPatternType.RANDOM_WALK,
                length_range=(100, 400),
                num_changepoints_range=(5, 20),
                step_size_range=(-3.0, 3.0),
                weight=0.2,
            ),
        ]
    )

    # Minimum number of subseries to combine
    min_subseries: int = 10
    max_subseries: int = 100

    # Transition smoothing between subseries
    enable_smooth_transitions: bool = False
    transition_length: int = 5

    # Base level and global parameters
    base_level_range: tuple[float, float] = (5.0, 15.0)
    noise_level_range: tuple[float, float] = (0.001, 0.01)

    # Seasonal component parameters
    add_seasonality: bool = True
    daily_seasonality_amplitude_range: tuple[float, float] = (0.0, 0.8)
    weekly_seasonality_amplitude_range: tuple[float, float] = (0.0, 0.7)

    # Trend parameters
    add_trend: bool = False
    trend_slope_range: tuple[float, float] = (-0.005, 0.005)

    # Scaling parameters
    scale_range: tuple[float, float] = (0.1, 10.0)

    # Anomaly injection parameters
    inject_anomalies: bool = False
    anomaly_probability: float = 0.02
    anomaly_magnitude_range: tuple[float, float] = (2.0, 5.0)

    # Level continuity between subseries
    maintain_level_continuity: bool = True
    max_level_jump_between_subseries: float = 5.0


class AnomalyType(Enum):
    """Types of anomalies that can be generated."""

    SPIKE_UP = "spike_up"
    SPIKE_DOWN = "spike_down"


class MagnitudePattern(Enum):
    """Spike magnitude patterns."""

    CONSTANT = "constant"  # All spikes have similar magnitude
    INCREASING = "increasing"  # Magnitude increases over time
    DECREASING = "decreasing"  # Magnitude decreases over time
    CYCLICAL = "cyclical"  # Magnitude follows a cyclical pattern
    RANDOM_BOUNDED = "random_bounded"  # Random within bounds but with some correlation


@dataclass
class AnomalyGeneratorParams(GeneratorParams):
    """Parameters for anomaly time series generation."""

    # Base signal parameters
    base_level_range: tuple[float, float] = (-100.0, 100.0)

    # Spike direction (50% up-only, 50% down-only series)
    spike_direction_probability: float = 0.5  # Probability of up-only vs down-only series

    # Periodicity parameters (uniform singles are always generated; variance/jitter ignored for base schedule)
    base_period_range: tuple[int, int] = (100, 300)  # Base period between spike events
    period_variance: float = 0.0  # Not used for base schedule anymore

    # Series-level behavior probabilities
    cluster_series_probability: float = 0.25  # 25% of series add clusters near base spikes
    random_series_probability: float = 0.25  # 25% of series add random single spikes

    # Cluster augmentation parameters (relative to base uniform spikes)
    # Fraction of base spike events that will receive nearby extra spikes
    cluster_event_fraction: float = 0.3
    # Number of additional spikes to add per selected event (upper bound exclusive like np.random.randint)
    cluster_additional_spikes_range: tuple[int, int] = (1, 4)  # yields 1..3
    # Offset window (in time steps) around the base spike for additional spikes (inclusive of negatives)
    cluster_offset_range: tuple[int, int] = (-10, 11)  # yields [-10..10]

    # Random single spikes augmentation across the series (not tied to base events)
    # Number of random spikes as a fraction of the number of base spikes
    random_spike_fraction_of_base: float = 0.3

    # Spike magnitude parameters
    magnitude_pattern: MagnitudePattern = MagnitudePattern.RANDOM_BOUNDED
    base_magnitude_range: tuple[float, float] = (10.0, 50.0)
    magnitude_correlation: float = 0.7  # Correlation between consecutive spike magnitudes (0-1)
    magnitude_trend_strength: float = 0.02  # Strength of increasing/decreasing trend
    cyclical_period_ratio: float = 0.3  # Ratio of cyclical period to series length

    # Noise parameters
    magnitude_noise: float = 0.1  # Random noise added to magnitude (as fraction of base magnitude)
    timing_jitter: float = 0.0  # Not used for base schedule anymore

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not (0 <= self.spike_direction_probability <= 1):
            raise ValueError("spike_direction_probability must be between 0 and 1")
        if not (0 <= self.period_variance <= 0.5):
            raise ValueError("period_variance must be between 0 and 0.5")
        if not (0 <= self.magnitude_correlation <= 1):
            raise ValueError("magnitude_correlation must be between 0 and 1")
        if self.base_period_range[0] >= self.base_period_range[1]:
            raise ValueError("base_period_range must have min < max")
        # Validate series-type probabilities
        if not (0.0 <= self.cluster_series_probability <= 1.0):
            raise ValueError("cluster_series_probability must be between 0 and 1")
        if not (0.0 <= self.random_series_probability <= 1.0):
            raise ValueError("random_series_probability must be between 0 and 1")
        if self.cluster_series_probability + self.random_series_probability > 1.0:
            raise ValueError("Sum of cluster_series_probability and random_series_probability must be <= 1")
        # Validate cluster augmentation
        if not (0.0 <= self.cluster_event_fraction <= 1.0):
            raise ValueError("cluster_event_fraction must be between 0 and 1")
        if self.cluster_additional_spikes_range[0] >= self.cluster_additional_spikes_range[1]:
            raise ValueError("cluster_additional_spikes_range must have min < max")
        if self.cluster_offset_range[0] >= self.cluster_offset_range[1]:
            raise ValueError("cluster_offset_range must have min < max")
        # Validate random augmentation
        if not (0.0 <= self.random_spike_fraction_of_base <= 1.0):
            raise ValueError("random_spike_fraction_of_base must be between 0 and 1")


class SpikeShape(Enum):
    """Enumeration of spike shapes."""

    V_SHAPE = "v"
    INVERTED_V = "inverted_v"
    CHOPPED_V = "chopped_v"
    CHOPPED_INVERTED_V = "chopped_inverted_v"


@dataclass
class SpikesGeneratorParams(GeneratorParams):
    """Parameters for spike time series generation."""

    # Separate spike counts for different modes
    spike_count_burst: tuple[int, int] = (2, 4)
    spike_count_uniform: tuple[int, int] = (4, 7)

    # Spike amplitude parameters (absolute values, sign determined per series)
    spike_amplitude: float | tuple[float, float] = (50.0, 300.0)

    # Spike angle range in degrees (controls steepness) - sampled once per series
    spike_angle_range: tuple[float, float] = (70.0, 85.0)

    # Probability of burst mode vs spread mode (5% burst, 95% spread)
    burst_mode_probability: float = 0.05

    # Plateau duration for chopped spikes (in time steps)
    plateau_duration: tuple[int, int] = (30, 50)

    # Baseline value (should be close to zero)
    baseline: float | tuple[float, float] = (-200, 200)

    # Burst clustering parameters - fraction of series length for burst width
    burst_width_fraction: tuple[float, float] = (0.1, 0.25)

    # Spread mode edge margin ratio: edges are set to this fraction of the
    # inter-spike spacing. Smaller values yield smaller left/right margins and
    # larger spacing between spikes. Example: 0.2 => edge margins are 20% of
    # the spacing between spikes.
    edge_margin_ratio: float = 0.2

    # Probability of spikes being above baseline (vs below baseline) per series
    spikes_above_baseline_probability: float = 0.5

    # Probability of each series type
    series_type_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "v_only": 0.4,
            "chopped_only": 0.3,
            "mixed": 0.3,
        }
    )

    # Minimum spike width in time steps (to ensure visible spikes)
    min_spike_width: int = 30

    # Maximum spike width in time steps (to prevent overly wide spikes)
    max_spike_width: int = 100

    # Minimum margin between spikes (only used in burst mode)
    min_spike_margin: int = 10

    # Noise parameters - applied to entire signal
    noise_std: float = 2
    noise_probability: float = 0.5
    brown_noise_alpha: float = 2.0  # Power law exponent (2.0 = brown noise)
    noise_cutoff_freq: float = 0.1  # Relative to Nyquist frequency


@dataclass
class CauKerGeneratorParams(GeneratorParams):
    """Parameters for the CauKer (SCM-GP) generator."""

    # Number of channels (features) to sample per series. If a tuple(range)
    # or list is provided, the wrapper will pick a single value for the whole batch.
    num_channels: int | tuple[int, int] | list[int] = 6

    # Maximum number of parents per node in the DAG
    max_parents: int = 3

    # Total number of nodes in the underlying DAG
    num_nodes: int = 6


class TrendType(Enum):
    """Types of trends that can be applied to the OU process."""

    NONE = "none"  # No trend, classic OU behavior
    LINEAR = "linear"  # Linear drift in mu over time
    EXPONENTIAL = "exponential"  # Exponential growth/decay in mu
    LOGISTIC = "logistic"  # S-curve growth pattern
    SINUSOIDAL = "sinusoidal"  # Cyclical trend
    PIECEWISE_LINEAR = "piecewise_linear"  # Multiple linear segments
    POLYNOMIAL = "polynomial"  # Polynomial trend (quadratic/cubic)


@dataclass
class TrendConfig:
    """Configuration for time-varying trends in OU process parameters."""

    trend_type: TrendType = TrendType.NONE

    # Linear trend parameters
    linear_slope_range: tuple[float, float] = (-0.01, 0.01)

    # Exponential trend parameters
    exp_rate_range: tuple[float, float] = (-0.005, 0.005)
    exp_asymptote_range: tuple[float, float] = (-5.0, 5.0)

    # Logistic trend parameters
    logistic_growth_rate_range: tuple[float, float] = (0.01, 0.1)
    logistic_capacity_range: tuple[float, float] = (5.0, 20.0)
    logistic_midpoint_ratio_range: tuple[float, float] = (
        0.3,
        0.7,
    )  # As fraction of series length

    # Sinusoidal trend parameters
    sin_amplitude_range: tuple[float, float] = (1.0, 5.0)
    sin_period_ratio_range: tuple[float, float] = (
        0.1,
        0.5,
    )  # As fraction of series length
    sin_phase_range: tuple[float, float] = (0.0, 2.0 * np.pi)

    # Piecewise linear parameters
    num_segments_range: tuple[int, int] = (2, 5)
    segment_slope_range: tuple[float, float] = (-0.02, 0.02)

    # Polynomial trend parameters
    poly_degree_range: tuple[int, int] = (2, 3)
    poly_coeff_range: tuple[float, float] = (
        -1e-6,
        1e-6,
    )  # Small coefficients for stability

    # Structural change parameters
    enable_structural_changes: bool = True
    num_structural_changes_range: tuple[int, int] = (0, 3)
    structural_change_magnitude_range: tuple[float, float] = (1.0, 5.0)
    min_segment_length: int = 200  # Minimum length between structural changes


@dataclass
class OrnsteinUhlenbeckProcessGeneratorParams(GeneratorParams):
    """Parameters for the Regime-Switching Ornstein-Uhlenbeck generator.

    The generator samples concrete values per series using these ranges.
    Enhanced with time-varying parameter support for realistic non-stationary behavior.
    """

    # Integration step size used inside the generator
    dt: float = 0.01

    # Regime 0 parameter distributions
    regime0_theta_range: tuple[float, float] = (1.0, 5.0)
    regime0_mu_mean_std: tuple[float, float] = (-2.0, 1.0)
    regime0_sigma_lognormal_params: tuple[float, float] = (float(np.log(0.3)), 0.3)

    # Regime 0 volatility process parameters
    regime0_vol_reversion_range: tuple[float, float] = (2.0, 5.0)  # kappa_v
    regime0_vol_mean_range: tuple[float, float] = (0.2, 0.4)  # theta_v
    regime0_vol_vol_range: tuple[float, float] = (0.1, 0.3)  # xi_v

    # Regime 1 parameter distributions
    regime1_theta_range: tuple[float, float] = (0.05, 0.5)
    regime1_mu_mean_std: tuple[float, float] = (2.0, 1.0)
    regime1_sigma_lognormal_params: tuple[float, float] = (float(np.log(1.5)), 0.5)

    # Regime 1 volatility process parameters
    regime1_vol_reversion_range: tuple[float, float] = (0.5, 2.0)  # kappa_v
    regime1_vol_mean_range: tuple[float, float] = (0.8, 1.2)  # theta_v
    regime1_vol_vol_range: tuple[float, float] = (0.3, 0.5)  # xi_v

    # Initial value distributions
    x0_mean_std: tuple[float, float] = (0.0, 2.0)

    # Transition matrix diagonal probabilities (allow more frequent regime changes)
    p00_range: tuple[float, float] = (0.85, 0.999)  # Allow more frequent transitions
    p11_range: tuple[float, float] = (0.85, 0.999)

    # Time-varying parameter support
    trend_config: TrendConfig = field(default_factory=TrendConfig)

    # Probability of applying trends to different parameters
    mu_trend_probability: float = 0.7  # High probability for realistic non-stationarity
    theta_trend_probability: float = 0.2  # Occasional changes in mean reversion speed
    sigma_trend_probability: float = 0.3  # Occasional changes in volatility

    # Global scaling and level parameters for real-world applicability
    global_level_range: tuple[float, float] = (
        -100.0,
        100.0,
    )  # Base level around which process evolves
    global_scale_range: tuple[float, float] = (
        0.1,
        50.0,
    )  # Scale factor for entire series

    # Noise injection for additional realism
    measurement_noise_std_range: tuple[float, float] = (
        0.0,
        0.1,
    )  # Additive measurement noise

    # Long-term memory parameters (for more realistic autocorrelation)
    enable_long_memory: bool = False
    hurst_exponent_range: tuple[float, float] = (
        0.3,
        0.8,
    )  # Fractional Brownian motion component

    # Seasonality parameters
    enable_seasonality: bool = True
    num_seasonal_components_range: tuple[int, int] = (
        1,
        3,
    )  # Number of seasonal components
    seasonal_periods: tuple[float, ...] = (
        7.0,  # Weekly
        30.0,  # Monthly
        90.0,  # Quarterly
        365.25,  # Yearly
        182.625,  # Semi-annual
    )  # Available seasonal periods (in time units)
    seasonal_amplitude_range: tuple[float, float] = (
        0.5,
        3.0,
    )  # Amplitude of seasonal components
    seasonal_phase_range: tuple[float, float] = (0.0, 2.0 * np.pi)  # Phase shift range
    seasonal_period_jitter: float = 0.05  # Jitter applied to periods for realism (±5%)

    # Probability of applying seasonality to different parameters
    mu_seasonality_probability: float = 0.6  # Probability of seasonal mean
    sigma_seasonality_probability: float = 0.3  # Probability of seasonal volatility

    # Seasonal component decay/growth over time
    enable_seasonal_evolution: bool = True
    seasonal_amplitude_trend_range: tuple[float, float] = (
        -0.001,
        0.001,
    )  # Trend in seasonal amplitude

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive for OU process simulation")

        if not (0.0 <= self.mu_trend_probability <= 1.0):
            raise ValueError("mu_trend_probability must be between 0 and 1")
        if not (0.0 <= self.theta_trend_probability <= 1.0):
            raise ValueError("theta_trend_probability must be between 0 and 1")
        if not (0.0 <= self.sigma_trend_probability <= 1.0):
            raise ValueError("sigma_trend_probability must be between 0 and 1")

        if self.global_level_range[0] >= self.global_level_range[1]:
            raise ValueError("global_level_range must have min < max")
        if self.global_scale_range[0] <= 0:
            raise ValueError("global_scale_range values must be positive")


# =====================
# Audio generator params
# =====================


@dataclass
class AudioGeneratorParams(GeneratorParams):
    """Common parameters for audio-based time series generators (pyo-backed)."""

    # Offline pyo rendering configuration
    server_duration: float = 2.0  # seconds
    sample_rate: int = 44100  # Hz

    # Output post-processing
    normalize_output: bool = True  # Normalize to unit max abs before returning


@dataclass
class FinancialVolatilityAudioParams(AudioGeneratorParams):
    """Parameters for the FinancialVolatility audio generator."""

    # Trend LFO controlling slow drift
    trend_lfo_freq_range: tuple[float, float] = (0.1, 0.5)
    trend_lfo_mul_range: tuple[float, float] = (0.2, 0.5)

    # Volatility clustering
    volatility_carrier_freq_range: tuple[float, float] = (1.0, 5.0)
    follower_freq_range: tuple[float, float] = (1.0, 4.0)
    volatility_range: tuple[float, float] = (0.1, 0.8)

    # Market jumps/shocks
    jump_metro_time_range: tuple[float, float] = (0.3, 1.0)
    jump_env_start_range: tuple[float, float] = (0.5, 1.0)
    jump_env_decay_time_range: tuple[float, float] = (0.05, 0.2)
    jump_freq_range: tuple[float, float] = (20.0, 80.0)
    jump_direction_up_probability: float = 0.5


@dataclass
class MultiScaleFractalAudioParams(AudioGeneratorParams):
    """Parameters for the Multi-Scale Fractal audio generator."""

    base_noise_mul_range: tuple[float, float] = (0.3, 0.8)
    num_scales_range: tuple[int, int] = (3, 6)
    scale_freq_base_range: tuple[float, float] = (20.0, 2000.0)
    q_factor_range: tuple[float, float] = (0.5, 3.0)
    per_scale_attenuation_range: tuple[float, float] = (
        0.5,
        0.8,
    )  # multiplier per scale index


@dataclass
class StochasticRhythmAudioParams(AudioGeneratorParams):
    """Parameters for the Stochastic Rhythm audio generator."""

    base_tempo_hz_range: tuple[float, float] = (2.0, 8.0)
    num_layers_range: tuple[int, int] = (3, 5)
    subdivisions: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    attack_range: tuple[float, float] = (0.001, 0.01)
    decay_range: tuple[float, float] = (0.05, 0.3)
    tone_freq_range: tuple[float, float] = (50.0, 800.0)
    tone_mul_range: tuple[float, float] = (0.2, 0.5)


@dataclass
class NetworkTopologyAudioParams(AudioGeneratorParams):
    """Parameters for the Network Topology audio generator."""

    # Base traffic flow
    traffic_lfo_freq_range: tuple[float, float] = (0.2, 1.0)
    traffic_lfo_mul_range: tuple[float, float] = (0.2, 0.5)

    # Packet bursts
    burst_rate_hz_range: tuple[float, float] = (3.0, 12.0)
    burst_duration_range: tuple[float, float] = (0.02, 0.1)
    burst_mul_range: tuple[float, float] = (0.2, 0.6)

    # Periodic congestion
    congestion_period_range: tuple[float, float] = (1.0, 3.0)  # seconds between events
    congestion_depth_range: tuple[float, float] = (-0.6, -0.2)
    congestion_release_time_range: tuple[float, float] = (0.3, 0.8)

    # Protocol overhead
    overhead_lfo_freq_range: tuple[float, float] = (20.0, 50.0)
    overhead_mul_range: tuple[float, float] = (0.05, 0.15)

    # DDoS-like spikes / attacks
    attack_period_range: tuple[float, float] = (2.0, 5.0)
    attack_env_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 1.2),
        (0.1, 0.8),
        (0.8, 0.0),
    )
    attack_mul_range: tuple[float, float] = (0.4, 0.8)
