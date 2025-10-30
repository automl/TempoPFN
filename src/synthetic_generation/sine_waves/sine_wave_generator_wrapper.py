from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import SineWaveGeneratorParams
from src.synthetic_generation.sine_waves.sine_wave_generator import SineWaveGenerator


class SineWaveGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for SineWaveGenerator to generate batches of multivariate time series data
    by stacking multiple univariate sine wave series. Accepts a SineWaveGeneratorParams
    dataclass for configuration.
    """

    def __init__(self, params: SineWaveGeneratorParams):
        super().__init__(params)
        self.params: SineWaveGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with SineWaveGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                # Core sinusoidal parameters
                "num_components_range": self.params.num_components_range,
                "period_range": self.params.period_range,
                "amplitude_range": self.params.amplitude_range,
                "phase_range": self.params.phase_range,
                # Trend parameters
                "trend_slope_range": self.params.trend_slope_range,
                "base_level_range": self.params.base_level_range,
                # Noise parameters
                "noise_probability": self.params.noise_probability,
                "noise_level_range": self.params.noise_level_range,
                # Time-varying parameters (subtle modulation)
                "enable_amplitude_modulation": self.params.enable_amplitude_modulation,
                "amplitude_modulation_strength": self.params.amplitude_modulation_strength,
                "enable_frequency_modulation": self.params.enable_frequency_modulation,
                "frequency_modulation_strength": self.params.frequency_modulation_strength,
            }
        )
        return params

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using SineWaveGenerator.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed for this batch (default: None).
        params : Dict[str, Any], optional
            Pre-sampled parameters to use. If None, parameters will be sampled.

        Returns
        -------
        BatchTimeSeriesContainer
            A container with the generated time series data.
        """
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters(batch_size)

        generator = SineWaveGenerator(
            length=params["length"],
            # Core sinusoidal parameters
            num_components_range=params["num_components_range"],
            period_range=params["period_range"],
            amplitude_range=params["amplitude_range"],
            phase_range=params["phase_range"],
            # Trend parameters
            trend_slope_range=params["trend_slope_range"],
            base_level_range=params["base_level_range"],
            # Noise parameters
            noise_probability=params["noise_probability"],
            noise_level_range=params["noise_level_range"],
            # Time-varying parameters (subtle modulation)
            enable_amplitude_modulation=params["enable_amplitude_modulation"],
            amplitude_modulation_strength=params["amplitude_modulation_strength"],
            enable_frequency_modulation=params["enable_frequency_modulation"],
            frequency_modulation_strength=params["frequency_modulation_strength"],
            random_seed=seed,
        )

        batch_values = []
        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i
            values = generator.generate_time_series(random_seed=batch_seed)
            batch_values.append(values)

        return TimeSeriesContainer(
            values=np.array(batch_values),
            start=params["start"],
            frequency=params["frequency"],
        )
