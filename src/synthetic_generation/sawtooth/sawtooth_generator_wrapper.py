from typing import Any

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import SawToothGeneratorParams
from src.synthetic_generation.sawtooth.sawtooth_generator import SawToothGenerator


class SawToothGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for SawToothGenerator to generate batches of multivariate time series data
    by stacking multiple univariate sawtooth wave series. Accepts a SawToothGeneratorParams
    dataclass for configuration.
    """

    def __init__(self, params: SawToothGeneratorParams):
        super().__init__(params)
        self.params: SawToothGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        """
        Sample parameter values for batch generation with SawToothGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters(batch_size)
        params.update(
            {
                "length": self.params.length,
                "periods": self.params.periods,
                "amplitude_range": self.params.amplitude_range,
                "phase_range": self.params.phase_range,
                "trend_slope_range": self.params.trend_slope_range,
                "seasonality_amplitude_range": self.params.seasonality_amplitude_range,
                "add_trend": self.params.add_trend,
                "add_seasonality": self.params.add_seasonality,
            }
        )
        return params

    def generate_batch(
        self,
        batch_size: int,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> TimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using SawToothGenerator.

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
        TimeSeriesContainer
            A container with the generated time series data.
        """
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters(batch_size)

        generator = SawToothGenerator(
            length=params["length"],
            periods=params["periods"],
            amplitude_range=params["amplitude_range"],
            phase_range=params["phase_range"],
            trend_slope_range=params["trend_slope_range"],
            seasonality_amplitude_range=params["seasonality_amplitude_range"],
            add_trend=params["add_trend"],
            add_seasonality=params["add_seasonality"],
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
