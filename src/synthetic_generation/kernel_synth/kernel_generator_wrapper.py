from typing import Any, Dict, Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import KernelGeneratorParams
from src.synthetic_generation.kernel_synth.kernel_synth import KernelSynthGenerator


class KernelGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for KernelSynthGenerator to generate batches of multivariate time series data
    by stacking multiple univariate series. Accepts a KernelGeneratorParams dataclass for configuration.
    """

    def __init__(self, params: KernelGeneratorParams):
        super().__init__(params)
        self.params: KernelGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with KernelSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters(batch_size)

        params.update(
            {
                "length": self.params.length,
                "max_kernels": self.params.max_kernels,
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
        Generate a batch of synthetic multivariate time series using KernelSynthGenerator.

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

        generator = KernelSynthGenerator(
            length=params["length"],
            max_kernels=params["max_kernels"],
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
