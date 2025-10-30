from typing import Optional

import numpy as np

from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import StepGeneratorParams
from src.synthetic_generation.steps.step_generator import StepGenerator


class StepGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for StepGenerator that handles batch generation and formatting.
    """

    def __init__(self, params: StepGeneratorParams):
        """
        Initialize the StepGeneratorWrapper.

        Parameters
        ----------
        params : StepGeneratorParams
            Parameters for the step generator.
        """
        super().__init__(params)
        self.generator = StepGenerator(params)

    def generate_batch(
        self, batch_size: int, seed: Optional[int] = None
    ) -> TimeSeriesContainer:
        """
        Generate a batch of step function time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        TimeSeriesContainer
            TimeSeriesContainer containing the generated time series.
        """
        if seed is not None:
            self._set_random_seeds(seed)

        # Sample parameters for the batch
        sampled_params = self._sample_parameters(batch_size)

        # Generate time series
        values = []
        for i in range(batch_size):
            # Use a different seed for each series in the batch
            series_seed = (seed + i) if seed is not None else None
            series = self.generator.generate_time_series(series_seed)
            values.append(series)

        return TimeSeriesContainer(
            values=np.array(values),
            start=sampled_params["start"],
            frequency=sampled_params["frequency"],
        )
