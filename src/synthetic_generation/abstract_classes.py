from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from src.data.containers import TimeSeriesContainer
from src.data.frequency import (
    select_safe_random_frequency,
    select_safe_start_date,
)
from src.synthetic_generation.generator_params import GeneratorParams


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series generators.
    """

    @abstractmethod
    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate synthetic time series data.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Time series values of shape (length,) for univariate or
            (length, num_channels) for multivariate time series.
        """
        pass


class GeneratorWrapper:
    """
    Unified base class for all generator wrappers, using a GeneratorParams dataclass
    for configuration. Provides parameter sampling, validation, and batch formatting utilities.
    """

    def __init__(self, params: GeneratorParams):
        """
        Initialize the GeneratorWrapper with a GeneratorParams dataclass.

        Parameters
        ----------
        params : GeneratorParams
            Dataclass instance containing all generator configuration parameters.
        """
        self.params = params
        self._set_random_seeds(self.params.global_seed)

    def _set_random_seeds(self, seed: int) -> None:
        # For parameter sampling, we want diversity across batches even with similar seeds
        # Use a hash of the generator class name to ensure different generators get different parameter sequences
        param_seed = seed + hash(self.__class__.__name__) % 2**31
        self.rng = np.random.default_rng(param_seed)

        # Set global numpy and torch seeds for deterministic behavior in underlying generators
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        """
        Sample parameters with total_length fixed and history_length calculated.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values where
            history_length = total_length - future_length.
        """

        # Select a suitable frequency based on the total length
        frequency = [select_safe_random_frequency(self.params.length, self.rng) for _ in range(batch_size)]
        start = [select_safe_start_date(self.params.length, frequency[i], self.rng) for i in range(batch_size)]

        return {
            "frequency": frequency,
            "start": start,
        }

    @abstractmethod
    def generate_batch(self, batch_size: int, seed: int | None = None, **kwargs) -> TimeSeriesContainer:
        raise NotImplementedError("Subclasses must implement generate_batch()")
