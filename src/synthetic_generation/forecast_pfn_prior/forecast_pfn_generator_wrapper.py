import logging
from typing import Any

import numpy as np
from src.data.containers import TimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator import (
    ForecastPFNGenerator,
)
from src.synthetic_generation.generator_params import ForecastPFNGeneratorParams


class ForecastPFNGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: ForecastPFNGeneratorParams):
        super().__init__(params)
        self.params: ForecastPFNGeneratorParams = params

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        """
        Sample parameters for generating a batch of time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate parameters for

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameters including:
            - frequency: List of frequencies (one per batch item)
            - start: List of start dates (one per batch item)
            - length: Series length
            - All ForecastPFN-specific parameters
        """
        params = super()._sample_parameters(batch_size)

        params.update(
            {
                "length": self.params.length,
                "trend_exp": self.params.trend_exp,
                "scale_noise": self.params.scale_noise,
                "harmonic_scale_ratio": self.params.harmonic_scale_ratio,
                "harmonic_rate": self.params.harmonic_rate,
                "period_factor": self.params.period_factor,
                "seasonal_only": self.params.seasonal_only,
                "trend_additional": self.params.trend_additional,
                "transition_ratio": self.params.transition_ratio,
                "random_walk": self.params.random_walk,
                # Univariate augmentation parameters
                "time_warp_prob": self.params.time_warp_prob,
                "time_warp_strength": self.params.time_warp_strength,
                "magnitude_scale_prob": self.params.magnitude_scale_prob,
                "magnitude_scale_range": self.params.magnitude_scale_range,
                "damping_prob": self.params.damping_prob,
                "spike_prob": self.params.spike_prob,
                "pure_spike_prob": self.params.pure_spike_prob,
            }
        )
        return params

    def _apply_augmentations(self, batch_values: np.ndarray, mixup_prob: float, mixup_series: int) -> np.ndarray:
        """
        Apply multivariate augmentations to the batch.

        Parameters
        ----------
        batch_values : np.ndarray
            Batch of time series values with shape (batch_size, length)
        mixup_prob : float
            Probability of applying mixup augmentation
        mixup_series : int
            Maximum number of series to mix in mixup

        Returns
        -------
        np.ndarray
            Augmented batch values with same shape as input
        """
        batch_size = batch_values.shape[0]

        # Apply mixup augmentation if enabled
        if self.rng.random() < mixup_prob:
            mixup_series = self.rng.integers(2, mixup_series + 1)
            mixup_indices = self.rng.choice(batch_size, mixup_series, replace=False)
            original_vals = batch_values[mixup_indices, :].copy()
            for _, idx in enumerate(mixup_indices):
                mixup_weights = self.rng.random(mixup_series)
                mixup_weights /= np.sum(mixup_weights)
                batch_values[idx, :] = np.sum(original_vals * mixup_weights[:, np.newaxis], axis=0)

        return batch_values

    def generate_batch(
        self,
        batch_size: int,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> TimeSeriesContainer:
        """
        Generate a batch of time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate
        seed : Optional[int], default=None
            Random seed for reproducibility
        params : Optional[Dict[str, Any]], default=None
            Generation parameters. If None, will be sampled automatically

        Returns
        -------
        TimeSeriesContainer
            Container with generated time series values, start dates, and frequencies
        """
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters(batch_size)

        generator = ForecastPFNGenerator(
            params=ForecastPFNGeneratorParams(**params),
            length=params["length"],
            random_seed=seed,
            max_retries=100,
        )

        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i
            # Extract individual parameters for this batch item
            frequency_i = params["frequency"][i] if isinstance(params["frequency"], list) else params["frequency"]
            start_i = params["start"][i] if isinstance(params["start"], list) else params["start"]

            try:
                values = generator.generate_time_series(
                    start=start_i,
                    random_seed=batch_seed,
                    apply_augmentations=True,
                    frequency=frequency_i,
                )
                batch_values.append(values)
            except RuntimeError as e:
                # Log the failure and generate a fallback series
                logging.warning(f"Failed to generate series {i} in batch: {e}")

        # Convert to numpy array before applying augmentations
        batch_values = np.array(batch_values)

        # Apply batch augmentations
        batch_values = self._apply_augmentations(
            batch_values=batch_values,
            mixup_prob=self.params.mixup_prob,
            mixup_series=self.params.mixup_series,
        )

        return TimeSeriesContainer(
            values=batch_values,
            start=params["start"],
            frequency=params["frequency"],
        )
