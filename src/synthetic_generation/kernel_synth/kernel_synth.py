import functools

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator


class KernelSynthGenerator(AbstractTimeSeriesGenerator):
    """
    Generate independent synthetic univariate time series using kernel synthesis.

    Each series is sampled from a Gaussian process prior with a random composite kernel.
    """

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        random_seed: int | None = None,
    ):
        """
        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 1024).
        max_kernels : int, optional
            Maximum number of base kernels to combine (default: 5).
        random_seed : int, optional
            Seed for the random number generator.
        """
        self.length = length
        self.max_kernels = max_kernels
        self.rng = np.random.default_rng(random_seed)
        self.kernel_bank = [
            ExpSineSquared(periodicity=24 / length),  # H
            ExpSineSquared(periodicity=48 / length),  # 0.5H
            ExpSineSquared(periodicity=96 / length),  # 0.25H
            ExpSineSquared(periodicity=24 * 7 / length),  # H-week
            ExpSineSquared(periodicity=48 * 7 / length),  # 0.5H-week
            ExpSineSquared(periodicity=96 * 7 / length),  # 0.25H-week
            ExpSineSquared(periodicity=7 / length),  # day
            ExpSineSquared(periodicity=14 / length),  # 0.5-day
            ExpSineSquared(periodicity=30 / length),  # day
            ExpSineSquared(periodicity=60 / length),  # 0.5-day
            ExpSineSquared(periodicity=365 / length),  # year
            ExpSineSquared(periodicity=365 * 2 / length),  # 0.5-year
            ExpSineSquared(periodicity=4 / length),  # week
            ExpSineSquared(periodicity=26 / length),  # week
            ExpSineSquared(periodicity=52 / length),  # week
            ExpSineSquared(periodicity=4 / length),  # month
            ExpSineSquared(periodicity=6 / length),  # month
            ExpSineSquared(periodicity=12 / length),  # month
            ExpSineSquared(periodicity=4 / length),  # quarter
            ExpSineSquared(periodicity=4 * 10 / length),  # quarter
            ExpSineSquared(periodicity=10 / length),  # year
            DotProduct(sigma_0=0.0),
            DotProduct(sigma_0=1.0),
            DotProduct(sigma_0=10.0),
            RBF(length_scale=0.1),
            RBF(length_scale=1.0),
            RBF(length_scale=10.0),
            RationalQuadratic(alpha=0.1),
            RationalQuadratic(alpha=1.0),
            RationalQuadratic(alpha=10.0),
            WhiteKernel(noise_level=0.1),
            WhiteKernel(noise_level=1.0),
            ConstantKernel(),
        ]

    def _random_binary_map(self, a: Kernel, b: Kernel) -> Kernel:
        """
        Randomly combine two kernels with + or *.
        """
        ops = [lambda x, y: x + y, lambda x, y: x * y]
        return self.rng.choice(ops)(a, b)

    def _sample_from_gp_prior(
        self,
        kernel: Kernel,
        X: np.ndarray,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Draw a sample from GP prior using GaussianProcessRegressor.
        """
        if X.ndim == 1:
            X = X[:, None]
        gpr = GaussianProcessRegressor(kernel=kernel)
        ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

        return ts.squeeze()

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate a single independent univariate time series.

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

        X = np.linspace(0, 1, self.length)
        num_kernels = self.rng.integers(1, self.max_kernels + 1)
        selected = self.rng.choice(self.kernel_bank, num_kernels, replace=True)
        composite = functools.reduce(self._random_binary_map, selected)
        try:
            values = self._sample_from_gp_prior(composite, X, random_seed=random_seed)
        except np.linalg.LinAlgError:
            new_seed = (random_seed + 1) if random_seed is not None else None
            return self.generate_time_series(new_seed)

        return values
