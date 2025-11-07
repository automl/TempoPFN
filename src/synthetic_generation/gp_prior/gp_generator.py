import functools

import gpytorch
import numpy as np
import torch
from src.data.frequency import FREQUENCY_MAPPING
from src.synthetic_generation.generator_params import GPGeneratorParams
from src.synthetic_generation.gp_prior.constants import (
    KERNEL_BANK,
    KERNEL_PERIODS_BY_FREQ,
)
from src.synthetic_generation.gp_prior.utils import (
    create_kernel,
    extract_periodicities,
    random_binary_map,
)
from src.synthetic_generation.utils import generate_peak_spikes


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGenerator:
    def __init__(
        self,
        params: GPGeneratorParams,
        length: int = 1024,
        random_seed: int | None = None,
    ):
        self.params = params
        self.length = length
        self.rng = np.random.default_rng(random_seed)
        self.frequency = params.frequency
        self.max_kernels = params.max_kernels
        self.likelihood_noise_level = params.likelihood_noise_level
        self.noise_level = params.noise_level
        self.use_original_gp = params.use_original_gp
        self.gaussians_periodic = params.gaussians_periodic
        self.peak_spike_ratio = params.peak_spike_ratio
        self.subfreq_ratio = params.subfreq_ratio
        self.periods_per_freq = params.periods_per_freq
        self.gaussian_sampling_ratio = params.gaussian_sampling_ratio
        self.kernel_periods = params.kernel_periods
        self.max_period_ratio = params.max_period_ratio
        self.kernel_bank = params.kernel_bank

    def generate_time_series(
        self,
        random_seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        with torch.inference_mode():
            if random_seed is not None:
                self.rng = np.random.default_rng(random_seed)
                torch.manual_seed(random_seed)

            # Determine kernel_bank and gaussians_periodic
            if self.use_original_gp:
                kernel_bank = KERNEL_BANK
                gaussians_periodic = False
            else:
                # Convert kernel_bank from {str: float} format to {int: (str, float)} format
                kernel_bank = {
                    i: (kernel_name, weight) for i, (kernel_name, weight) in enumerate(self.kernel_bank.items())
                }
                gaussians_periodic = self.gaussians_periodic

            # Map frequency to freq and subfreq
            freq, subfreq, timescale = FREQUENCY_MAPPING.get(self.frequency, ("D", "", 0))

            # Decide if using exact frequencies
            exact_freqs = self.rng.random() < self.periods_per_freq
            if exact_freqs and freq in KERNEL_PERIODS_BY_FREQ:
                kernel_periods = KERNEL_PERIODS_BY_FREQ[freq]
                if subfreq:
                    subfreq_int = int(subfreq)
                    kernel_periods = [p // subfreq_int for p in kernel_periods if p >= subfreq_int]
            else:
                kernel_periods = self.kernel_periods

            # Sample number of kernels
            num_kernels = self.rng.integers(1, self.max_kernels + 1)
            # Always expect kernel_bank as dict {int: (str, float)}
            kernel_weights = np.array([v[1] for v in kernel_bank.values()])
            kernel_ids = self.rng.choice(
                list(kernel_bank.keys()),
                size=num_kernels,
                p=kernel_weights / kernel_weights.sum(),
            )
            kernel_names = [kernel_bank[i][0] for i in kernel_ids]

            # Create composite kernel
            composite_kernel = functools.reduce(
                lambda a, b: random_binary_map(a, b, rng=self.rng),
                [
                    create_kernel(
                        k,
                        self.length,
                        int(self.max_period_ratio * self.length),
                        gaussians_periodic,
                        kernel_periods,
                        rng=self.rng,
                    )
                    for k in kernel_names
                ],
            )

            # Set up GP model
            train_x = torch.linspace(0, 1, self.length)
            trend = self.rng.choice([True, False])
            mean_module = gpytorch.means.LinearMean(input_size=1) if trend else gpytorch.means.ConstantMean()
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_covar=torch.diag(torch.full_like(train_x, self.likelihood_noise_level**2))
            )
            model = GPModel(train_x, None, likelihood, mean_module, composite_kernel)

            # Determine noise level
            noise = {"high": 1e-1, "moderate": 1e-2, "low": 1e-3}.get(
                self.noise_level,
                self.rng.choice([1e-1, 1e-2, 1e-3], p=[0.1, 0.2, 0.7]),
            )

            # Sample from GP prior with robust error handling
            model.eval()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with (
                        torch.no_grad(),
                        gpytorch.settings.fast_pred_var(),
                        gpytorch.settings.cholesky_jitter(
                            max(noise * (10**attempt), 1e-4)
                        ),  # Increase jitter on retries, with a minimum floor
                        gpytorch.settings.max_cholesky_size(self.length),  # Limit decomposition size
                    ):
                        y_sample = model(train_x).sample().numpy()
                        # y_sample shape: (self.length,) (should be 1D)
                    break
                except (RuntimeError, IndexError) as e:
                    if attempt == max_retries - 1:
                        # If all attempts fail, generate a simple fallback
                        print(f"GP sampling failed after {max_retries} attempts: {e}")
                        print("Generating fallback sample with simpler kernel")
                        # Create a simple RBF kernel as fallback
                        simple_kernel = gpytorch.kernels.RBFKernel()
                        simple_model = GPModel(train_x, None, likelihood, mean_module, simple_kernel)
                        simple_model.eval()
                        with torch.no_grad():
                            y_sample = simple_model(train_x).sample().numpy()
                        break
                    else:
                        print(f"GP sampling attempt {attempt + 1} failed: {e}. Retrying with higher jitter...")

            # Optionally add peak spikes
            if self.rng.random() < self.peak_spike_ratio:
                periodicities = extract_periodicities(composite_kernel, self.length)
                if len(periodicities) > 0:
                    p = int(np.round(max(periodicities)))
                    spikes_type = self.rng.choice(["regular", "patchy"], p=[0.3, 0.7])
                    spikes = generate_peak_spikes(self.length, p, spikes_type=spikes_type)
                    # y_sample is 1D, so use y_sample[:p].argmax()
                    spikes_shift = p - y_sample[:p].argmax() if p > 0 and p <= len(y_sample) else 0
                    spikes = np.roll(spikes, -spikes_shift)
                    if spikes.max() < 0:
                        y_sample = y_sample + spikes + 1
                    else:
                        y_sample = y_sample * spikes

            return y_sample
