import gpytorch
import numpy as np
from gpytorch.kernels import AdditiveKernel, PeriodicKernel, ProductKernel, ScaleKernel


def custom_gaussian_sample(
    max_period_length,
    kernel_periods=None,
    gaussian_sample=True,
    allow_extension=True,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    means = (
        np.array(kernel_periods)
        if kernel_periods is not None
        else np.array([3, 5, 7, 14, 20, 21, 24, 30, 60, 90, 120])
    )

    if allow_extension:
        if max_period_length > 200:
            st = (
                max_period_length // 2
                if max(means) < max_period_length // 2
                else max(means) + 100
            )
            means = np.append(means, np.arange(st, max_period_length, 100))
        else:
            if max(means) < max_period_length / 2:
                means = np.append(
                    means, np.array([max_period_length // 2, max_period_length])
                )
            elif max(means) < max_period_length:
                means = np.append(means, max_period_length)

    means = means[means <= max_period_length]
    selected_mean = rng.choice(means)

    if gaussian_sample:
        # Define corresponding standard deviations using np.sqrt(means) * 2
        std_devs = np.sqrt(means) ** 1.2  # / (means *0.008)
        selected_std = std_devs[np.where(means == selected_mean)][0]
        sample = rng.normal(selected_mean, selected_std)
    else:
        sample = selected_mean

    if sample < 1:
        sample = np.ceil(np.abs(sample))

    return int(sample)


def create_kernel(
    kernel: str,
    seq_len: int,
    max_period_length: int = 365,
    max_degree: int = 5,
    gaussians_periodic: bool = False,
    kernel_periods=None,
    kernel_counter=None,
    freq=None,
    exact_freqs=False,
    gaussian_sample=True,
    subfreq="",
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    scale_kernel = rng.choice([True, False])
    lengthscale = rng.uniform(0.1, 5.0)
    if kernel == "linear_kernel":
        sigma_prior = gpytorch.priors.GammaPrior(rng.uniform(1, 6), rng.uniform(0.1, 1))
        kernel = gpytorch.kernels.LinearKernel(variance_prior=sigma_prior)
    elif kernel == "rbf_kernel":
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = lengthscale
    elif kernel == "periodic_kernel":
        if gaussians_periodic:
            if exact_freqs and freq != "Y" and kernel_counter is not None:
                period_length = custom_gaussian_sample(
                    max_period_length,
                    kernel_periods=kernel_periods[:-3]
                    if (kernel_counter["periodic_kernel"] <= 2) and (subfreq == "")
                    else kernel_periods,
                    gaussian_sample=gaussian_sample,
                    allow_extension=(kernel_counter["periodic_kernel"] > 2),
                    rng=rng,
                )
                kernel_counter["periodic_kernel"] -= 1
            else:
                period_length = custom_gaussian_sample(
                    max_period_length, kernel_periods, gaussian_sample=True, rng=rng
                )
        else:
            period_length = rng.integers(1, max_period_length)
        kernel = gpytorch.kernels.PeriodicKernel()
        kernel.period_length = period_length / seq_len
        kernel.lengthscale = lengthscale
    elif kernel == "polynomial_kernel":
        offset_prior = gpytorch.priors.GammaPrior(
            rng.uniform(1, 4), rng.uniform(0.1, 1)
        )
        degree = rng.integers(1, max_degree)
        kernel = gpytorch.kernels.PolynomialKernel(
            offset_prior=offset_prior, power=degree
        )
    elif kernel == "matern_kernel":
        nu = rng.choice([0.5, 1.5, 2.5])  # Roughness parameter
        kernel = gpytorch.kernels.MaternKernel(nu=nu)
        kernel.lengthscale = lengthscale
    elif kernel == "rational_quadratic_kernel":
        alpha = rng.uniform(0.1, 10.0)  # Scale mixture parameter
        kernel = gpytorch.kernels.RQKernel(alpha=alpha)
        kernel.lengthscale = lengthscale
    elif kernel == "spectral_mixture_kernel":
        num_mixtures = rng.integers(2, 6)  # Number of spectral mixture components
        kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if scale_kernel:
        kernel = gpytorch.kernels.ScaleKernel(kernel)
    return kernel


def extract_periodicities(kernel, seq_len):
    periodicities = []

    # Base case: if the kernel is a PeriodicKernel, extract its period_length
    if isinstance(kernel, PeriodicKernel):
        periodicities.append(kernel.period_length.item() * seq_len)

    # If the kernel is a composite kernel (Additive, Product, Scale), recursively extract periodicities
    elif isinstance(kernel, (AdditiveKernel, ProductKernel)):
        for sub_kernel in kernel.kernels:
            periodicities.extend(extract_periodicities(sub_kernel, seq_len))

    elif isinstance(kernel, ScaleKernel):
        periodicities.extend(extract_periodicities(kernel.base_kernel, seq_len))

    return periodicities


def random_binary_map(a: gpytorch.kernels.Kernel, b: gpytorch.kernels.Kernel, rng=None):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    if rng is None:
        rng = np.random.default_rng()
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return rng.choice(binary_maps)(a, b)
