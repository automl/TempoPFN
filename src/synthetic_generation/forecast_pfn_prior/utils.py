import numpy as np


def weibull_noise(k=2, length=1, median=1, rng=None):
    """
    Function to generate weibull noise with a fixed median
    """
    if rng is None:
        rng = np.random.default_rng()
    # Lambda is chosen such that the median is a given value
    lamda = median / (np.log(2) ** (1 / k))
    return lamda * rng.weibull(k, length)


def shift_axis(days, shift):
    if shift is None:
        return days
    return days - shift * days[-1]


def get_random_walk_series(length, movements=[-1, 1], rng=None):
    """
    Function to generate a random walk series with a specified length
    """
    if rng is None:
        rng = np.random.default_rng()
    random_walk = list()
    random_walk.append(rng.choice(movements))
    for i in range(1, length):
        movement = rng.choice(movements)
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return np.array(random_walk)


def sample_scale(low_ratio=0.6, moderate_ratio=0.3, rng=None):
    """
    Function to sample scale such that it follows 60-30-10 distribution
    i.e. 60% of the times it is very low, 30% of the times it is moderate and
    the rest 10% of the times it is high
    """
    if rng is None:
        rng = np.random.default_rng()
    rand = rng.random()
    # very low noise
    if rand <= low_ratio:
        return rng.uniform(0, 0.1)
    # moderate noise
    elif rand <= (low_ratio + moderate_ratio):
        return rng.uniform(0.2, 0.5)
    # high noise
    else:
        return rng.uniform(0.7, 0.9)


def get_transition_coefficients(context_length):
    """
    Transition series refers to the linear combination of 2 series
    S1 and S2 such that the series S represents S1 for a period and S2
    for the remaining period. We model S as S = (1 - f) * S1 + f * S2
    Here f = 1 / (1 + e^{-k (x-m)}) where m = (a + b) / 2 and k is chosen
    such that f(a) = 0.1 (and hence f(b) = 0.9). a and b refer to
    0.2 * CONTEXT_LENGTH and 0.8 * CONTEXT_LENGTH
    """
    # a and b are chosen with 0.2 and 0.8 parameters
    a, b = 0.2 * context_length, 0.8 * context_length

    # fixed to this value
    f_a = 0.1

    m = (a + b) / 2
    k = 1 / (a - m) * np.log(f_a / (1 - f_a))

    coeff = 1 / (1 + np.exp(-k * (np.arange(1, context_length + 1) - m)))
    return coeff
