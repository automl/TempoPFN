import numpy as np
import torch
from scipy import signal
from statsmodels.tsa.stattools import acf


def lempel_ziv_complexity(binary_sequence: np.ndarray) -> int:
    """Computes the Lempel-Ziv complexity of a binary sequence."""
    sub_strings = set()
    n = len(binary_sequence)
    i = 0
    count = 0
    while i < n:
        sub_str = ""
        for j in range(i, n):
            sub_str += str(binary_sequence[j])
            if sub_str not in sub_strings:
                sub_strings.add(sub_str)
                count += 1
                i = j + 1
                break
        else:
            i += 1
    return count


def is_low_quality(
    series: torch.Tensor,
    autocorr_threshold: float = 0.2,
    snr_threshold: float = 0.5,
    complexity_threshold: float = 0.4,
) -> bool:
    """
    Returns True if the series appears non-forecastable (noise-like):
    - weak autocorrelation
    - low SNR proxy
    - high normalized Lempel-Ziv complexity
    """
    x = series.squeeze().detach().cpu().numpy()
    if x.size < 20:
        return True
    if np.var(x) < 1e-10:
        return True

    x_detrended = signal.detrend(x)

    try:
        max_lags = min(len(x_detrended) // 4, 40)
        if max_lags < 1:
            autocorr_strength = 0.0
        else:
            acf_vals = acf(x_detrended, nlags=max_lags, fft=True)[1:]
            autocorr_strength = float(np.max(np.abs(acf_vals)))
    except Exception:
        autocorr_strength = 0.0

    win_size = max(3, min(len(x) // 10, 15))
    signal_est = np.convolve(x, np.ones(win_size) / win_size, mode="valid")
    noise_est = x[win_size - 1 :] - signal_est
    var_signal = float(np.var(signal_est))
    var_noise = float(np.var(noise_est))
    snr_proxy = var_signal / var_noise if var_noise > 1e-8 else 1.0

    median_val = float(np.median(x_detrended))
    binary_seq = (x_detrended > median_val).astype(np.uint8)
    complexity_score = lempel_ziv_complexity(binary_seq)
    normalized_complexity = complexity_score / max(1, len(binary_seq))

    is_random_like = (snr_proxy < snr_threshold) and (
        normalized_complexity > complexity_threshold
    )
    is_uncorrelated = autocorr_strength < autocorr_threshold
    return bool(is_uncorrelated and is_random_like)
