import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.fft as fft
import torch
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature._base import (
    day_of_month,
    day_of_month_index,
    day_of_week,
    day_of_week_index,
    day_of_year,
    hour_of_day,
    hour_of_day_index,
    minute_of_hour,
    minute_of_hour_index,
    month_of_year,
    month_of_year_index,
    second_of_minute,
    second_of_minute_index,
    week_of_year,
    week_of_year_index,
)
from gluonts.time_feature.holiday import (
    BLACK_FRIDAY,
    CHRISTMAS_DAY,
    CHRISTMAS_EVE,
    CYBER_MONDAY,
    EASTER_MONDAY,
    EASTER_SUNDAY,
    GOOD_FRIDAY,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    MEMORIAL_DAY,
    NEW_YEARS_DAY,
    NEW_YEARS_EVE,
    THANKSGIVING,
    SpecialDateFeatureSet,
    exponential_kernel,
    squared_exponential_kernel,
)
from gluonts.time_feature.seasonality import get_seasonality
from scipy.signal import find_peaks

from src.data.constants import BASE_END_DATE, BASE_START_DATE
from src.data.frequency import (
    Frequency,
    validate_frequency_safety,
)
from src.utils.utils import device

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Enhanced feature sets for different frequencies
ENHANCED_TIME_FEATURES = {
    # High-frequency features (seconds, minutes)
    "high_freq": {
        "normalized": [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
        ],
        "index": [
            second_of_minute_index,
            minute_of_hour_index,
            hour_of_day_index,
            day_of_week_index,
        ],
    },
    # Medium-frequency features (hourly, daily)
    "medium_freq": {
        "normalized": [
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
            month_of_year,
        ],
        "index": [
            hour_of_day_index,
            day_of_week_index,
            day_of_month_index,
            week_of_year_index,
        ],
    },
    # Low-frequency features (weekly, monthly)
    "low_freq": {
        "normalized": [day_of_week, day_of_month, month_of_year, week_of_year],
        "index": [day_of_week_index, month_of_year_index, week_of_year_index],
    },
}

# Holiday features for different markets/regions
HOLIDAY_FEATURE_SETS = {
    "us_business": [
        NEW_YEARS_DAY,
        MEMORIAL_DAY,
        INDEPENDENCE_DAY,
        LABOR_DAY,
        THANKSGIVING,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
    "us_retail": [
        NEW_YEARS_DAY,
        EASTER_SUNDAY,
        MEMORIAL_DAY,
        INDEPENDENCE_DAY,
        LABOR_DAY,
        THANKSGIVING,
        BLACK_FRIDAY,
        CYBER_MONDAY,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
    "christian": [
        NEW_YEARS_DAY,
        GOOD_FRIDAY,
        EASTER_SUNDAY,
        EASTER_MONDAY,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
}


class TimeFeatureGenerator:
    """
    Enhanced time feature generator that leverages full GluonTS capabilities.
    """

    def __init__(
        self,
        use_enhanced_features: bool = True,
        use_holiday_features: bool = True,
        holiday_set: str = "us_business",
        holiday_kernel: str = "exponential",
        holiday_kernel_alpha: float = 1.0,
        use_index_features: bool = True,
        k_max: int = 15,
        include_seasonality_info: bool = True,
        use_auto_seasonality: bool = False,  # New parameter
        max_seasonal_periods: int = 3,  # New parameter
    ):
        """
        Initialize enhanced time feature generator.

        Parameters
        ----------
        use_enhanced_features : bool
            Whether to use frequency-specific enhanced features
        use_holiday_features : bool
            Whether to include holiday features
        holiday_set : str
            Which holiday set to use ('us_business', 'us_retail', 'christian')
        holiday_kernel : str
            Holiday kernel type ('indicator', 'exponential', 'squared_exponential')
        holiday_kernel_alpha : float
            Kernel parameter for exponential kernels
        use_index_features : bool
            Whether to include index-based features alongside normalized ones
        k_max : int
            Maximum number of time features to pad to
        include_seasonality_info : bool
            Whether to include seasonality information as features
        use_auto_seasonality : bool
            Whether to use automatic FFT-based seasonality detection
        max_seasonal_periods : int
            Maximum number of seasonal periods to detect automatically
        """
        self.use_enhanced_features = use_enhanced_features
        self.use_holiday_features = use_holiday_features
        self.holiday_set = holiday_set
        self.use_index_features = use_index_features
        self.k_max = k_max
        self.include_seasonality_info = include_seasonality_info
        self.use_auto_seasonality = use_auto_seasonality
        self.max_seasonal_periods = max_seasonal_periods

        # Initialize holiday feature set
        self.holiday_feature_set = None
        if use_holiday_features and holiday_set in HOLIDAY_FEATURE_SETS:
            kernel_func = self._get_holiday_kernel(holiday_kernel, holiday_kernel_alpha)
            self.holiday_feature_set = SpecialDateFeatureSet(
                HOLIDAY_FEATURE_SETS[holiday_set], kernel_func
            )

    def _get_holiday_kernel(self, kernel_type: str, alpha: float):
        """Get holiday kernel function."""
        if kernel_type == "exponential":
            return exponential_kernel(alpha)
        elif kernel_type == "squared_exponential":
            return squared_exponential_kernel(alpha)
        else:
            # Default indicator kernel
            return lambda x: float(x == 0)

    def _get_feature_category(self, freq_str: str) -> str:
        """Determine feature category based on frequency."""
        if freq_str in ["s", "1min", "5min", "10min", "15min"]:
            return "high_freq"
        elif freq_str in ["h", "D"]:
            return "medium_freq"
        else:
            return "low_freq"

    def _compute_enhanced_features(
        self, period_index: pd.PeriodIndex, freq_str: str
    ) -> np.ndarray:
        """Compute enhanced time features based on frequency."""
        if not self.use_enhanced_features:
            return np.array([]).reshape(len(period_index), 0)

        category = self._get_feature_category(freq_str)
        feature_config = ENHANCED_TIME_FEATURES[category]

        features = []

        # Add normalized features
        for feat_func in feature_config["normalized"]:
            try:
                feat_values = feat_func(period_index)
                features.append(feat_values)
            except Exception:
                continue

        # Add index features if enabled
        if self.use_index_features:
            for feat_func in feature_config["index"]:
                try:
                    feat_values = feat_func(period_index)
                    # Normalize index features to [0, 1] range
                    if feat_values.max() > 0:
                        feat_values = feat_values / feat_values.max()
                    features.append(feat_values)
                except Exception:
                    continue

        if features:
            return np.stack(features, axis=-1)
        else:
            return np.array([]).reshape(len(period_index), 0)

    def _compute_holiday_features(self, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Compute holiday features."""
        if not self.use_holiday_features or self.holiday_feature_set is None:
            return np.array([]).reshape(len(date_range), 0)

        try:
            holiday_features = self.holiday_feature_set(date_range)
            return holiday_features.T  # Transpose to get [time, features] shape
        except Exception:
            return np.array([]).reshape(len(date_range), 0)

    def _detect_auto_seasonality(self, time_series_values: np.ndarray) -> list:
        """
        Detect seasonal periods automatically using FFT analysis.

        Parameters
        ----------
        time_series_values : np.ndarray
            Time series values for seasonality detection

        Returns
        -------
        list
            List of detected seasonal periods
        """
        if not self.use_auto_seasonality or len(time_series_values) < 10:
            return []

        try:
            # Remove NaN values
            values = time_series_values[~np.isnan(time_series_values)]
            if len(values) < 10:
                return []

            # Simple linear detrending
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            trend = np.polyval(coeffs, x)
            detrended = values - trend

            # Apply Hann window to reduce spectral leakage
            window = np.hanning(len(detrended))
            windowed = detrended * window

            # Zero padding for better frequency resolution
            padded_length = len(windowed) * 2
            padded_values = np.zeros(padded_length)
            padded_values[: len(windowed)] = windowed

            # Compute FFT
            fft_values = fft.rfft(padded_values)
            fft_magnitudes = np.abs(fft_values)
            freqs = np.fft.rfftfreq(padded_length)

            # Exclude DC component
            fft_magnitudes[0] = 0.0

            # Find peaks with threshold (5% of max magnitude)
            threshold = 0.05 * np.max(fft_magnitudes)
            peak_indices, _ = find_peaks(fft_magnitudes, height=threshold)

            if len(peak_indices) == 0:
                return []

            # Sort by magnitude and take top periods
            sorted_indices = peak_indices[
                np.argsort(fft_magnitudes[peak_indices])[::-1]
            ]
            top_indices = sorted_indices[: self.max_seasonal_periods]

            # Convert frequencies to periods
            periods = []
            for idx in top_indices:
                if freqs[idx] > 0:
                    period = 1.0 / freqs[idx]
                    # Scale back to original length and round
                    period = round(period / 2)  # Account for zero padding
                    if 2 <= period <= len(values) // 2:  # Reasonable period range
                        periods.append(period)

            return list(set(periods))  # Remove duplicates

        except Exception:
            return []

    def _compute_seasonality_features(
        self,
        period_index: pd.PeriodIndex,
        freq_str: str,
        time_series_values: np.ndarray = None,
    ) -> np.ndarray:
        """Compute seasonality-aware features."""
        if not self.include_seasonality_info:
            return np.array([]).reshape(len(period_index), 0)

        all_seasonal_features = []

        # Original frequency-based seasonality
        try:
            seasonality = get_seasonality(freq_str)
            if seasonality > 1:
                positions = np.arange(len(period_index))
                sin_feat = np.sin(2 * np.pi * positions / seasonality)
                cos_feat = np.cos(2 * np.pi * positions / seasonality)
                all_seasonal_features.extend([sin_feat, cos_feat])
        except Exception:
            pass

        # Automatic seasonality detection
        if self.use_auto_seasonality and time_series_values is not None:
            auto_periods = self._detect_auto_seasonality(time_series_values)
            for period in auto_periods:
                try:
                    positions = np.arange(len(period_index))
                    sin_feat = np.sin(2 * np.pi * positions / period)
                    cos_feat = np.cos(2 * np.pi * positions / period)
                    all_seasonal_features.extend([sin_feat, cos_feat])
                except Exception:
                    continue

        if all_seasonal_features:
            return np.stack(all_seasonal_features, axis=-1)
        else:
            return np.array([]).reshape(len(period_index), 0)

    def compute_features(
        self,
        period_index: pd.PeriodIndex,
        date_range: pd.DatetimeIndex,
        freq_str: str,
        time_series_values: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute all time features for given period index.

        Parameters
        ----------
        period_index : pd.PeriodIndex
            Period index for computing features
        date_range : pd.DatetimeIndex
            Corresponding datetime index for holiday features
        freq_str : str
            Frequency string
        time_series_values : np.ndarray, optional
            Time series values for automatic seasonality detection

        Returns
        -------
        np.ndarray
            Time features array of shape [time_steps, num_features]
        """
        all_features = []

        # Standard GluonTS features
        try:
            standard_features = time_features_from_frequency_str(freq_str)
            if standard_features:
                std_feat = np.stack(
                    [feat(period_index) for feat in standard_features], axis=-1
                )
                all_features.append(std_feat)
        except Exception:
            pass

        # Enhanced features
        enhanced_feat = self._compute_enhanced_features(period_index, freq_str)
        if enhanced_feat.shape[1] > 0:
            all_features.append(enhanced_feat)

        # Holiday features
        holiday_feat = self._compute_holiday_features(date_range)
        if holiday_feat.shape[1] > 0:
            all_features.append(holiday_feat)

        # Seasonality features (including auto-detected)
        seasonality_feat = self._compute_seasonality_features(
            period_index, freq_str, time_series_values
        )
        if seasonality_feat.shape[1] > 0:
            all_features.append(seasonality_feat)

        if all_features:
            combined_features = np.concatenate(all_features, axis=-1)
        else:
            combined_features = np.zeros((len(period_index), 1))

        return combined_features


def compute_batch_time_features(
    start: List[np.datetime64],
    history_length: int,
    future_length: int,
    batch_size: int,
    frequency: List[Frequency],
    K_max: int = 6,
    time_feature_config: Optional[Dict[str, Any]] = None,
):
    """
    Compute time features from start timestamps and frequency.

    Parameters
    ----------
    start : array-like, shape (batch_size,)
        Start timestamps for each batch item.
    history_length : int
        Length of history sequence.
    future_length : int
        Length of target sequence.
    batch_size : int
        Batch size.
    frequency : array-like, shape (batch_size,)
        Frequency of the time series.
    K_max : int, optional
        Maximum number of time features to pad to (default: 6).
    time_feature_config : dict, optional
        Configuration for enhanced time features.

    Returns
    -------
    tuple
        (history_time_features, target_time_features) where each is a torch.Tensor
        of shape (batch_size, length, K_max).
    """
    # Initialize enhanced feature generator
    feature_config = time_feature_config or {}
    feature_generator = TimeFeatureGenerator(**feature_config)

    # Generate timestamps and features
    history_features_list = []
    future_features_list = []
    total_length = history_length + future_length
    for i in range(batch_size):
        frequency_i = frequency[i]
        freq_str = frequency_i.to_pandas_freq(for_date_range=True)
        period_freq_str = frequency_i.to_pandas_freq(for_date_range=False)

        # Validate start timestamp is within safe bounds
        start_ts = pd.Timestamp(start[i])
        if not validate_frequency_safety(start_ts, total_length, frequency_i):
            logger.debug(
                f"Start date {start_ts} not safe for total_length={total_length}, frequency={frequency_i}. "
                f"Using BASE_START_DATE instead."
            )
            start_ts = BASE_START_DATE

        # Create history range with bounds checking
        history_range = pd.date_range(
            start=start_ts, periods=history_length, freq=freq_str
        )

        # Check if history range goes beyond safe bounds
        if history_range[-1] > BASE_END_DATE:
            safe_start = BASE_END_DATE - pd.tseries.frequencies.to_offset(freq_str) * (
                history_length + future_length
            )
            if safe_start < BASE_START_DATE:
                safe_start = BASE_START_DATE
            history_range = pd.date_range(
                start=safe_start, periods=history_length, freq=freq_str
            )

        future_start = history_range[-1] + pd.tseries.frequencies.to_offset(freq_str)
        future_range = pd.date_range(
            start=future_start, periods=future_length, freq=freq_str
        )

        # Convert to period indices
        history_period_idx = history_range.to_period(period_freq_str)
        future_period_idx = future_range.to_period(period_freq_str)

        # Compute enhanced features
        history_features = feature_generator.compute_features(
            history_period_idx, history_range, freq_str
        )
        future_features = feature_generator.compute_features(
            future_period_idx, future_range, freq_str
        )

        # Pad or truncate to K_max
        history_features = _pad_or_truncate_features(history_features, K_max)
        future_features = _pad_or_truncate_features(future_features, K_max)

        history_features_list.append(history_features)
        future_features_list.append(future_features)

    # Stack into batch tensors
    history_time_features = np.stack(history_features_list, axis=0)
    future_time_features = np.stack(future_features_list, axis=0)

    return (
        torch.from_numpy(history_time_features).float().to(device),
        torch.from_numpy(future_time_features).float().to(device),
    )


def _pad_or_truncate_features(features: np.ndarray, K_max: int) -> np.ndarray:
    """Pad with zeros or truncate features to K_max dimensions."""
    seq_len, num_features = features.shape

    if num_features < K_max:
        # Pad with zeros
        padding = np.zeros((seq_len, K_max - num_features))
        features = np.concatenate([features, padding], axis=-1)
    elif num_features > K_max:
        # Truncate to K_max (keep most important features first)
        features = features[:, :K_max]

    return features
