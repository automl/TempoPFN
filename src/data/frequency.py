"""
Comprehensive frequency management module for time series forecasting.

This module centralizes all frequency-related functionality including:
- Frequency enum with helper methods
- Frequency parsing and validation
- Pandas frequency string conversion
- Safety checks for date ranges
- Frequency selection utilities
- All frequency constants and mappings
"""

import logging
import re
from enum import Enum

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.data.constants import BASE_END_DATE, BASE_START_DATE, MAX_YEARS

logger = logging.getLogger(__name__)


class Frequency(Enum):
    """
    Enhanced Frequency enum with comprehensive helper methods.

    Each frequency includes methods for pandas conversion, safety checks,
    and other frequency-specific operations.
    """

    A = "A"  # Annual
    Q = "Q"  # Quarterly
    M = "M"  # Monthly
    W = "W"  # Weekly
    D = "D"  # Daily
    H = "h"  # Hourly
    S = "s"  # Seconds
    T1 = "1min"  # 1 minute
    T5 = "5min"  # 5 minutes
    T10 = "10min"  # 10 minutes
    T15 = "15min"  # 15 minutes
    T30 = "30min"  # 30 minutes

    def to_pandas_freq(self, for_date_range: bool = True) -> str:
        """
        Convert to pandas frequency string.

        Args:
            for_date_range: If True, use strings suitable for pd.date_range().
                           If False, use strings suitable for pd.PeriodIndex().

        Returns:
            Pandas frequency string
        """
        base, prefix, _ = FREQUENCY_MAPPING[self]

        # Special handling for date_range vs period compatibility
        if for_date_range:
            # For date_range, use modern pandas frequency strings
            if self == Frequency.M:
                return "ME"  # Month End
            elif self == Frequency.A:
                return "YE"  # Year End
            elif self == Frequency.Q:
                return "QE"  # Quarter End
        else:
            # For periods, use legacy frequency strings
            if self == Frequency.M:
                return "M"  # Month for periods
            elif self == Frequency.A:
                return "Y"  # Year for periods (not YE)
            elif self == Frequency.Q:
                return "Q"  # Quarter for periods (not QE)

        # Construct frequency string for other frequencies
        if prefix:
            return f"{prefix}{base}"
        else:
            return base

    def to_pandas_offset(self) -> str:
        """Get pandas offset string for time delta calculations."""
        return FREQUENCY_TO_OFFSET[self]

    def get_days_per_period(self) -> float:
        """Get approximate days per period for this frequency."""
        _, _, days = FREQUENCY_MAPPING[self]
        return days

    def get_max_safe_length(self) -> int:
        """Get maximum safe sequence length to prevent timestamp overflow."""
        return ALL_FREQUENCY_MAX_LENGTHS.get(self, float("inf"))

    def is_high_frequency(self) -> bool:
        """Check if this is a high frequency (minute/second level)."""
        return self in [
            Frequency.S,
            Frequency.T1,
            Frequency.T5,
            Frequency.T10,
            Frequency.T15,
            Frequency.T30,
        ]

    def is_low_frequency(self) -> bool:
        """Check if this is a low frequency (annual/quarterly/monthly)."""
        return self in [Frequency.A, Frequency.Q, Frequency.M]

    def get_seasonality(self) -> int:
        """Get typical seasonality for this frequency."""
        seasonality_map = {
            Frequency.S: 3600,  # 1 hour of seconds
            Frequency.T1: 60,  # 1 hour of minutes
            Frequency.T5: 12,  # 1 hour of 5-minute intervals
            Frequency.T10: 6,  # 1 hour of 10-minute intervals
            Frequency.T15: 4,  # 1 hour of 15-minute intervals
            Frequency.T30: 2,  # 1 hour of 30-minute intervals
            Frequency.H: 24,  # 1 day of hours
            Frequency.D: 7,  # 1 week of days
            Frequency.W: 52,  # 1 year of weeks
            Frequency.M: 12,  # 1 year of months
            Frequency.Q: 4,  # 1 year of quarters
            Frequency.A: 1,  # No clear seasonality for annual
        }
        return seasonality_map.get(self, 1)

    def get_gift_eval_weight(self) -> float:
        """Get GIFT eval dataset frequency weight."""
        return GIFT_EVAL_FREQUENCY_WEIGHTS.get(self, 0.1)

    def get_length_range(self) -> tuple[int, int, int, int]:
        """Get (min_length, max_length, optimal_start, optimal_end) for this frequency."""
        return GIFT_EVAL_LENGTH_RANGES.get(self, (50, 1000, 100, 500))


# ============================================================================
# Frequency Mappings and Constants
# ============================================================================

# Core frequency mapping: (pandas_base, prefix, days_per_period)
FREQUENCY_MAPPING: dict[Frequency, tuple[str, str, float]] = {
    Frequency.A: (
        "YE",
        "",
        365.25,
    ),  # Average days per year (accounting for leap years)
    Frequency.Q: ("Q", "", 91.3125),  # 365.25/4 - average days per quarter
    Frequency.M: ("M", "", 30.4375),  # 365.25/12 - average days per month
    Frequency.W: ("W", "", 7),
    Frequency.D: ("D", "", 1),
    Frequency.H: ("h", "", 1 / 24),
    Frequency.S: ("s", "", 1 / 86400),  # 24*60*60
    Frequency.T1: ("min", "1", 1 / 1440),  # 24*60
    Frequency.T5: ("min", "5", 1 / 288),  # 24*60/5
    Frequency.T10: ("min", "10", 1 / 144),  # 24*60/10
    Frequency.T15: ("min", "15", 1 / 96),  # 24*60/15
    Frequency.T30: ("min", "30", 1 / 48),  # 24*60/30
}

# Frequency to pandas offset mapping for calculating time deltas
FREQUENCY_TO_OFFSET: dict[Frequency, str] = {
    Frequency.A: "AS",  # Annual start
    Frequency.Q: "QS",  # Quarter start
    Frequency.M: "MS",  # Month start
    Frequency.W: "W",  # Weekly
    Frequency.D: "D",  # Daily
    Frequency.H: "H",  # Hourly
    Frequency.T1: "1T",  # 1 minute
    Frequency.T5: "5T",  # 5 minutes
    Frequency.T10: "10T",  # 10 minutes
    Frequency.T15: "15T",  # 15 minutes
    Frequency.T30: "30T",  # 30 minutes
    Frequency.S: "S",  # Seconds
}

# Maximum sequence lengths to avoid pandas OutOfBoundsDatetime errors
SHORT_FREQUENCY_MAX_LENGTHS = {
    Frequency.A: MAX_YEARS,
    Frequency.Q: MAX_YEARS * 4,
    Frequency.M: MAX_YEARS * 12,
    Frequency.W: int(MAX_YEARS * 52.1775),
    Frequency.D: int(MAX_YEARS * 365.2425),
}

HIGH_FREQUENCY_MAX_LENGTHS = {
    Frequency.H: int(MAX_YEARS * 365.2425 * 24),
    Frequency.S: int(MAX_YEARS * 365.2425 * 24 * 60 * 60),
    Frequency.T1: int(MAX_YEARS * 365.2425 * 24 * 60),
    Frequency.T5: int(MAX_YEARS * 365.2425 * 24 * 12),
    Frequency.T10: int(MAX_YEARS * 365.2425 * 24 * 6),
    Frequency.T15: int(MAX_YEARS * 365.2425 * 24 * 4),
    Frequency.T30: int(MAX_YEARS * 365.2425 * 24 * 2),
}

# Combined max lengths for all frequencies
ALL_FREQUENCY_MAX_LENGTHS = {
    **SHORT_FREQUENCY_MAX_LENGTHS,
    **HIGH_FREQUENCY_MAX_LENGTHS,
}

# GIFT eval-based frequency weights from actual dataset analysis
GIFT_EVAL_FREQUENCY_WEIGHTS: dict[Frequency, float] = {
    Frequency.H: 25.0,  # Hourly - most common
    Frequency.D: 23.4,  # Daily - second most common
    Frequency.W: 12.9,  # Weekly - third most common
    Frequency.T15: 9.7,  # 15-minute
    Frequency.T5: 9.7,  # 5-minute
    Frequency.M: 7.3,  # Monthly
    Frequency.T10: 4.8,  # 10-minute
    Frequency.S: 4.8,  # 10-second
    Frequency.T1: 1.6,  # 1-minute
    Frequency.Q: 0.8,  # Quarterly
    Frequency.A: 0.8,  # Annual
}

# GIFT eval-based length ranges derived from actual dataset analysis
# Format: (min_length, max_length, optimal_start, optimal_end)
GIFT_EVAL_LENGTH_RANGES: dict[Frequency, tuple[int, int, int, int]] = {
    # Low frequency ranges (based on actual GIFT eval data + logical extensions)
    Frequency.A: (25, 100, 30, 70),
    Frequency.Q: (25, 150, 50, 120),
    Frequency.M: (40, 1000, 100, 600),
    Frequency.W: (50, 3500, 100, 1500),
    # Medium frequency ranges
    Frequency.D: (150, 25000, 300, 7000),  # Daily: covers 1-year+ scenarios
    Frequency.H: (600, 35000, 700, 17000),
    # High frequency ranges (extended for shorter realistic scenarios)
    Frequency.T1: (200, 2500, 1200, 1800),  # 1-minute: day to few days
    Frequency.S: (7500, 9500, 7900, 9000),
    Frequency.T15: (1000, 140000, 50000, 130000),
    Frequency.T5: (200, 105000, 20000, 95000),
    Frequency.T10: (40000, 55000, 47000, 52000),
    Frequency.T30: (100, 50000, 10000, 40000),
}


# ============================================================================
# Frequency Parsing and Validation
# ============================================================================


def parse_frequency(freq_str: str) -> Frequency:
    """
    Parse frequency string to Frequency enum, robust to variations.

    Handles various frequency string formats:
    - Standard: "A", "Q", "M", "W", "D", "H", "S"
    - Pandas-style: "A-DEC", "W-SUN", "QE-MAR"
    - Minutes: "5T", "10min", "1T"
    - Case variations: "a", "h", "D"

    Args:
        freq_str: The frequency string to parse (e.g., "5T", "W-SUN", "M")

    Returns:
        Corresponding Frequency enum member

    Raises:
        ValueError: If the frequency string is not supported
    """
    # Handle minute-based frequencies BEFORE pandas standardization
    # because pandas converts "5T" to just "min", losing the multiplier
    minute_match = re.match(r"^(\d*)T$", freq_str, re.IGNORECASE) or re.match(r"^(\d*)min$", freq_str, re.IGNORECASE)
    if minute_match:
        multiplier = int(minute_match.group(1)) if minute_match.group(1) else 1
        enum_key = f"T{multiplier}"
        try:
            return Frequency[enum_key]
        except KeyError:
            logger.warning(
                f"Unsupported minute frequency '{freq_str}' (multiplier: {multiplier}). "
                f"Falling back to '1min' ({Frequency.T1.value})."
            )
            return Frequency.T1

    # Now standardize frequency string for other cases
    try:
        offset = pd.tseries.frequencies.to_offset(freq_str)
        standardized_freq = offset.name
    except Exception:
        standardized_freq = freq_str

    # Handle other frequencies by their base (e.g., 'W-SUN' -> 'W', 'A-DEC' -> 'A')
    base_freq = standardized_freq.split("-")[0].upper()

    freq_map = {
        "A": Frequency.A,
        "Y": Frequency.A,  # Alias for Annual
        "YE": Frequency.A,  # Alias for Annual
        "Q": Frequency.Q,
        "QE": Frequency.Q,  # Alias for Quarterly
        "M": Frequency.M,
        "ME": Frequency.M,  # Alias for Monthly
        "W": Frequency.W,
        "D": Frequency.D,
        "H": Frequency.H,
        "S": Frequency.S,
    }

    if base_freq in freq_map:
        return freq_map[base_freq]

    raise NotImplementedError(f"Frequency '{standardized_freq}' is not supported.")


def validate_frequency_safety(start_date: np.datetime64, total_length: int, frequency: Frequency) -> bool:
    """
    Check if start date and frequency combination is safe for pandas datetime operations.

    This function verifies that pd.date_range(start=start_date, periods=total_length, freq=freq_str)
    will not raise an OutOfBoundsDatetime error, accounting for pandas' datetime bounds
    (1677-09-21 to 2262-04-11) and realistic frequency limitations.

    Args:
        start_date: The proposed start date for the time series
        total_length: Total length of the time series
        frequency: The frequency of the time series

    Returns:
        True if the combination is safe, False otherwise
    """
    try:
        # Get the pandas frequency string
        freq_str = frequency.to_pandas_freq(for_date_range=True)

        # Convert numpy datetime64 to pandas Timestamp for date_range
        start_pd = pd.Timestamp(start_date)

        # Check if start date is within pandas' valid datetime range
        if start_pd < pd.Timestamp.min or start_pd > pd.Timestamp.max:
            return False

        # Check maximum length constraints
        max_length = frequency.get_max_safe_length()
        if total_length > max_length:
            return False

        # For low frequencies, be extra conservative
        if frequency.is_low_frequency():
            if frequency == Frequency.A and total_length > 500:  # Max ~500 years
                return False
            elif frequency == Frequency.Q and total_length > 2000:  # Max ~500 years
                return False
            elif frequency == Frequency.M and total_length > 6000:  # Max ~500 years
                return False

        # Calculate approximate end date
        days_per_period = frequency.get_days_per_period()
        approx_days = total_length * days_per_period

        # For annual/quarterly frequencies, add extra safety margin
        if frequency in [Frequency.A, Frequency.Q]:
            approx_days *= 1.1  # 10% safety margin

        end_date = start_pd + pd.Timedelta(days=approx_days)

        # Check if end date is within pandas' valid datetime range
        if end_date < pd.Timestamp.min or end_date > pd.Timestamp.max:
            return False

        # Try to create the date range as final validation
        pd.date_range(start=start_pd, periods=total_length, freq=freq_str)
        return True

    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
        return False


# ============================================================================
# Frequency Selection Utilities
# ============================================================================


def select_safe_random_frequency(total_length: int, rng: Generator) -> Frequency:
    """
    Select a random frequency suitable for a given total length of a time series,
    based on actual GIFT eval dataset patterns and distributions.

    The selection logic:
    1. Filters frequencies that can handle the given total_length
    2. Applies base weights derived from actual GIFT eval frequency distribution
    3. Strongly boosts frequencies that are in their optimal length ranges
    4. Handles edge cases gracefully with fallbacks

    Args:
        total_length: The total length of the time series (history + future)
        rng: A numpy random number generator instance

    Returns:
        A randomly selected frequency that matches GIFT eval patterns
    """
    # Find valid frequencies and calculate weighted scores
    valid_frequencies = []
    frequency_scores = []

    for freq in Frequency:
        # Check basic timestamp overflow limits
        max_allowed = freq.get_max_safe_length()
        if total_length > max_allowed:
            continue

        # Check if frequency has defined ranges
        min_len, max_len, optimal_start, optimal_end = freq.get_length_range()

        # Must be within the frequency's realistic range
        if total_length < min_len or total_length > max_len:
            continue

        valid_frequencies.append(freq)

        # Calculate fitness score based on GIFT eval patterns
        base_weight = freq.get_gift_eval_weight()

        # Enhanced length-based fitness scoring
        if optimal_start <= total_length <= optimal_end:
            # In optimal range - very strong preference
            length_multiplier = 5.0
        else:
            # Outside optimal but within valid range - calculate penalty
            if total_length < optimal_start:
                # Below optimal range
                distance_ratio = (optimal_start - total_length) / (optimal_start - min_len)
            else:
                # Above optimal range
                distance_ratio = (total_length - optimal_end) / (max_len - optimal_end)

            # Apply graduated penalty: closer to optimal = higher score
            length_multiplier = 0.3 + 1.2 * (1.0 - distance_ratio)  # Range: 0.3-1.5

        final_score = base_weight * length_multiplier
        frequency_scores.append(final_score)

    # Handle edge cases with smart fallbacks
    if not valid_frequencies:
        # Fallback strategy based on typical length patterns
        if total_length <= 100:
            # Very short series - prefer low frequencies
            fallback_order = [
                Frequency.A,
                Frequency.Q,
                Frequency.M,
                Frequency.W,
                Frequency.D,
            ]
        elif total_length <= 1000:
            # Medium short series - prefer daily/weekly
            fallback_order = [Frequency.D, Frequency.W, Frequency.H, Frequency.M]
        else:
            # Longer series - prefer higher frequencies
            fallback_order = [Frequency.H, Frequency.D, Frequency.T15, Frequency.T5]

        for fallback_freq in fallback_order:
            max_allowed = fallback_freq.get_max_safe_length()
            if total_length <= max_allowed:
                return fallback_freq
        # Last resort
        return Frequency.D

    if len(valid_frequencies) == 1:
        return valid_frequencies[0]

    # Select based on weighted probabilities
    scores = np.array(frequency_scores)
    probabilities = scores / scores.sum()

    return rng.choice(valid_frequencies, p=probabilities)


def select_safe_start_date(
    total_length: int,
    frequency: Frequency,
    rng: Generator | None = None,
    max_retries: int = 10,
) -> np.datetime64:
    """
    Select a safe start date that ensures the entire time series (history + future)
    will not exceed pandas' datetime bounds.

    Args:
        total_length: Total length of the time series (history + future)
        frequency: Time series frequency
        rng: Random number generator instance
        max_retries: Maximum number of retry attempts

    Returns:
        A safe start date that prevents timestamp overflow

    Raises:
        ValueError: If no safe start date is found after max_retries or if the required
                   time span exceeds the available date window
    """
    if rng is None:
        rng = np.random.default_rng()

    days_per_period = frequency.get_days_per_period()

    # Calculate approximate duration in days
    total_days = total_length * days_per_period

    # Define safe bounds: ensure end date doesn't exceed BASE_END_DATE
    latest_safe_start = BASE_END_DATE - np.timedelta64(int(total_days), "D")
    earliest_safe_start = BASE_START_DATE

    # Check if the required time span exceeds the available window
    if latest_safe_start < earliest_safe_start:
        available_days = (BASE_END_DATE - BASE_START_DATE).astype("timedelta64[D]").astype(int)
        available_years = available_days / 365.25
        required_years = total_days / 365.25
        raise ValueError(
            f"Required time span ({required_years:.1f} years, {total_days:.0f} days) "
            f"exceeds available date window ({available_years:.1f} years, {available_days} days). "
            f"Reduce total_length ({total_length}) or extend the date window."
        )

    # Convert to nanoseconds for random sampling
    earliest_ns = earliest_safe_start.astype("datetime64[ns]").astype(np.int64)
    latest_ns = latest_safe_start.astype("datetime64[ns]").astype(np.int64)

    for _ in range(max_retries):
        # Uniformly sample a start date within bounds
        random_ns = rng.integers(earliest_ns, latest_ns + 1)
        start_date = np.datetime64(int(random_ns), "ns")

        # Verify safety
        if validate_frequency_safety(start_date, total_length, frequency):
            return start_date

    # Default to base start date if no safe start date is found
    return BASE_START_DATE
