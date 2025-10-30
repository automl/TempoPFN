import random
from typing import Optional, Tuple, Union


def sample_future_length(
    range: Union[Tuple[int, int], str] = "gift_eval",
    total_length: Optional[int] = None,
) -> int:
    """
    Sample a forecast length.

    - If `range` is a tuple, uniformly sample in [min, max]. When `total_length` is
      provided, enforce a cap so the result is at most floor(0.45 * total_length).
    - If `range` is "gift_eval", sample from a pre-defined weighted set. When
      `total_length` is provided, filter out candidates greater than
      floor(0.45 * total_length) before sampling.
    """
    # Compute the cap when total_length is provided
    cap: Optional[int] = None
    if total_length is not None:
        cap = max(1, int(0.45 * int(total_length)))

    if isinstance(range, tuple):
        min_len, max_len = range
        if cap is not None:
            effective_max_len = min(max_len, cap)
            # Ensure valid bounds
            if min_len > effective_max_len:
                return effective_max_len
            return random.randint(min_len, effective_max_len)
        return random.randint(min_len, max_len)
    elif range == "gift_eval":
        # Gift eval forecast lengths with their frequencies
        GIFT_EVAL_FORECAST_LENGTHS = {
            48: 5,
            720: 38,
            480: 38,
            30: 3,
            300: 16,
            8: 2,
            120: 3,
            450: 8,
            80: 8,
            12: 2,
            900: 10,
            180: 3,
            600: 10,
            60: 3,
            210: 3,
            195: 3,
            140: 3,
            130: 3,
            14: 1,
            18: 1,
            13: 1,
            6: 1,
        }

        lengths = list(GIFT_EVAL_FORECAST_LENGTHS.keys())
        weights = list(GIFT_EVAL_FORECAST_LENGTHS.values())

        if cap is not None:
            filtered = [
                (length_candidate, weight)
                for length_candidate, weight in zip(lengths, weights)
                if length_candidate <= cap
            ]
            if filtered:
                lengths, weights = zip(*filtered)
                lengths = list(lengths)
                weights = list(weights)

        return random.choices(lengths, weights=weights)[0]
    else:
        raise ValueError(f"Invalid range: {range}")
