from datetime import date

import numpy as np

DEFAULT_START_DATE = date(1700, 1, 1)
DEFAULT_END_DATE = date(2200, 1, 1)
BASE_START_DATE = np.datetime64(DEFAULT_START_DATE)
BASE_END_DATE = np.datetime64(DEFAULT_END_DATE)

# Maximum years to prevent timestamp overflow
MAX_YEARS = 500

LENGTH_CHOICES = [128, 256, 512, 1024, 1536, 2048]

DEFAULT_NAN_STATS_PATH: str = "./data/nan_stats.json"

LENGTH_WEIGHTS: dict[int, float] = {
    128: 0.05,
    256: 0.10,
    512: 0.10,
    1024: 0.10,
    1536: 0.15,
    2048: 0.50,
}
