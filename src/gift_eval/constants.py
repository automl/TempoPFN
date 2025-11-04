import json
import logging
import os
from pathlib import Path

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)


logger = logging.getLogger(__name__)


# Environment setup
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# Use absolute path relative to the project root
_MODULE_DIR = Path(__file__).parent.parent.parent  # Goes to project root
DATASET_PROPERTIES_PATH = _MODULE_DIR / "data" / "dataset_properties.json"


try:
    with open(DATASET_PROPERTIES_PATH, "r") as f:
        DATASET_PROPERTIES = json.load(f)
except Exception as exc:  # pragma: no cover - logging path
    DATASET_PROPERTIES = {}
    logger.warning(
        "Could not load dataset properties from %s: %s. Domain and num_variates will fall back to defaults.",
        DATASET_PROPERTIES_PATH,
        exc,
    )


# Datasets
SHORT_DATASETS = (
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
)

MED_LONG_DATASETS = (
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
)

# Preserve insertion order from SHORT_DATASETS followed by MED_LONG_DATASETS
ALL_DATASETS = list(dict.fromkeys(SHORT_DATASETS + MED_LONG_DATASETS))


# Evaluation terms
TERMS = ("short", "medium", "long")


# Pretty names mapping (following GIFT eval standard)
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


METRICS = (
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
)


STANDARD_METRIC_NAMES = (
    "MSE[mean]",
    "MSE[0.5]",
    "MAE[0.5]",
    "MASE[0.5]",
    "MAPE[0.5]",
    "sMAPE[0.5]",
    "MSIS",
    "RMSE[mean]",
    "NRMSE[mean]",
    "ND[0.5]",
    "mean_weighted_sum_quantile_loss",
)


__all__ = [
    "ALL_DATASETS",
    "DATASET_PROPERTIES",
    "DATASET_PROPERTIES_PATH",
    "MED_LONG_DATASETS",
    "METRICS",
    "PRETTY_NAMES",
    "SHORT_DATASETS",
    "STANDARD_METRIC_NAMES",
    "TERMS",
]
