import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast

from src.data.frequency import parse_frequency
from src.plotting.plot_timeseries import (
    plot_multivariate_timeseries,
)

logger = logging.getLogger(__name__)


def _prepare_data_for_plotting(
    input_data: dict, label_data: dict, max_context_length: int
):
    history_values = np.asarray(input_data["target"], dtype=np.float32)
    future_values = np.asarray(label_data["target"], dtype=np.float32)
    start_period = input_data["start"]

    def ensure_time_first(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        elif arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:
                return arr.T
            return arr
        else:
            return arr.reshape(arr.shape[-1], -1).T

    history_values = ensure_time_first(history_values)
    future_values = ensure_time_first(future_values)

    if max_context_length is not None and history_values.shape[0] > max_context_length:
        history_values = history_values[-max_context_length:]

    # Convert Period to Timestamp if needed
    start_timestamp = (
        start_period.to_timestamp()
        if hasattr(start_period, "to_timestamp")
        else pd.Timestamp(start_period)
    )
    return history_values, future_values, start_timestamp


def _extract_quantile_predictions(
    forecast,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    def ensure_2d_time_first(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        elif arr.ndim == 2:
            return arr
        else:
            return arr.reshape(arr.shape[0], -1)

    if isinstance(forecast, QuantileForecast):
        try:
            median_pred = forecast.quantile(0.5)
            try:
                lower_bound = forecast.quantile(0.1)
                upper_bound = forecast.quantile(0.9)
            except (KeyError, ValueError):
                lower_bound = None
                upper_bound = None
            median_pred = ensure_2d_time_first(median_pred)
            lower_bound = ensure_2d_time_first(lower_bound)
            upper_bound = ensure_2d_time_first(upper_bound)
            return median_pred, lower_bound, upper_bound
        except Exception:
            try:
                median_pred = forecast.quantile(0.5)
                median_pred = ensure_2d_time_first(median_pred)
                return median_pred, None, None
            except Exception:
                return None, None, None
    else:
        try:
            samples = forecast.samples
            if samples.ndim == 1:
                median_pred = samples
            elif samples.ndim == 2:
                if samples.shape[0] == 1:
                    median_pred = samples[0]
                else:
                    median_pred = np.median(samples, axis=0)
            elif samples.ndim == 3:
                median_pred = np.median(samples, axis=0)
            else:
                median_pred = samples[0] if len(samples) > 0 else samples
            median_pred = ensure_2d_time_first(median_pred)
            return median_pred, None, None
        except Exception:
            return None, None, None


def _create_plot(
    input_data: dict,
    label_data: dict,
    forecast,
    dataset_full_name: str,
    dataset_freq: str,
    max_context_length: int,
    title: Optional[str] = None,
):
    try:
        history_values, future_values, start_timestamp = _prepare_data_for_plotting(
            input_data, label_data, max_context_length
        )
        median_pred, lower_bound, upper_bound = _extract_quantile_predictions(forecast)
        if median_pred is None:
            logger.warning(f"Could not extract predictions for {dataset_full_name}")
            return None

        def ensure_compatible_shape(pred_arr, target_arr):
            if pred_arr is None:
                return None
            pred_arr = np.asarray(pred_arr)
            target_arr = np.asarray(target_arr)
            if pred_arr.ndim == 1:
                pred_arr = pred_arr.reshape(-1, 1)
            if target_arr.ndim == 1:
                target_arr = target_arr.reshape(-1, 1)
            if pred_arr.shape != target_arr.shape:
                if pred_arr.shape[0] == target_arr.shape[0]:
                    if pred_arr.shape[1] == 1 and target_arr.shape[1] > 1:
                        pred_arr = np.broadcast_to(pred_arr, target_arr.shape)
                    elif pred_arr.shape[1] > 1 and target_arr.shape[1] == 1:
                        pred_arr = pred_arr[:, :1]
                elif pred_arr.shape[1] == target_arr.shape[1]:
                    min_time = min(pred_arr.shape[0], target_arr.shape[0])
                    pred_arr = pred_arr[:min_time]
                else:
                    if pred_arr.T.shape == target_arr.shape:
                        pred_arr = pred_arr.T
                    else:
                        if pred_arr.size >= target_arr.shape[0]:
                            pred_arr = pred_arr.flatten()[
                                : target_arr.shape[0]
                            ].reshape(-1, 1)
                            if target_arr.shape[1] > 1:
                                pred_arr = np.broadcast_to(pred_arr, target_arr.shape)
            return pred_arr

        median_pred = ensure_compatible_shape(median_pred, future_values)
        lower_bound = ensure_compatible_shape(lower_bound, future_values)
        upper_bound = ensure_compatible_shape(upper_bound, future_values)

        title = title or f"GIFT-Eval: {dataset_full_name}"
        frequency = parse_frequency(dataset_freq)
        fig = plot_multivariate_timeseries(
            history_values=history_values,
            future_values=future_values,
            predicted_values=median_pred,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            start=start_timestamp,
            frequency=frequency,
            title=title,
            show=False,
        )
        return fig
    except Exception as e:
        logger.warning(f"Failed to create plot for {dataset_full_name}: {e}")
        return None


def create_plots_for_dataset(
    forecasts: List,
    test_data,
    dataset_metadata,
    max_plots: int,
    max_context_length: int,
) -> List[Tuple[object, str]]:
    input_data_list = list(test_data.input)
    label_data_list = list(test_data.label)
    num_plots = min(len(forecasts), max_plots)
    logger.info(
        f"Creating {num_plots} plots for {getattr(dataset_metadata, 'full_name', str(dataset_metadata))}"
    )

    figures_with_names: List[Tuple[object, str]] = []
    for i in range(num_plots):
        try:
            forecast = forecasts[i]
            input_data = input_data_list[i]
            label_data = label_data_list[i]
            title = (
                f"GIFT-Eval: {dataset_metadata.full_name} - Window {i + 1}/{num_plots}"
                if hasattr(dataset_metadata, "full_name")
                else f"Window {i + 1}/{num_plots}"
            )
            fig = _create_plot(
                input_data=input_data,
                label_data=label_data,
                forecast=forecast,
                dataset_full_name=getattr(dataset_metadata, "full_name", "dataset"),
                dataset_freq=getattr(dataset_metadata, "freq", "D"),
                max_context_length=max_context_length,
                title=title,
            )
            if fig is not None:
                filename = (
                    f"{getattr(dataset_metadata, 'freq', 'D')}_window_{i + 1:03d}.png"
                )
                figures_with_names.append((fig, filename))
        except Exception as e:
            logger.warning(f"Error creating plot for window {i + 1}: {e}")
            continue
    return figures_with_names
