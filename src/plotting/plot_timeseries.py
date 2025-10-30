import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from matplotlib.figure import Figure

from src.data.containers import BatchTimeSeriesContainer
from src.data.frequency import Frequency

logger = logging.getLogger(__name__)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    pred_tensor = torch.from_numpy(y_pred).float()
    true_tensor = torch.from_numpy(y_true).float()
    return torchmetrics.SymmetricMeanAbsolutePercentageError()(
        pred_tensor, true_tensor
    ).item()


def _create_date_ranges(
    start: Optional[Union[np.datetime64, pd.Timestamp]],
    frequency: Optional[Union[Frequency, str]],
    history_length: int,
    prediction_length: int,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Create date ranges for history and future periods."""
    if start is not None and frequency is not None:
        start_timestamp = pd.Timestamp(start)
        pandas_freq = frequency.to_pandas_freq(for_date_range=True)

        history_dates = pd.date_range(
            start=start_timestamp, periods=history_length, freq=pandas_freq
        )

        if prediction_length > 0:
            next_timestamp = history_dates[-1] + pd.tseries.frequencies.to_offset(
                pandas_freq
            )
            future_dates = pd.date_range(
                start=next_timestamp, periods=prediction_length, freq=pandas_freq
            )
        else:
            future_dates = pd.DatetimeIndex([])
    else:
        # Fallback to default daily frequency
        history_dates = pd.date_range(
            end=pd.Timestamp.now(), periods=history_length, freq="D"
        )

        if prediction_length > 0:
            future_dates = pd.date_range(
                start=history_dates[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="D",
            )
        else:
            future_dates = pd.DatetimeIndex([])

    return history_dates, future_dates


def _plot_single_channel(
    ax: plt.Axes,
    channel_idx: int,
    history_dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    history_values: np.ndarray,
    future_values: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> None:
    """Plot a single channel's time series data."""
    # Plot history
    ax.plot(
        history_dates, history_values[:, channel_idx], color="black", label="History"
    )

    # Plot ground truth future
    if future_values is not None:
        ax.plot(
            future_dates,
            future_values[:, channel_idx],
            color="blue",
            label="Ground Truth",
        )

    # Plot predictions
    if predicted_values is not None:
        ax.plot(
            future_dates,
            predicted_values[:, channel_idx],
            color="orange",
            linestyle="--",
            label="Prediction (Median)",
        )

    # Plot uncertainty band
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            future_dates,
            lower_bound[:, channel_idx],
            upper_bound[:, channel_idx],
            color="orange",
            alpha=0.2,
            label="Uncertainty Band",
        )

    ax.set_title(f"Channel {channel_idx + 1}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


def _setup_figure(num_channels: int) -> Tuple[Figure, List[plt.Axes]]:
    """Create and configure the matplotlib figure and axes."""
    fig, axes = plt.subplots(
        num_channels, 1, figsize=(15, 3 * num_channels), sharex=True
    )
    if num_channels == 1:
        axes = [axes]
    return fig, axes


def _finalize_plot(
    fig: Figure,
    axes: List[plt.Axes],
    title: Optional[str] = None,
    smape_value: Optional[float] = None,
    output_file: Optional[str] = None,
    show: bool = True,
) -> None:
    """Add legend, title, and save/show the plot."""
    # Create legend from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # Set title with optional SMAPE
    if title:
        if smape_value is not None:
            title = f"{title} | SMAPE: {smape_value:.4f}"
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)

    # Save and/or show
    if output_file:
        plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    future_values: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    start: Optional[Union[np.datetime64, pd.Timestamp]] = None,
    frequency: Optional[Union[Frequency, str]] = None,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    show: bool = True,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> Figure:
    """Plot a multivariate time series with history, future, predictions, and uncertainty bands."""
    # Calculate SMAPE if both predicted and true values are available
    smape_value = None
    if predicted_values is not None and future_values is not None:
        try:
            smape_value = calculate_smape(future_values, predicted_values)
        except Exception as e:
            logger.warning(f"Failed to calculate SMAPE: {str(e)}")

    # Extract dimensions
    num_channels = history_values.shape[1]
    history_length = history_values.shape[0]
    prediction_length = (
        predicted_values.shape[0]
        if predicted_values is not None
        else (future_values.shape[0] if future_values is not None else 0)
    )

    # Create date ranges
    history_dates, future_dates = _create_date_ranges(
        start, frequency, history_length, prediction_length
    )

    # Setup figure
    fig, axes = _setup_figure(num_channels)

    # Plot each channel
    for i in range(num_channels):
        _plot_single_channel(
            ax=axes[i],
            channel_idx=i,
            history_dates=history_dates,
            future_dates=future_dates,
            history_values=history_values,
            future_values=future_values,
            predicted_values=predicted_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    # Finalize plot
    _finalize_plot(fig, axes, title, smape_value, output_file, show)

    return fig


def _extract_quantile_predictions(
    predicted_values: np.ndarray,
    model_quantiles: List[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract median, lower, and upper bound predictions from quantile output."""
    try:
        median_idx = model_quantiles.index(0.5)
        lower_idx = model_quantiles.index(0.1)
        upper_idx = model_quantiles.index(0.9)

        median_preds = predicted_values[..., median_idx]
        lower_bound = predicted_values[..., lower_idx]
        upper_bound = predicted_values[..., upper_idx]

        return median_preds, lower_bound, upper_bound
    except (ValueError, IndexError):
        logger.warning(
            "Could not find 0.1, 0.5, 0.9 quantiles for plotting. Using median of available quantiles."
        )
        median_preds = predicted_values[..., predicted_values.shape[-1] // 2]
        return median_preds, None, None


def plot_from_container(
    batch: BatchTimeSeriesContainer,
    sample_idx: int,
    predicted_values: Optional[np.ndarray] = None,
    model_quantiles: Optional[List[float]] = None,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Plot a single sample from a BatchTimeSeriesContainer with proper quantile handling."""
    # Extract data for the specific sample
    history_values = batch.history_values[sample_idx].cpu().numpy()
    future_values = batch.future_values[sample_idx].cpu().numpy()

    # Process predictions
    if predicted_values is not None:
        # Handle batch vs single sample predictions
        if predicted_values.ndim >= 3 or (
            predicted_values.ndim == 2
            and predicted_values.shape[0] > future_values.shape[0]
        ):
            sample_preds = predicted_values[sample_idx]
        else:
            sample_preds = predicted_values

        # Extract quantile information if available
        if model_quantiles:
            median_preds, lower_bound, upper_bound = _extract_quantile_predictions(
                sample_preds, model_quantiles
            )
        else:
            median_preds = sample_preds
            lower_bound = None
            upper_bound = None
    else:
        median_preds = None
        lower_bound = None
        upper_bound = None

    # Create the plot
    return plot_multivariate_timeseries(
        history_values=history_values,
        future_values=future_values,
        predicted_values=median_preds,
        start=batch.start[sample_idx],
        frequency=batch.frequency[sample_idx],
        title=title,
        output_file=output_file,
        show=show,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
