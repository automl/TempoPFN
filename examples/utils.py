import logging
import os
import urllib.request
from typing import List

import numpy as np
import torch
import yaml

from src.data.containers import BatchTimeSeriesContainer
from src.models.model import TimeSeriesModel
from src.plotting.plot_timeseries import plot_from_container

logger = logging.getLogger(__name__)


def load_model(
    config_path: str, model_path: str, device: torch.device
) -> TimeSeriesModel:
    """Load the TimeSeriesModel from config and checkpoint."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = TimeSeriesModel(**config["TimeSeriesModel"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Successfully loaded TimeSeriesModel from {model_path} on {device}")
    return model


def download_checkpoint_if_needed(url: str, target_dir: str = "models") -> str:
    """Download checkpoint from URL into target_dir if not present and return its path.

    Ensures direct download for Dropbox links by forcing dl=1.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "checkpoint.pth")

    # Normalize Dropbox URL to force direct download
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("dl=0", "dl=1")

    if not os.path.exists(target_path):
        logger.info(f"Downloading checkpoint from {url} to {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        logger.info("Checkpoint downloaded successfully.")
    else:
        logger.info(f"Using existing checkpoint at {target_path}")

    return target_path


def plot_with_library(
    container: BatchTimeSeriesContainer,
    predictions_np: np.ndarray,  # [B, P, N, Q]
    model_quantiles: List[float] | None,
    output_dir: str = "outputs",
    show_plots: bool = True,
    save_plots: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    batch_size = container.batch_size
    for i in range(batch_size):
        output_file = (
            os.path.join(output_dir, f"sine_wave_prediction_sample_{i + 1}.png")
            if save_plots
            else None
        )
        plot_from_container(
            batch=container,
            sample_idx=i,
            predicted_values=predictions_np,
            model_quantiles=model_quantiles,
            title=f"Sine Wave Time Series Prediction - Sample {i + 1}",
            output_file=output_file,
            show=show_plots,
        )


def run_inference_and_plot(
    model: TimeSeriesModel,
    container: BatchTimeSeriesContainer,
    output_dir: str = "outputs",
    use_bfloat16: bool = True,
) -> None:
    """Run model inference with optional bfloat16 and plot using shared utilities."""
    device_type = "cuda" if (container.history_values.device.type == "cuda") else "cpu"
    autocast_enabled = use_bfloat16 and device_type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ),
    ):
        model_output = model(container)

    preds_full = model_output["result"].to(torch.float32)
    if hasattr(model, "scaler") and "scale_statistics" in model_output:
        preds_full = model.scaler.inverse_scale(
            preds_full, model_output["scale_statistics"]
        )

    preds_np = preds_full.detach().cpu().numpy()
    model_quantiles = (
        model.quantiles if getattr(model, "loss_type", None) == "quantile" else None
    )
    plot_with_library(
        container=container,
        predictions_np=preds_np,
        model_quantiles=model_quantiles,
        output_dir=output_dir,
        show_plots=True,
        save_plots=True,
    )
