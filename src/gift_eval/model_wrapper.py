import logging
from typing import Iterator, List, Optional

import numpy as np
import torch
import yaml
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.containers import BatchTimeSeriesContainer
from src.data.frequency import parse_frequency
from src.data.scalers import RobustScaler
from src.models.model import TimeSeriesModel
from src.utils.utils import device

logger = logging.getLogger(__name__)


class TimeSeriesPredictor(Predictor):
    """
    Unified predictor for TimeSeriesModel supporting both in-memory and file-based construction.

    Use classmethods `from_model` and `from_paths` to construct instances.
    Provides `set_dataset_context` to adjust dataset-specific parameters without reloading the model.
    """

    def __init__(
        self,
        model: TimeSeriesModel,
        config: dict,
        ds_prediction_length: int,
        ds_freq: str,
        batch_size: int = 32,
        max_context_length: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        # Dataset-specific context (can be updated per dataset/term)
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.debug = debug

        # Persistent model/config (unwrap DDP if needed)
        self.model = model.module if isinstance(model, DDP) else model
        self.model.eval()
        self.config = config

        # Initialize scaler (using same type as model)
        scaler_type = self.config.get("TimeSeriesModel", {}).get(
            "scaler", "custom_robust"
        )
        epsilon = self.config.get("TimeSeriesModel", {}).get("epsilon", 1e-3)
        if scaler_type == "custom_robust":
            self.scaler = RobustScaler(epsilon=epsilon)
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def set_dataset_context(
        self,
        prediction_length: Optional[int] = None,
        freq: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_context_length: Optional[int] = None,
    ) -> None:
        """Update lightweight dataset-specific attributes without reloading the model."""
        if prediction_length is not None:
            self.ds_prediction_length = prediction_length
        if freq is not None:
            self.ds_freq = freq
        if batch_size is not None:
            self.batch_size = batch_size
        if max_context_length is not None:
            self.max_context_length = max_context_length

    @classmethod
    def from_model(
        cls,
        model: TimeSeriesModel,
        config: dict,
        ds_prediction_length: int,
        ds_freq: str,
        batch_size: int = 32,
        max_context_length: Optional[int] = None,
        debug: bool = False,
    ) -> "TimeSeriesPredictor":
        return cls(
            model=model,
            config=config,
            ds_prediction_length=ds_prediction_length,
            ds_freq=ds_freq,
            batch_size=batch_size,
            max_context_length=max_context_length,
            debug=debug,
        )

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        config_path: str,
        ds_prediction_length: int,
        ds_freq: str,
        batch_size: int = 32,
        max_context_length: Optional[int] = None,
        debug: bool = False,
    ) -> "TimeSeriesPredictor":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = cls._load_model_from_path(config=config, model_path=model_path)
        return cls(
            model=model,
            config=config,
            ds_prediction_length=ds_prediction_length,
            ds_freq=ds_freq,
            batch_size=batch_size,
            max_context_length=max_context_length,
            debug=debug,
        )

    @staticmethod
    def _load_model_from_path(config: dict, model_path: str) -> TimeSeriesModel:
        try:
            model = TimeSeriesModel(**config["TimeSeriesModel"]).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

    def predict(self, test_data_input) -> Iterator[QuantileForecast]:
        """
        Generate forecasts for the test data.

        Args:
            test_data_input: Iterator of gluonts DataEntry objects

        Returns:
            Iterable of QuantileForecast objects
        """
        if hasattr(test_data_input, "__iter__") and not isinstance(
            test_data_input, list
        ):
            test_data_input = list(test_data_input)
        logger.debug(f"Processing {len(test_data_input)} time series")

        # Group series by their effective length (after optional truncation),
        # then process each uniform-length group in sub-batches up to batch_size.
        def _effective_length(entry) -> int:
            target = entry["target"]
            if target.ndim == 1:
                seq_len = len(target)
            else:
                # target shape is [num_channels, seq_len]
                seq_len = target.shape[1]
            if self.max_context_length is not None:
                seq_len = min(seq_len, self.max_context_length)
            return seq_len

        length_to_items: dict[int, List[tuple[int, object]]] = {}
        for idx, entry in enumerate(test_data_input):
            L = _effective_length(entry)
            length_to_items.setdefault(L, []).append((idx, entry))

        total = len(test_data_input)
        ordered_results: List[Optional[QuantileForecast]] = [None] * total

        for _, items in length_to_items.items():
            for i in range(0, len(items), self.batch_size):
                chunk = items[i : i + self.batch_size]
                entries = [e for (_orig_idx, e) in chunk]
                batch_forecasts = self._predict_batch(entries)
                for f_idx, (orig_idx, _e) in enumerate(chunk):
                    ordered_results[orig_idx] = batch_forecasts[f_idx]

        # All results should be populated
        return ordered_results  # type: ignore[return-value]

    def _predict_batch(self, test_data_batch: List) -> List[QuantileForecast]:
        """Generate predictions for a batch of time series."""
        logger.debug(f"Processing batch of size: {len(test_data_batch)}")

        try:
            # Convert gluonts data to BatchTimeSeriesContainer
            batch_container = self._convert_to_batch_container(test_data_batch)

            # Autocast only when running on CUDA
            if isinstance(device, torch.device):
                device_type = device.type
            else:
                device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
            enable_autocast = device_type == "cuda"

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=enable_autocast
            ):
                with torch.no_grad():
                    model_output = self.model(batch_container, drop_enc_allow=False)

            # Convert predictions to QuantileForecast objects
            forecasts = self._convert_to_forecasts(
                model_output, test_data_batch, batch_container
            )

            logger.debug(f"Generated {len(forecasts)} forecasts")
            return forecasts

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def _convert_to_batch_container(
        self, test_data_batch: List
    ) -> BatchTimeSeriesContainer:
        """Convert gluonts test data to BatchTimeSeriesContainer."""
        batch_size = len(test_data_batch)

        # Extract data from test entries (all series in this batch must have equal length)
        history_values_list = []
        start_dates = []
        frequencies = []

        for entry in test_data_batch:
            target = entry["target"]

            # Handle both univariate and multivariate cases
            if target.ndim == 1:
                # Univariate: reshape to [seq_len, 1]
                target = target.reshape(-1, 1)
            else:
                # Multivariate: assume shape is [num_channels, seq_len] -> transpose to [seq_len, num_channels]
                target = target.T

            # Apply context length limit if specified
            if (
                self.max_context_length is not None
                and len(target) > self.max_context_length
            ):
                target = target[-self.max_context_length :]

            history_values_list.append(target)
            start_dates.append(entry["start"].to_timestamp().to_datetime64())
            frequencies.append(parse_frequency(entry["freq"]))

        # Stack sequences directly (no padding) -> shapes are uniform by construction
        history_values_np = np.stack(history_values_list, axis=0)
        num_channels = history_values_np.shape[2]

        # Convert to tensors
        history_values = torch.tensor(
            history_values_np, dtype=torch.float32, device=device
        )

        # Create future values tensor (empty for prediction)
        future_values = torch.zeros(
            (batch_size, self.ds_prediction_length, num_channels),
            dtype=torch.float32,
            device=device,
        )

        return BatchTimeSeriesContainer(
            history_values=history_values,
            future_values=future_values,
            start=start_dates,
            frequency=frequencies,
        )

    def _convert_to_forecasts(
        self,
        model_output: dict,
        test_data_batch: List,
        batch_container: BatchTimeSeriesContainer,
    ) -> List[QuantileForecast]:
        """Convert model predictions to QuantileForecast objects."""
        predictions = model_output[
            "result"
        ]  # Shape: [batch_size, pred_len, num_channels] or [batch_size, pred_len, num_channels, num_quantiles]
        scale_statistics = model_output["scale_statistics"]

        # Apply inverse scaling to get predictions in original scale
        if predictions.ndim == 4:
            # Quantile predictions: [batch_size, pred_len, num_channels, num_quantiles]
            predictions_unscaled = self.scaler.inverse_scale(
                predictions, scale_statistics
            )
            is_quantile = True
            quantile_levels = self.model.quantiles
        else:
            # Point predictions: [batch_size, pred_len, num_channels]
            predictions_unscaled = self.scaler.inverse_scale(
                predictions, scale_statistics
            )
            is_quantile = False
            quantile_levels = [0.5]  # Treat as median forecast

        forecasts: List[QuantileForecast] = []
        for i, entry in enumerate(test_data_batch):
            # Get prediction start date
            history_length = int(batch_container.history_values.shape[1])
            start_date = entry["start"]
            forecast_start = start_date + history_length

            if is_quantile:
                # Handle quantile forecasts
                pred_array = (
                    predictions_unscaled[i].cpu().numpy()
                )  # [pred_len, num_channels, num_quantiles]

                if pred_array.shape[1] == 1:
                    # Univariate case: [pred_len, 1, num_quantiles] -> [pred_len, num_quantiles]
                    pred_array = pred_array.squeeze(1)
                    forecast_arrays = pred_array.T  # [num_quantiles, pred_len]
                else:
                    # Multivariate case: [pred_len, num_channels, num_quantiles] -> [num_quantiles, pred_len, num_channels]
                    forecast_arrays = pred_array.transpose(2, 0, 1)

                forecast = QuantileForecast(
                    forecast_arrays=forecast_arrays,
                    forecast_keys=[str(q) for q in quantile_levels],
                    start_date=forecast_start,
                )
            else:
                # Handle point forecasts
                pred_array = (
                    predictions_unscaled[i].cpu().numpy()
                )  # [pred_len, num_channels]

                if pred_array.shape[1] == 1:
                    # Univariate case: [pred_len, 1] -> [pred_len]
                    pred_array = pred_array.squeeze(1)
                    forecast_arrays = pred_array.reshape(1, -1)  # [1, pred_len]
                else:
                    # Multivariate case: [pred_len, num_channels] -> [1, pred_len, num_channels]
                    forecast_arrays = pred_array.reshape(1, *pred_array.shape)

                forecast = QuantileForecast(
                    forecast_arrays=forecast_arrays,
                    forecast_keys=["0.5"],
                    start_date=forecast_start,
                )

            forecasts.append(forecast)

        return forecasts
