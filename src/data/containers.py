from dataclasses import dataclass

import numpy as np
import torch

from src.data.frequency import Frequency


@dataclass
class BatchTimeSeriesContainer:
    """
    Container for a batch of multivariate time series data and their associated features.

    Attributes:
        history_values: Tensor of historical observations.
            Shape: [batch_size, seq_len, num_channels]
        future_values: Tensor of future observations to predict.
            Shape: [batch_size, pred_len, num_channels]
        start: Timestamp of the first history value.
            Type: List[np.datetime64]
        frequency: Frequency of the time series.
            Type: List[Frequency]
        history_mask: Optional boolean/float tensor indicating missing entries in history_values across channels.
            Shape: [batch_size, seq_len]
        future_mask: Optional boolean/float tensor indicating missing entries in future_values across channels.
            Shape: [batch_size, pred_len]
    """

    history_values: torch.Tensor
    future_values: torch.Tensor
    start: list[np.datetime64]
    frequency: list[Frequency]

    history_mask: torch.Tensor | None = None
    future_mask: torch.Tensor | None = None

    def __post_init__(self):
        """Validate all tensor shapes and consistency."""
        # --- Tensor Type Checks ---
        if not isinstance(self.history_values, torch.Tensor):
            raise TypeError("history_values must be a torch.Tensor")
        if not isinstance(self.future_values, torch.Tensor):
            raise TypeError("future_values must be a torch.Tensor")
        if not isinstance(self.start, list) or not all(isinstance(x, np.datetime64) for x in self.start):
            raise TypeError("start must be a List[np.datetime64]")
        if not isinstance(self.frequency, list) or not all(isinstance(x, Frequency) for x in self.frequency):
            raise TypeError("frequency must be a List[Frequency]")

        batch_size, seq_len, num_channels = self.history_values.shape
        pred_len = self.future_values.shape[1]

        # --- Core Shape Checks ---
        if self.future_values.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between history and future_values")
        if self.future_values.shape[2] != num_channels:
            raise ValueError("Channel size mismatch between history and future_values")

        # --- Optional Mask Checks ---
        if self.history_mask is not None:
            if not isinstance(self.history_mask, torch.Tensor):
                raise TypeError("history_mask must be a Tensor or None")
            if self.history_mask.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"Shape mismatch in history_mask: {self.history_mask.shape[:2]} vs {(batch_size, seq_len)}"
                )

        if self.future_mask is not None:
            if not isinstance(self.future_mask, torch.Tensor):
                raise TypeError("future_mask must be a Tensor or None")
            if not (
                self.future_mask.shape == (batch_size, pred_len) or self.future_mask.shape == self.future_values.shape
            ):
                raise ValueError(
                    "Shape mismatch in future_mask: "
                    f"expected {(batch_size, pred_len)} or {self.future_values.shape}, got {self.future_mask.shape}"
                )

    def to_device(self, device: torch.device, attributes: list[str] | None = None) -> None:
        """
        Move specified tensors to the target device in place.

        Args:
            device: Target device (e.g., 'cpu', 'cuda').
            attributes: Optional list of attribute names to move. If None, move all tensors.

        Raises:
            ValueError: If an invalid attribute is specified or device transfer fails.
        """
        all_tensors = {
            "history_values": self.history_values,
            "future_values": self.future_values,
            "history_mask": self.history_mask,
            "future_mask": self.future_mask,
        }

        if attributes is None:
            attributes = [k for k, v in all_tensors.items() if v is not None]

        for attr in attributes:
            if attr not in all_tensors:
                raise ValueError(f"Invalid attribute: {attr}")
            if all_tensors[attr] is not None:
                setattr(self, attr, all_tensors[attr].to(device))

    def to(self, device: torch.device, attributes: list[str] | None = None):
        """
        Alias for to_device method for consistency with PyTorch conventions.

        Args:
            device: Target device (e.g., 'cpu', 'cuda').
            attributes: Optional list of attribute names to move. If None, move all tensors.
        """
        self.to_device(device, attributes)
        return self

    @property
    def batch_size(self) -> int:
        return self.history_values.shape[0]

    @property
    def history_length(self) -> int:
        return self.history_values.shape[1]

    @property
    def future_length(self) -> int:
        return self.future_values.shape[1]

    @property
    def num_channels(self) -> int:
        return self.history_values.shape[2]


@dataclass
class TimeSeriesContainer:
    """
    Container for batch of time series data without explicit history/future split.

    This container is used for storing generated synthetic time series data where
    the entire series is treated as a single entity, typically for further processing
    or splitting into history/future components later.

    Attributes:
        values: np.ndarray of time series values.
            Shape: [batch_size, seq_len, num_channels] for multivariate series
                   [batch_size, seq_len] for univariate series
        start: List of start timestamps for each series in the batch.
            Type: List[np.datetime64], length should match batch_size
        frequency: List of frequency for each series in the batch.
            Type: List[Frequency], length should match batch_size
    """

    values: np.ndarray
    start: list[np.datetime64]
    frequency: list[Frequency]

    def __post_init__(self):
        """Validate all shapes and consistency."""
        # --- Numpy Type Checks ---
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a np.ndarray")
        if not isinstance(self.start, list) or not all(isinstance(x, np.datetime64) for x in self.start):
            raise TypeError("start must be a List[np.datetime64]")
        if not isinstance(self.frequency, list) or not all(isinstance(x, Frequency) for x in self.frequency):
            raise TypeError("frequency must be a List[Frequency]")

        # --- Shape and Length Consistency Checks ---
        if len(self.values.shape) < 2 or len(self.values.shape) > 3:
            raise ValueError(
                "values must have 2 or 3 dimensions "
                "[batch_size, seq_len] or [batch_size, seq_len, num_channels], "
                f"got shape {self.values.shape}"
            )

        batch_size = self.values.shape[0]

        if len(self.start) != batch_size:
            raise ValueError(f"Length of start ({len(self.start)}) must match batch_size ({batch_size})")
        if len(self.frequency) != batch_size:
            raise ValueError(f"Length of frequency ({len(self.frequency)}) must match batch_size ({batch_size})")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def seq_length(self) -> int:
        return self.values.shape[1]

    @property
    def num_channels(self) -> int:
        return self.values.shape[2] if len(self.values.shape) == 3 else 1
