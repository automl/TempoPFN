from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class BaseScaler(ABC):
    """
    Abstract base class for time series scalers.

    Defines the interface for scaling multivariate time series data with support
    for masked values and channel-wise scaling.
    """

    @abstractmethod
    def compute_statistics(
        self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scaling statistics from historical data.
        """
        pass

    @abstractmethod
    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply scaling transformation to data.
        """
        pass

    @abstractmethod
    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse scaling transformation to recover original scale.
        """
        pass


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and IQR for normalization.
    """

    def __init__(self, epsilon: float = 1e-6, min_scale: float = 1e-3):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.epsilon = epsilon
        self.min_scale = min_scale

    def compute_statistics(
        self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute median and IQR statistics from historical data with improved numerical stability.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        medians = torch.zeros(batch_size, 1, num_channels, device=device)
        iqrs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                if len(valid_data) == 0:
                    continue

                valid_data = valid_data[torch.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue

                median_val = torch.median(valid_data)
                medians[b, 0, c] = median_val

                if len(valid_data) > 1:
                    try:
                        q75 = torch.quantile(valid_data, 0.75)
                        q25 = torch.quantile(valid_data, 0.25)
                        iqr_val = q75 - q25
                        iqr_val = torch.max(
                            iqr_val, torch.tensor(self.min_scale, device=device)
                        )
                        iqrs[b, 0, c] = iqr_val
                    except Exception:
                        std_val = torch.std(valid_data)
                        iqrs[b, 0, c] = torch.max(
                            std_val, torch.tensor(self.min_scale, device=device)
                        )
                else:
                    iqrs[b, 0, c] = self.min_scale

        return {"median": medians, "iqr": iqrs}

    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply robust scaling: (data - median) / (iqr + epsilon).
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )
        scaled_data = (data - median) / denominator
        scaled_data = torch.clamp(scaled_data, -50.0, 50.0)

        return scaled_data

    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse robust scaling, now compatible with 3D or 4D tensors.
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )

        if scaled_data.ndim == 4:
            denominator = denominator.unsqueeze(-1)
            median = median.unsqueeze(-1)

        return scaled_data * denominator + median


class MinMaxScaler(BaseScaler):
    """
    Min-Max scaler that normalizes data to the range [-1, 1].
    """

    def __init__(self, epsilon: float = 1e-8):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def compute_statistics(
        self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute min and max statistics from historical data.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        mins = torch.zeros(batch_size, 1, num_channels, device=device)
        maxs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                if len(valid_data) == 0:
                    continue

                min_val = torch.min(valid_data)
                max_val = torch.max(valid_data)

                mins[b, 0, c] = min_val
                maxs[b, 0, c] = max_val

                if torch.abs(max_val - min_val) < self.epsilon:
                    maxs[b, 0, c] = min_val + 1.0

        return {"min": mins, "max": maxs}

    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply min-max scaling to range [-1, 1].
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        normalized = (data - min_val) / (max_val - min_val + self.epsilon)
        return normalized * 2.0 - 1.0

    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse min-max scaling, now compatible with 3D or 4D tensors.
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        if scaled_data.ndim == 4:
            min_val = min_val.unsqueeze(-1)
            max_val = max_val.unsqueeze(-1)

        normalized = (scaled_data + 1.0) / 2.0
        return normalized * (max_val - min_val + self.epsilon) + min_val


class MeanScaler(BaseScaler):
    """
    A scaler that centers the data by subtracting the channel-wise mean.

    This scaler only performs centering and does not affect the scale of the data.
    """

    def compute_statistics(
            self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mean for each channel from historical data.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        # Initialize a tensor to store the mean for each channel in each batch item
        means = torch.zeros(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                # Use the mask to select only valid (observed) data points
                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                # Skip if there's no valid data for this channel
                if len(valid_data) == 0:
                    continue

                # Filter out non-finite values like NaN or Inf before computing
                valid_data = valid_data[torch.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue

                # Compute the mean and store it
                means[b, 0, c] = torch.mean(valid_data)

        return {"mean": means}

    def scale(
            self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply mean centering: data - mean.
        """
        mean = statistics["mean"]
        return data - mean

    def inverse_scale(
            self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse mean centering: scaled_data + mean.

        Handles both 3D (e.g., training input) and 4D (e.g., model output samples) tensors.
        """
        mean = statistics["mean"]

        # Adjust shape for 4D tensors (batch, seq_len, channels, samples)
        if scaled_data.ndim == 4:
            mean = mean.unsqueeze(-1)

        return scaled_data + mean


class MedianScaler(BaseScaler):
    """
    A scaler that centers the data by subtracting the channel-wise median.

    This scaler only performs centering and does not affect the scale of the data.
    It is more robust to outliers than the MeanScaler.
    """

    def compute_statistics(
            self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the median for each channel from historical data.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        # Initialize a tensor to store the median for each channel in each batch item
        medians = torch.zeros(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                # Use the mask to select only valid (observed) data points
                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                # Skip if there's no valid data for this channel
                if len(valid_data) == 0:
                    continue

                # Filter out non-finite values like NaN or Inf before computing
                valid_data = valid_data[torch.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue

                # Compute the median and store it
                medians[b, 0, c] = torch.median(valid_data)

        return {"median": medians}

    def scale(
            self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply median centering: data - median.
        """
        median = statistics["median"]
        return data - median

    def inverse_scale(
            self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse median centering: scaled_data + median.

        Handles both 3D (e.g., training input) and 4D (e.g., model output samples) tensors.
        """
        median = statistics["median"]

        # Adjust shape for 4D tensors (batch, seq_len, channels, samples)
        if scaled_data.ndim == 4:
            median = median.unsqueeze(-1)

        return scaled_data + median
