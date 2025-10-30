import torch
import torch.nn as nn
from fla.modules import GatedMLP

from src.data.containers import BatchTimeSeriesContainer
from src.data.scalers import MinMaxScaler, RobustScaler
from src.data.time_features import compute_batch_time_features
from src.models.blocks import GatedDeltaProductEncoder
from src.utils.utils import device


def create_scaler(scaler_type: str, epsilon: float = 1e-3):
    """Create scaler instance based on type."""
    if scaler_type == "custom_robust":
        return RobustScaler(epsilon=epsilon)
    elif scaler_type == "min_max":
        return MinMaxScaler(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")


def apply_channel_noise(values: torch.Tensor, noise_scale: float = 0.1):
    """Add noise to constant channels to prevent model instability."""
    is_constant = torch.all(values == values[:, 0:1, :], dim=1)
    noise = torch.randn_like(values) * noise_scale * is_constant.unsqueeze(1)
    return values + noise


class TimeSeriesModel(nn.Module):
    """Time series forecasting model combining embedding, encoding, and prediction."""

    def __init__(
        self,
        # Core architecture
        embed_size: int = 128,
        num_encoder_layers: int = 2,
        # Scaling and preprocessing
        scaler: str = "custom_robust",
        epsilon: float = 1e-3,
        scaler_clamp_value: float = None,
        handle_constants: bool = False,
        # Time features
        K_max: int = 6,
        time_feature_config: dict = None,
        encoding_dropout: float = 0.0,
        # Encoder configuration
        encoder_config: dict = None,
        # Loss configuration
        loss_type: str = "huber",  # "huber", "quantile"
        quantiles: list[float] = None,
        **kwargs,
    ):
        super().__init__()

        # Core parameters
        self.embed_size = embed_size
        self.num_encoder_layers = num_encoder_layers
        self.epsilon = epsilon
        self.scaler_clamp_value = scaler_clamp_value
        self.handle_constants = handle_constants
        self.encoding_dropout = encoding_dropout
        self.K_max = K_max
        self.time_feature_config = time_feature_config or {}
        self.encoder_config = encoder_config or {}

        # Store loss parameters
        self.loss_type = loss_type
        self.quantiles = quantiles
        if self.loss_type == "quantile" and self.quantiles is None:
            raise ValueError("Quantiles must be provided for quantile loss.")
        if self.quantiles:
            self.register_buffer(
                "qt", torch.tensor(self.quantiles, device=device).view(1, 1, 1, -1)
            )

        # Validate configuration before initialization
        self._validate_configuration()

        # Initialize components
        self.scaler = create_scaler(scaler, epsilon)
        self._init_embedding_layers()
        self._init_encoder_layers(self.encoder_config, num_encoder_layers)
        self._init_projection_layers()

    def _validate_configuration(self):
        """Validate essential model configuration parameters."""
        if "num_heads" not in self.encoder_config:
            raise ValueError("encoder_config must contain 'num_heads' parameter")

        if self.embed_size % self.encoder_config["num_heads"] != 0:
            raise ValueError(
                f"embed_size ({self.embed_size}) must be divisible by "
                f"num_heads ({self.encoder_config['num_heads']})"
            )

    def _init_embedding_layers(self):
        """Initialize value and time feature embedding layers."""
        self.expand_values = nn.Linear(1, self.embed_size, bias=True)
        self.nan_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.embed_size) / self.embed_size,
            requires_grad=True,
        )
        self.time_feature_projection = nn.Linear(self.K_max, self.embed_size)

    def _init_encoder_layers(self, encoder_config: dict, num_encoder_layers: int):
        """Initialize encoder layers."""
        self.num_encoder_layers = num_encoder_layers

        # Ensure encoder_config has token_embed_dim
        encoder_config = encoder_config.copy()
        encoder_config["token_embed_dim"] = self.embed_size
        self.encoder_layers = nn.ModuleList(
            [
                GatedDeltaProductEncoder(layer_idx=layer_idx, **encoder_config)
                for layer_idx in range(self.num_encoder_layers)
            ]
        )

    def _init_projection_layers(self):
        if self.loss_type == "quantile":
            output_dim = len(self.quantiles)
        else:
            output_dim = 1
        self.final_output_layer = nn.Linear(self.embed_size, output_dim)

        self.mlp = GatedMLP(
            hidden_size=self.embed_size,
            hidden_ratio=4,
            hidden_act="swish",
            fuse_swiglu=True,
        )
        # Initialize learnable initial hidden state for the first encoder layer
        # This will be expanded to match batch size during forward pass
        head_k_dim = self.embed_size // self.encoder_config["num_heads"]

        # Get expand_v from encoder_config, default to 1.0 if not present
        expand_v = self.encoder_config.get("expand_v", 1.0)
        head_v_dim = int(head_k_dim * expand_v)

        num_initial_hidden_states = self.num_encoder_layers
        self.initial_hidden_state = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        1, self.encoder_config["num_heads"], head_k_dim, head_v_dim
                    )
                    / head_k_dim,
                    requires_grad=True,
                )
                for _ in range(num_initial_hidden_states)
            ]
        )

    def _preprocess_data(self, data_container: BatchTimeSeriesContainer):
        """Extract data shapes and handle constants without padding."""
        history_values = data_container.history_values
        future_values = data_container.future_values
        history_mask = data_container.history_mask

        batch_size, history_length, num_channels = history_values.shape
        future_length = future_values.shape[1] if future_values is not None else 0

        # Handle constants
        if self.handle_constants:
            history_values = apply_channel_noise(history_values)

        return {
            "history_values": history_values,
            "future_values": future_values,
            "history_mask": history_mask,
            "num_channels": num_channels,
            "history_length": history_length,
            "future_length": future_length,
            "batch_size": batch_size,
        }

    def _compute_scaling(
        self, history_values: torch.Tensor, history_mask: torch.Tensor = None
    ):
        """Compute scaling statistics and apply scaling."""
        scale_statistics = self.scaler.compute_statistics(history_values, history_mask)
        return scale_statistics

    def _apply_scaling_and_masking(
        self, values: torch.Tensor, scale_statistics: dict, mask: torch.Tensor = None
    ):
        """Apply scaling and optional masking to values."""
        scaled_values = self.scaler.scale(values, scale_statistics)

        if mask is not None:
            scaled_values = scaled_values * mask.unsqueeze(-1).float()

        if self.scaler_clamp_value is not None:
            scaled_values = torch.clamp(
                scaled_values, -self.scaler_clamp_value, self.scaler_clamp_value
            )

        return scaled_values

    def _get_positional_embeddings(
        self,
        time_features: torch.Tensor,
        num_channels: int,
        batch_size: int,
        drop_enc_allow: bool = False,
    ):
        """Generate positional embeddings from time features."""
        seq_len = time_features.shape[1]

        if (torch.rand(1).item() < self.encoding_dropout) and drop_enc_allow:
            return torch.zeros(
                batch_size, seq_len, num_channels, self.embed_size, device=device
            ).to(torch.float32)

        pos_embed = self.time_feature_projection(time_features)
        return pos_embed.unsqueeze(2).expand(-1, -1, num_channels, -1)

    def _compute_embeddings(
        self,
        scaled_history: torch.Tensor,
        history_pos_embed: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ):
        """Compute value embeddings and combine with positional embeddings."""

        nan_mask = torch.isnan(scaled_history)
        history_for_embedding = torch.nan_to_num(scaled_history, nan=0.0)
        channel_embeddings = self.expand_values(history_for_embedding.unsqueeze(-1))
        channel_embeddings[nan_mask] = self.nan_embedding.to(channel_embeddings.dtype)
        channel_embeddings = channel_embeddings + history_pos_embed

        # Suppress padded time steps completely so padding is a pure batching artifact
        # history_mask: [B, S] -> broadcast to [B, S, 1, 1]
        if history_mask is not None:
            mask_broadcast = (
                history_mask.unsqueeze(-1).unsqueeze(-1).to(channel_embeddings.dtype)
            )
            channel_embeddings = channel_embeddings * mask_broadcast

        batch_size, seq_len = scaled_history.shape[:2]
        all_channels_embedded = channel_embeddings.view(batch_size, seq_len, -1)

        return all_channels_embedded

    def _generate_predictions(
        self,
        embedded: torch.Tensor,
        target_pos_embed: torch.Tensor,
        prediction_length: int,
        num_channels: int,
        history_mask: torch.Tensor = None,
    ):
        """
        Generate predictions for all channels using vectorized operations.
        """
        batch_size, seq_len, _ = embedded.shape
        # embedded shape: [B, S, N*E] -> Reshape to [B, S, N, E]
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Vectorize across channels by merging the batch and channel dimensions.
        # [B, S, N, E] -> [B*N, S, E]
        channel_embedded = (
            embedded.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, self.embed_size)
        )

        # Reshape target positional embeddings similarly: [B, P, N, E] -> [B*N, P, E]
        target_pos_embed = (
            target_pos_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, prediction_length, self.embed_size)
        )
        x = channel_embedded
        target_repr = target_pos_embed
        x = torch.concatenate([x, target_repr], dim=1)
        if self.encoder_config.get("weaving", True):
            # initial hidden state is learnable
            hidden_state = torch.zeros_like(
                self.initial_hidden_state[0].repeat(batch_size * num_channels, 1, 1, 1)
            )
            for layer_idx, encoder_layer in enumerate(self.encoder_layers):
                x, hidden_state = encoder_layer(
                    x,
                    hidden_state
                    + self.initial_hidden_state[layer_idx].repeat(
                        batch_size * num_channels, 1, 1, 1
                    ),
                )
        else:
            # initial hidden state is separately learnable for each layer
            for layer_idx, encoder_layer in enumerate(self.encoder_layers):
                initial_hidden_state = self.initial_hidden_state[layer_idx].repeat(
                    batch_size * num_channels, 1, 1, 1
                )
                x, _ = encoder_layer(x, initial_hidden_state)

        # Use the last prediction_length positions
        prediction_embeddings = x[:, -prediction_length:, :]

        predictions = self.final_output_layer(self.mlp(prediction_embeddings))

        # Reshape output to handle quantiles
        # Original shape: [B*N, P, Q] where Q is num_quantiles or 1
        # Reshape the output back to [B, P, N, Q]
        output_dim = len(self.quantiles) if self.loss_type == "quantile" else 1
        predictions = predictions.view(
            batch_size, num_channels, prediction_length, output_dim
        )
        predictions = predictions.permute(0, 2, 1, 3)  # [B, P, N, Q]
        # Squeeze the last dimension if not in quantile mode for backward compatibility
        if self.loss_type != "quantile":
            predictions = predictions.squeeze(-1)  # [B, P, N]
        return predictions

    def forward(
        self, data_container: BatchTimeSeriesContainer, drop_enc_allow: bool = False
    ):
        """Main forward pass."""
        # Preprocess data
        preprocessed = self._preprocess_data(data_container)

        # Compute time features dynamically based on actual lengths
        history_time_features, target_time_features = compute_batch_time_features(
            start=data_container.start,
            history_length=preprocessed["history_length"],
            future_length=preprocessed["future_length"],
            batch_size=preprocessed["batch_size"],
            frequency=data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        # Compute scaling
        scale_statistics = self._compute_scaling(
            preprocessed["history_values"], preprocessed["history_mask"]
        )

        # Apply scaling
        history_scaled = self._apply_scaling_and_masking(
            preprocessed["history_values"],
            scale_statistics,
            preprocessed["history_mask"],
        )

        # Scale future values if present
        future_scaled = None
        if preprocessed["future_values"] is not None:
            future_scaled = self.scaler.scale(
                preprocessed["future_values"], scale_statistics
            )

        # Get positional embeddings
        history_pos_embed = self._get_positional_embeddings(
            history_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
            drop_enc_allow,
        )
        target_pos_embed = self._get_positional_embeddings(
            target_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
            drop_enc_allow,
        )

        # Compute embeddings
        history_embed = self._compute_embeddings(
            history_scaled, history_pos_embed, preprocessed["history_mask"]
        )

        # Generate predictions
        predictions = self._generate_predictions(
            history_embed,
            target_pos_embed,
            preprocessed["future_length"],
            preprocessed["num_channels"],
            preprocessed["history_mask"],
        )

        return {
            "result": predictions,
            "scale_statistics": scale_statistics,
            "future_scaled": future_scaled,
            "history_length": preprocessed["history_length"],
            "future_length": preprocessed["future_length"],
        }

    def _quantile_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Compute the quantile loss.
        y_true: [B, P, N]
        y_pred: [B, P, N, Q]
        """
        # Add a dimension to y_true to match y_pred: [B, P, N] -> [B, P, N, 1]
        y_true = y_true.unsqueeze(-1)

        # Calculate errors
        errors = y_true - y_pred

        # Calculate quantile loss
        # The max operator implements the two cases of the quantile loss formula
        loss = torch.max((self.qt - 1) * errors, self.qt * errors)

        # Average the loss across all dimensions
        return loss.mean()

    def compute_loss(self, y_true: torch.Tensor, y_pred: dict):
        """Compute loss between predictions and scaled ground truth."""
        predictions = y_pred["result"]
        scale_statistics = y_pred["scale_statistics"]

        if y_true is None:
            return torch.tensor(0.0, device=predictions.device)

        future_scaled = self.scaler.scale(y_true, scale_statistics)

        if self.loss_type == "huber":
            if predictions.shape != future_scaled.shape:
                raise ValueError(
                    f"Shape mismatch for Huber loss: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
                )
            return nn.functional.huber_loss(predictions, future_scaled)
        elif self.loss_type == "quantile":
            return self._quantile_loss(future_scaled, predictions)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
