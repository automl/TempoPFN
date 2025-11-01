import torch
import torch.nn as nn

from src.models.gated_deltaproduct import GatedDeltaProductConfig
from src.models.gated_deltaproduct.modeling_gated_deltaproduct import (
    GatedDeltaProductBlock,
)


class GatedDeltaProductEncoder(nn.Module):
    """
    GatedDeltaNet encoder using GatedDeltaProductBlock for sequence modeling.
    """

    def __init__(
        self,
        layer_idx: int,
        token_embed_dim: int,
        num_heads: int = 4,
        attn_mode: str = "chunk",
        expand_v: float = 1.0,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        allow_neg_eigval: bool = True,
        use_forget_gate: bool = True,
        num_householder: int = 1,
        **kwargs,
    ):
        super().__init__()
        config = GatedDeltaProductConfig(
            attn_mode=attn_mode,
            hidden_size=token_embed_dim,
            expand_v=expand_v,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            head_dim=token_embed_dim // num_heads,
            hidden_ratio=0.5,
            num_heads=num_heads,
            allow_neg_eigval=allow_neg_eigval,
            use_forget_gate=use_forget_gate,
            num_householder=num_householder,
        )

        self.encoder_layer = GatedDeltaProductBlock(layer_idx=layer_idx, config=config)

    def forward(self, x, initial_state=None):
        """
        Forward pass through the GatedDeltaProductBlock.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor of same shape as input
        """
        x, last_hidden_state, _ = self.encoder_layer(
            x, output_attentions=True, initial_state=initial_state
        )
        return x, last_hidden_state
