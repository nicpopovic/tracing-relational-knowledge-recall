"""Online 16-way episodes over both entity orders (Sec. 4.2, 4.3)."""

from .model import ensure_padding_token, load_model
from .prompts import build_orderfix_prompts_subset, mask_text_orderfix
from .runner import (
    FastFewShotEpisodeRunner,
    OrderfixFastFewShotEpisodeRunner,
    extract_att_h_orderfix_batched,
    extract_features_with_attention_orderfix_batched,
)

__all__ = [
    "ensure_padding_token",
    "load_model",
    "build_orderfix_prompts_subset",
    "mask_text_orderfix",
    "FastFewShotEpisodeRunner",
    "OrderfixFastFewShotEpisodeRunner",
    "extract_att_h_orderfix_batched",
    "extract_features_with_attention_orderfix_batched",
]
