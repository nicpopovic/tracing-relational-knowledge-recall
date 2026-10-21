"""
Probe-side attribution (Sec. 3.3): TokenScore over source tokens and HeadScore
over attention heads, plus the GxA baseline used in the apps (Sec. 4.3).
"""

from .tokenscore import (
    compute_tokenscore_value_aware,
    compute_delta_w,
    compute_avg_attention,
)
from .head_contribution import (
    compute_delta_w_batch,
    compute_head_contribution_share_batch,
    heads_needed_for_share,
)
from .gxa import compute_gxa_layerwise

__all__ = [
    "compute_tokenscore_value_aware",
    "compute_delta_w",
    "compute_avg_attention",
    "compute_delta_w_batch",
    "compute_head_contribution_share_batch",
    "heads_needed_for_share",
    "compute_gxa_layerwise",
]
