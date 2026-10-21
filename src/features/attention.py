"""
Attention-based feature extraction functions.

This module implements various attention contribution computations as described
in the paper "Tracing Relational Knowledge Recall in Large Language Models".
"""

import torch


def extract_per_head_per_token_attention(
    model,
    rkr_token_id: int,
    source_token_id: int
) -> torch.Tensor:
    """
    Extract per-head per-token attention contributions (Δ_att,e1,h).

    Computes the contribution of a specific source token (typically the last token
    of the subject entity) via each attention head to the RKR position.

    Args:
        model: The LLM model with attention hooks
        rkr_token_id: Target token position (RKR)
        source_token_id: Source token position (e.g., last token of entity)

    Returns:
        Tensor of shape (num_layers, num_heads, d_model)
    """
    per_layer = []

    for layer in model.model.layers:
        attn = layer.self_attn

        # Retrieve stored tensors from hooks
        attn_weights = attn._last_attn_weights[0].cpu()  # (H, T, J)
        V = attn._last_value[0].cpu()                    # (H, J, d_head)

        H = attn.num_heads
        d_head = attn.head_dim
        W = attn.o_proj.weight.detach().cpu()            # (d_model, H*d_head)

        # Slice W into per-head blocks
        W_blocks = [
            W[:, h * d_head : (h + 1) * d_head]
            for h in range(H)
        ]

        # Compute per-head contributions
        per_head = []
        for h in range(H):
            # Contribution: W_h @ (attn_weight[h,t,j] * V[h,j])
            contrib = W_blocks[h] @ (
                attn_weights[h, rkr_token_id, source_token_id] * V[h, source_token_id]
            )
            per_head.append(contrib)

        per_head = torch.stack(per_head)  # (H, d_model)
        per_layer.append(per_head)

    # Stack over layers: (num_layers, H, d_model)
    return torch.stack(per_layer)


def extract_layer_token_attention(
    model,
    rkr_token_id: int,
    source_token_id: int
) -> torch.Tensor:
    """
    Extract total contribution from a source token across all heads (Δ_att,e1).

    Args:
        model: The LLM model with attention hooks
        rkr_token_id: Target token position (RKR)
        source_token_id: Source token position

    Returns:
        Tensor of shape (num_layers, d_model)
    """
    per_head_token = extract_per_head_per_token_attention(
        model, rkr_token_id, source_token_id
    )
    # Sum over heads: (num_layers, d_model)
    return per_head_token.sum(dim=1)


