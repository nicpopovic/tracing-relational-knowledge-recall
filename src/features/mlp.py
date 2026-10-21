"""
MLP-based feature extraction functions.

This module implements MLP propagation of attention contributions, incorporating
the non-linearity (SiGLU) while freezing RMSNorm and gate values.
"""

import torch


def _get_propagate_function(model):
    """Get the appropriate MLP propagation function for the model."""
    model_name = model.__class__.__name__.lower()
    if "llama" in model_name:
        from src.probingmods.llama import propagate_residual_to_mlp_activations_for_token
    elif "qwen" in model_name:
        from src.probingmods.qwen3 import propagate_residual_to_mlp_activations_for_token
    else:
        raise NotImplementedError(f"Model {model.__class__.__name__} not supported for MLP propagation")
    return propagate_residual_to_mlp_activations_for_token


def extract_mlp_per_head_per_token(
    model,
    rkr_token_id: int,
    source_token_id: int
) -> torch.Tensor:
    """
    Extract per-head per-token contributions propagated through MLP (Δ_MLP,e1,h).

    For each layer and head, computes the contribution from source_token_id to
    rkr_token_id through that head, then propagates through the MLP.

    Args:
        model: The LLM model
        rkr_token_id: Target token position
        source_token_id: Source token position (e.g., head entity end)

    Returns:
        Tensor of shape (num_layers, num_heads, d_ff)
    """
    propagate_fn = _get_propagate_function(model)

    mlp_features = []

    for layer in model.model.layers:
        attn = layer.self_attn

        # Retrieve attention weights and values
        attn_w = attn._last_attn_weights[0].cpu()  # (heads, T, T)
        V = attn._last_value[0].cpu()              # (heads, T, d_head)

        num_heads = attn.num_heads
        d_head = attn.head_dim
        W = attn.o_proj.weight.detach().cpu()      # (d_model, heads*d_head)

        # Split W per-head
        W_blocks = [
            W[:, h*d_head : (h+1)*d_head]
            for h in range(num_heads)
        ]

        # Compute per-head residual contribution from source to target token
        per_head_resid = []
        for h in range(num_heads):
            # Contribution: W_h @ (alpha[t,j] * V[j])
            contrib = W_blocks[h] @ (attn_w[h, rkr_token_id, source_token_id] * V[h, source_token_id])
            per_head_resid.append(contrib)

        per_head_resid = torch.stack(per_head_resid)  # (heads, d_model)

        # Propagate each head through MLP
        per_head_mlp = []
        for h in range(num_heads):
            delta_mlp_t = propagate_fn(
                layer,
                rkr_token_id,
                per_head_resid[h],
            )
            per_head_mlp.append(delta_mlp_t)

        per_head_mlp = torch.stack(per_head_mlp)  # (heads, d_ff)
        mlp_features.append(per_head_mlp)

    mlp_features = torch.stack(mlp_features)  # (num_layers, heads, d_ff)
    return mlp_features


def extract_mlp_layer_token(
    model,
    rkr_token_id: int,
    source_token_id: int
) -> torch.Tensor:
    """
    Extract layer-token contributions propagated through MLP (Δ_MLP,e1).

    For each layer, computes the total contribution from source_token_id to
    rkr_token_id (summed across all heads), then propagates through MLP.

    Args:
        model: The LLM model
        rkr_token_id: Target token position
        source_token_id: Source token position (e.g., head entity end)

    Returns:
        Tensor of shape (num_layers, d_ff)
    """
    propagate_fn = _get_propagate_function(model)

    mlp_features = []

    for layer in model.model.layers:
        attn = layer.self_attn

        # Retrieve attention weights and values
        attn_w = attn._last_attn_weights[0].cpu()  # (heads, T, T)
        V = attn._last_value[0].cpu()              # (heads, T, d_head)

        num_heads = attn.num_heads
        d_head = attn.head_dim
        W = attn.o_proj.weight.detach().cpu()      # (d_model, heads*d_head)

        # Split W per-head
        W_blocks = [
            W[:, h*d_head : (h+1)*d_head]
            for h in range(num_heads)
        ]

        # Compute per-head write vector and sum across heads
        per_head = []
        for h in range(num_heads):
            contrib = W_blocks[h] @ (attn_w[h, rkr_token_id, source_token_id] * V[h, source_token_id])
            per_head.append(contrib)

        per_head = torch.stack(per_head)  # (heads, d_model)

        # Token-to-token residual contribution (sum across heads)
        write_tj = per_head.sum(0)  # (d_model,)

        # Propagate to MLP activation space
        delta_mlp_t = propagate_fn(
            layer,
            rkr_token_id,
            write_tj,
        )  # (d_ff,)

        mlp_features.append(delta_mlp_t)

    mlp_features = torch.stack(mlp_features)  # (num_layers, d_ff)
    return mlp_features


def extract_mlp_total(model, rkr_token_id: int) -> torch.Tensor:
    """
    Extract the full MLP state ("MLP" baseline) at the recall position.

    For each layer, returns the MLP activation act(W_gate x_t) * (W_up x_t)
    from the regular forward pass, i.e. the input to down_proj at token t.

    Args:
        model: The LLM model (after a forward pass)
        rkr_token_id: Token position for RKR

    Returns:
        Tensor of shape (num_layers, d_ff)
    """
    return torch.stack([
        layer.mlp._last_mlp_activations[0, rkr_token_id].detach().cpu()
        for layer in model.model.layers
    ])  # (num_layers, d_ff)
