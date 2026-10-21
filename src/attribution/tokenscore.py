"""
TokenScore: Token-level attribution for relation probe predictions.

TokenScore decomposes a linear probe's prediction into contributions from
individual source tokens, revealing which tokens support (or oppose) the
predicted relation.
"""

import torch
from typing import Tuple, List


def _get_probe_weight_matrix(probe: torch.nn.Module) -> torch.Tensor:
    """Linear weight matrix of an `nn.Linear` or a `LinearProbe`."""
    if hasattr(probe, "weight"):
        return probe.weight
    if hasattr(probe, "linear") and hasattr(probe.linear, "weight"):
        return probe.linear.weight
    raise AttributeError("Probe does not expose a linear weight matrix")


def compute_delta_w(
    probe: torch.nn.Module,
    x_row: torch.Tensor,
    contrast: str = "logsumexp_other",
    tau: float = 1.0
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute probe weight contrast direction for TokenScore.

    Args:
        probe: Trained linear probe
        x_row: [M] probe features for query example
        contrast: Contrast method ("runner_up", "mean_other", "logsumexp_other")
        tau: Temperature for logsumexp_other

    Returns:
        Tuple of:
            - delta_w: [M] contrast weight vector (on CPU)
            - pred_local: Predicted class index
            - other_id: Contrast class index (-1 for mixed)
    """
    device = next(probe.parameters()).device
    x = x_row.to(device).float().unsqueeze(0)  # [1, M]
    logits = probe(x).squeeze(0)  # [C]
    pred = int(torch.argmax(logits).item())

    W = _get_probe_weight_matrix(probe)  # [C, M]
    C, M = W.shape

    if contrast == "runner_up":
        top2 = torch.topk(logits, k=2).indices.tolist()
        runner = int(top2[1]) if int(top2[0]) == pred else int(top2[0])
        delta_w = (W[pred] - W[runner]).detach()
        other_id = runner

    elif contrast == "mean_other":
        mask = torch.ones(C, dtype=torch.bool, device=W.device)
        mask[pred] = False
        delta_w = (W[pred] - W[mask].mean(dim=0)).detach()
        other_id = -1

    elif contrast == "logsumexp_other":
        mask = torch.ones(C, dtype=torch.bool, device=logits.device)
        mask[pred] = False
        other_logits = logits[mask] / float(tau)
        pi = torch.softmax(other_logits, dim=0)  # [C-1]
        W_other = W[mask]  # [C-1, M]
        w_other_mix = (pi[:, None] * W_other).sum(dim=0)
        delta_w = (W[pred] - w_other_mix).detach()
        other_id = -1

    else:
        raise ValueError(f"Unknown contrast method: {contrast}")

    return delta_w.detach().cpu(), pred, other_id


def compute_tokenscore_value_aware(
    Xq_row: torch.Tensor,
    attn_t: torch.Tensor,
    values_by_layer: torch.Tensor,
    probe: torch.nn.Module,
    selected: List[Tuple[int, int, int, int]],
    model: torch.nn.Module,
    contrast: str = "logsumexp_other",
    tau: float = 1.0,
    aggregate_layers: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute value-aware TokenScore using probe weights and attention mechanism.

    Args:
        Xq_row: [M] probe features for query
        attn_t: [L, H, S] attention weights at RKR position
        values_by_layer: [L, H, S, Hd] value tensors
        probe: Trained linear probe
        selected: List of (layer, head, dim, class) neuron coordinates
        model: Model with o_proj weights
        contrast: Contrast method for probe weights
        tau: Temperature parameter
        aggregate_layers: If True, return [S], else return [L, S]

    Returns:
        Tuple of:
            - token_scores: [S] or [L, S] TokenScore per token
            - pred_local: Predicted class index
            - other_id: Contrast class index
    """
    # Compute probe weight contrast
    delta_w_cpu, pred_local, other_id = compute_delta_w(
        probe=probe,
        x_row=Xq_row,
        contrast=contrast,
        tau=tau,
    )

    L, H, S = attn_t.shape

    # Build sparse dict: g[l][h][d] = weight
    g = [[{} for _ in range(H)] for __ in range(L)]
    for m, sel in enumerate(selected):
        # Handle both dict and tuple formats
        if isinstance(sel, dict):
            l, h, d = sel["i"], sel["j"], sel["d"]
        else:
            l, h, d, _ = sel
        g[l][h][d] = g[l][h].get(d, 0.0) + float(delta_w_cpu[m].item())

    token_by_layer = torch.zeros((L, S), dtype=torch.float32)

    layers = model.model.layers
    for l in range(L):
        W_o = layers[l].self_attn.o_proj.weight.detach().cpu()  # [Dmodel, H*Hd]
        Hd = layers[l].self_attn.head_dim

        for h in range(H):
            if not g[l][h]:
                continue

            alpha = attn_t[l, h, :].float()  # [S]
            V = values_by_layer[l, h, :, :].float()  # [S, Hd]
            AV = V * alpha[:, None]  # [S, Hd]

            W_h = W_o[:, h * Hd : (h + 1) * Hd]  # [Dmodel, Hd]

            # Only compute dimensions we need
            for d, w_d in g[l][h].items():
                w_row = W_h[int(d), :]  # [Hd]
                contrib_d = AV @ w_row  # [S]
                token_by_layer[l, :] += float(w_d) * contrib_d

    if aggregate_layers:
        token_scores = token_by_layer.sum(dim=0)  # [S]
    else:
        token_scores = token_by_layer  # [L, S]

    return token_scores, pred_local, other_id


def compute_avg_attention(attn_t: torch.Tensor) -> torch.Tensor:
    """
    Compute average attention across layers and heads (baseline).

    Args:
        attn_t: [L, H, S] attention weights

    Returns:
        [S] average attention per token
    """
    return attn_t.float().mean(dim=(0, 1)).cpu()
