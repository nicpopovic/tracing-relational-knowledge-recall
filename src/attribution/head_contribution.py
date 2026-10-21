"""
HeadScore (Sec. 3.3): attribution of a linear probe's prediction to attention
heads.

Each probe feature m is one selected coordinate (layer l_m, head h_m, dim d_m)
of the per-head attention contributions, with value x_m. For predicted class
c_hat and probe weights W, the contrast direction (`contrast="logsumexp_other"`)
is

    dW = W[c_hat] - sum_{c != c_hat} pi_c W[c],
    pi = softmax over the other classes of logit_c / tau,

and the HeadScore of head (l, h) is the net signed contribution

    HeadScore[l, h] = sum_{m : (l_m, h_m) = (l, h)} dW_m * x_m.

Head shares are |HeadScore| / sum |HeadScore|; k_X is the minimum number of
heads whose cumulative |HeadScore| reaches a fraction X of the total.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .tokenscore import _get_probe_weight_matrix


def compute_delta_w_batch(
    probe: torch.nn.Module,
    X_rows: torch.Tensor,
    contrast: str = "logsumexp_other",
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched analogue of `compute_delta_w`.

    Args:
        probe:
            Trained linear probe.
        X_rows:
            Probe feature matrix of shape [N, M].
        contrast:
            Contrast method ("runner_up", "mean_other", or "logsumexp_other").
        tau:
            Temperature for `logsumexp_other`.

    Returns:
        Tuple of:
            - delta_w_batch: [N, M] contrast vectors
            - pred_local: [N] predicted class indices
            - other_id: [N] contrast class indices (-1 for mixed contrasts)
    """
    device = next(probe.parameters()).device
    X = X_rows.to(device).float()
    logits = probe(X)  # [N, C]
    pred = torch.argmax(logits, dim=1)  # [N]

    W = _get_probe_weight_matrix(probe)  # [C, M]
    C = W.shape[0]

    if contrast == "runner_up":
        top2 = torch.topk(logits, k=min(2, C), dim=1).indices
        if C == 1:
            runner = pred.clone()
        else:
            runner = top2[:, 1]
        delta_w = W[pred] - W[runner]
        other_id = runner

    elif contrast == "mean_other":
        if C == 1:
            delta_w = torch.zeros((X.shape[0], W.shape[1]), device=device, dtype=W.dtype)
        else:
            W_sum = W.sum(dim=0, keepdim=True)
            delta_w = W[pred] - (W_sum - W[pred]) / float(C - 1)
        other_id = torch.full_like(pred, -1)

    elif contrast == "logsumexp_other":
        if C == 1:
            delta_w = torch.zeros((X.shape[0], W.shape[1]), device=device, dtype=W.dtype)
        else:
            scaled_logits = logits / float(tau)
            pred_mask = torch.nn.functional.one_hot(pred, num_classes=C).bool()
            masked_logits = scaled_logits.masked_fill(pred_mask, float("-inf"))
            pi = torch.softmax(masked_logits, dim=1).masked_fill(pred_mask, 0.0)
            w_other_mix = pi @ W
            delta_w = W[pred] - w_other_mix
        other_id = torch.full_like(pred, -1)

    else:
        raise ValueError(f"Unknown contrast method: {contrast}")

    return delta_w.detach(), pred.detach(), other_id.detach()


def compute_head_contribution_share_batch(
    X_rows: torch.Tensor,
    probe: torch.nn.Module,
    selected: List[Dict],
    contrast: str = "logsumexp_other",
    tau: float = 1.0,
    aggregate_mode: str = "net_abs",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched probe-side head contribution share for many rows at once.

    Args:
        X_rows:
            Probe feature matrix [N, M].
        probe:
            Trained linear probe.
        selected:
            Selected RelSpec features with `i` and `j` head coordinates.
        contrast, tau:
            Probe contrast (see `compute_delta_w_batch`).
        aggregate_mode:
            Only "net_abs" (net contribution per head, shares of its absolute
            value) is supported.

    Returns:
        Tuple of:
            - signed_head_scores: [N, L, H]
            - head_shares: [N, L, H]
            - pred_local: [N]
            - other_id: [N]
    """
    if aggregate_mode != "net_abs":
        raise ValueError(
            "The direct probe-side head contribution implementation only "
            "supports aggregate_mode='net_abs'"
        )

    if len(selected) == 0:
        n_rows = int(X_rows.shape[0])
        zeros = torch.zeros((n_rows, 0, 0), dtype=torch.float32, device=X_rows.device)
        pred = torch.zeros((n_rows,), dtype=torch.long, device=X_rows.device)
        other = torch.full((n_rows,), -1, dtype=torch.long, device=X_rows.device)
        return zeros, zeros.clone(), pred, other

    delta_w_batch, pred_local, other_id = compute_delta_w_batch(
        probe=probe,
        X_rows=X_rows,
        contrast=contrast,
        tau=tau,
    )

    device = delta_w_batch.device
    num_layers = max(int(sel["i"]) for sel in selected) + 1
    num_heads = max(int(sel["j"]) for sel in selected) + 1
    head_ids = torch.tensor(
        [int(sel["i"]) * num_heads + int(sel["j"]) for sel in selected],
        dtype=torch.long,
        device=device,
    )

    contributions = delta_w_batch * X_rows.to(device).float()  # [N, M]
    signed_head_scores = torch.zeros(
        (contributions.shape[0], num_layers * num_heads),
        dtype=contributions.dtype,
        device=device,
    )
    signed_head_scores.scatter_add_(1, head_ids.unsqueeze(0).expand(contributions.shape[0], -1), contributions)
    signed_head_scores = signed_head_scores.view(contributions.shape[0], num_layers, num_heads)

    abs_scores = signed_head_scores.abs()
    totals = abs_scores.sum(dim=(1, 2), keepdim=True)
    head_shares = torch.where(
        totals > 1e-12,
        abs_scores / totals.clamp_min(1e-12),
        torch.zeros_like(abs_scores),
    )
    return signed_head_scores, head_shares, pred_local, other_id


def heads_needed_for_share(
    head_shares: torch.Tensor,
    threshold: float,
) -> int:
    """
    Count how many heads are needed to explain a target contribution share.

    Args:
        head_shares:
            Normalized contribution-share tensor of shape [L, H].
        threshold:
            Desired cumulative share in the open interval (0, 1], for example
            0.90 for "90% of the contribution share".

    Returns:
        Integer count of the smallest number of heads whose cumulative share
        reaches or exceeds `threshold`.
    """
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("threshold must satisfy 0 < threshold <= 1")

    flat = head_shares.reshape(-1)
    if flat.numel() == 0:
        return 0

    # Sort heads from most important to least important and accumulate until
    # the requested contribution-share target is reached.
    sorted_vals, _ = torch.sort(flat, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)

    for idx, value in enumerate(cumsum.tolist(), start=1):
        if value >= threshold:
            return idx

    # If numerical imprecision prevents an exact hit, all heads are required.
    return int(flat.numel())
