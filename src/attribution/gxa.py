"""
Gradient-times-Activation (GxA) token attribution baseline.

Implements the GxA method of Geva et al. (2023), Appendix B, for LLaMA-style
models. For decoder layer l and input position i,

    GxA^l_i = || (d f_c / d h^l_i) * h^l_i ||_2,

where h^l_i is the residual-stream output of layer l at position i and f_c is
the LLM's own next-token logit for target token c (by default the first object
token) at the probing position. GxA thus attributes the model's object
prediction and is unsigned.

TokenScore (`src/attribution/tokenscore.py`) is not gradient-based: it is a
linear decomposition of the probe's logit contrast (delta_w . x) over source
tokens, using each head's attention weights and values. It attributes the
probe's relation decision and is signed (negative values are evidence against
the predicted relation).
"""

import torch
from typing import Optional, Tuple


def compute_gxa_layerwise(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_pos: int,
    target_token_id: Optional[int] = None,
    normalize_per_layer: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Compute GxA scores per layer.

    For each decoder layer l and input position i:
        GxA^l_i = || (d f_c / d h^l_i) * h^l_i ||_2
    where h^l_i is the residual stream after layer l and f_c is the logit of
    the target token c at `query_pos`. Must not run under torch.no_grad().

    Args:
        model: Patched LLaMA / Qwen3 model.
        input_ids: [1, S] input token IDs.
        attention_mask: [1, S] attention mask.
        query_pos: Position at which to compute the target logit.
        target_token_id: Token ID to differentiate (None = next token, -1 = argmax).
        normalize_per_layer: If True, normalize each layer's scores to sum=1.

    Returns:
        Tuple of:
            - gxa_scores: [L, S] GxA scores (CPU tensor)
            - target_token_id: The token ID used
    """
    S = input_ids.shape[1]
    L = len(model.model.layers)

    # Release cached GPU memory before the grad-enabled forward pass.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if target_token_id is None:
        if query_pos + 1 < S:
            target_token_id = int(input_ids[0, query_pos + 1].item())
        else:
            target_token_id = -1

    hidden_by_layer = {}

    def make_fwd_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            hidden_by_layer[layer_idx] = h
        return hook

    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_fwd_hook(l)))

    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = output.logits[0, query_pos, :]
        if target_token_id == -1:
            target_token_id = int(logits.detach().argmax().item())

        logits[target_token_id].backward()

    finally:
        for h in hooks:
            h.remove()

    # Collect results immediately — move to CPU layer-by-layer to free GPU memory.
    gxa_by_layer = []
    for l in range(L):
        h = hidden_by_layer.pop(l, None)
        if h is None or h.grad is None:
            gxa_by_layer.append(torch.zeros(S, dtype=torch.float32))
            continue

        grad = h.grad[0].float().cpu()
        hval = h[0].detach().float().cpu()
        gxa_l = (grad * hval).norm(dim=-1)  # [S] on CPU

        if normalize_per_layer:
            total = gxa_l.sum()
            if total > 1e-8:
                gxa_l = gxa_l / total

        gxa_by_layer.append(gxa_l)

        # Free GPU memory
        h.grad = None
        del h, grad, hval

    # Free the computation graph and remaining cached memory
    del output, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(gxa_by_layer, dim=0), target_token_id  # [L, S]
