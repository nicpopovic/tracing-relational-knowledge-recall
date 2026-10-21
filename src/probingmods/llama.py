"""
Patched Llama modules that expose per-head attention contributions and the
quantities needed to propagate them through the MLP (RMSNorm scale, gate and
up-projection activations at the probing position).

The attention, MLP, RMSNorm and decoder-layer classes are adapted from
Hugging Face transformers (Apache License 2.0); `get_model` swaps them in before
loading the pretrained weights.
"""
import torch
from torch import nn
from typing import Optional

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    ACT2FN,
    GradientCheckpointingLayer,
    TransformersKwargs,
)

# Force LLaMA to use eager attention so attention weights are available.
LlamaConfig._attn_implementation = "eager"


class PatchedLlamaAttention(nn.Module):
    """
    Replacement for LlamaAttention that additionally records:
      • per-head residual stream writes (W_o,h @ z_h)
      • attention weights and values (for later token-level attribution)
    Returned values to the model remain identical to the Hugging Face implementation.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Head structure
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Standard Q,K,V,O projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self._save_auxiliary_cpu_tensors = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Standard attention forward pass, but additionally stores:
            • self._last_head_writes : (batch, seq_len, heads, hidden_dim)
            • self._last_attn_weights: (batch, heads, q_len, kv_len)
            • self._last_value       : (batch, heads, kv_len, head_dim)
        """
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        # ----- Q, K, V projections -----
        # With device_map="auto" each projection may live on a different GPU;
        # normalise everything to query's device before any cross-tensor op.
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        dev   = query.device
        key   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2).to(dev)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).to(dev)

        # ----- Apply rotary embeddings -----
        cos, sin = position_embeddings
        cos, sin = cos.to(dev), sin.to(dev)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # ----- Update KV cache if present -----
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)

        # ----- Actual attention computation (uses HF's eager implementation) -----
        attention_interface = (
            eager_attention_forward
            if self.config._attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        )

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask.to(dev) if attention_mask is not None else None,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        self._last_attn_weights = attn_weights

        # ----- Recompute per-head V-aggregation z_h = Σ α·V -----
        if self.num_key_value_groups > 1:
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        z = torch.matmul(attn_weights, value)                  # (b, heads, q_len, head_dim)
        z = z.permute(0, 2, 1, 3).contiguous()                 # (b, q_len, heads, head_dim)

        # ----- Apply per-head slices of the output projection -----
        # o_proj may be on a different GPU; move its weight to z's device.
        W = self.o_proj.weight.to(z.device)
        head_writes = []

        for h in range(self.num_heads):
            W_h = W[:, h*self.head_dim : (h+1)*self.head_dim]  # (hidden_dim, head_dim)
            writes_h = z[:, :, h, :] @ W_h.T                   # (b, q_len, hidden_dim)
            head_writes.append(writes_h)

        head_writes = torch.stack(head_writes, dim=2)          # (b, q_len, heads, hidden_dim)

        # ----- Standard final projection for the full attention output -----
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # Save for later introspection
        self._last_head_writes = head_writes

        # Save tensors needed for later token-level attribution. Callers that
        # only need `_last_head_writes` can disable this CPU transfer.
        if self._save_auxiliary_cpu_tensors:
            self._last_attn_weights = attn_weights.detach().cpu()
            self._last_value = value.detach().cpu()

        # Match HF's expected return signature
        return attn_output, attn_weights


class PatchedLlamaMLP(nn.Module):
    """
    LLaMA MLP with stored pre-down-projection activations and
    components needed for linearized attribution through the SiLU gate.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn    = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward:
            down_proj(act(gate_proj(x)) * up_proj(x))

        Additionally stores:
            • _last_gate_pre         : gate_proj(x)
            • _last_gate_act         : act(gate_proj(x))
            • _last_up_pre           : up_proj(x)
            • _last_mlp_activations  : act(gate_proj(x)) * up_proj(x)
        """
        gate_pre = self.gate_proj(x)                        # (b, t, d_ff)
        gate_act = self.act_fn(gate_pre)                   # (b, t, d_ff)
        # up_proj may be on a different GPU; normalise to gate's device.
        up_pre   = self.up_proj(x).to(gate_act.device)    # (b, t, d_ff)

        mlp_activations = gate_act * up_pre               # pre down_proj

        # Store for probing (detach to avoid dangling graph)
        self._last_gate_pre        = gate_pre.detach()
        self._last_gate_act        = gate_act.detach()
        self._last_up_pre          = up_pre.detach()
        self._last_mlp_activations = mlp_activations.detach()

        down_proj = self.down_proj(mlp_activations)
        return down_proj


class PatchedLlamaRMSNorm(nn.Module):
    """
    Simple RMSNorm that mirrors HF's LlamaRMSNorm behavior,
    but we keep it explicit instead of delegating to kernels.
    Used both for input_layernorm and post_attention_layernorm.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.variance_epsilon)

        normed = hidden_states * inv_rms
        return (self.weight * normed).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PatchedLlamaDecoderLayer(GradientCheckpointingLayer):
    """
    Decoder layer that wires in PatchedLlamaAttention and PatchedLlamaMLP,
    and stores the residual stream right before the MLP RMSNorm + the MLP input.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PatchedLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = PatchedLlamaMLP(config)

        self.input_layernorm = PatchedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PatchedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: TransformersKwargs,
    ) -> torch.Tensor:
        # ----- Self-Attention block -----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # accelerate may move hidden_states to a different device inside the
        # sub-module call; ensure residual follows before adding.
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Save post-attention residual stream (pre-MLP RMSNorm)
        # shape: (batch, seq, d_model)
        self._last_post_attn_residual = hidden_states.detach()

        # ----- MLP block -----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Save actual MLP input (post-RMSNorm)
        self._last_mlp_input = hidden_states.detach()

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Helper: linearized propagation of a residual contribution into MLP activations
# ---------------------------------------------------------------------------

def propagate_residual_to_mlp_activations_for_token(
    layer,
    token_id: int,
    delta_residual_token: torch.Tensor,
):
    """
    Linearized propagation of a residual-stream contribution at a single token
    position into the MLP pre-down-projection activations (RMSNorm scale and
    SiLU gate frozen at their values from the last forward pass).

    Args:
        layer: PatchedLlamaDecoderLayer
        token_id: int
        delta_residual_token: (d_model,) contribution at this token in the
                              pre-MLP-residual stream.

    Returns:
        delta_mlp_t: (d_ff,) tensor, contribution to MLP pre-down activations.
    """
    device = layer._last_mlp_input.device

    # 1. RMSNorm linearization at token t
    pre_ln_full_t = layer._last_post_attn_residual[0, token_id].to(device)  # (d_model,)
    delta_residual_t = delta_residual_token.to(device)                      # (d_model,)

    weight = layer.post_attention_layernorm.weight.to(device)
    eps = layer.post_attention_layernorm.variance_epsilon

    variance = pre_ln_full_t.pow(2).mean()
    inv_rms = torch.rsqrt(variance + eps)                                   # scalar

    delta_normed_t = delta_residual_t * inv_rms * weight                    # (d_model,)

    # 2. MLP linearized propagation (frozen SiLU) at token t
    mlp = layer.mlp
    gate_pre_t = mlp._last_gate_pre[0, token_id].to(device)                 # (d_ff,)
    gate_act_t = mlp._last_gate_act[0, token_id].to(device)                 # (d_ff,)
    up_pre_t   = mlp._last_up_pre[0, token_id].to(device)                   # (d_ff,)

    # SiLU derivative at token t
    sig = torch.sigmoid(gate_pre_t)
    d_silu_t = sig + gate_pre_t * sig * (1 - sig)                           # (d_ff,)

    # Linear projections
    G = mlp.gate_proj.weight.to(device)                                     # (d_ff, d_model)
    U = mlp.up_proj.weight.to(device)                                       # (d_ff, d_model)

    delta_gate_pre_t = G @ delta_normed_t                                   # (d_ff,)
    delta_up_pre_t   = U @ delta_normed_t                                   # (d_ff,)

    # frozen gating formula
    delta_activ_t = d_silu_t * delta_gate_pre_t * up_pre_t + \
                    gate_act_t * delta_up_pre_t

    return delta_activ_t.detach().cpu()


# ---------------------------------------------------------------------------
# Model loader with monkey-patching
# ---------------------------------------------------------------------------

def get_model(model_name: str, device_map=None):
    """
    Loads a model with patched attention, MLP, RMSNorm, and decoder layer
    injected globally. Ensures eager attention is used by default.

    Pass device_map="auto" to split the model across all available GPUs
    (requires the `accelerate` package).  When device_map is set, do NOT
    call model.to(device) afterwards — placement is handled by accelerate.

    After a forward pass, for each layer `L = model.model.layers[i]` you get:
      • L.self_attn._last_head_writes
      • L.self_attn._last_attn_weights
      • L.self_attn._last_value
      • L._last_post_attn_residual
      • L._last_mlp_input
      • L.mlp._last_mlp_activations
    and `propagate_residual_to_mlp_activations_for_token(L, token_id,
    delta_residual_token)` maps a residual contribution at one token into MLP
    pre-down activations.
    """
    import transformers
    from transformers import AutoModelForCausalLM

    # Monkey-patch HF classes
    transformers.models.llama.modeling_llama.LlamaAttention    = PatchedLlamaAttention
    transformers.models.llama.modeling_llama.LlamaMLP          = PatchedLlamaMLP
    transformers.models.llama.modeling_llama.LlamaRMSNorm      = PatchedLlamaRMSNorm
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = PatchedLlamaDecoderLayer

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    model.config._attn_implementation = "eager"
    return model
