"""
Patched Qwen3 modules that expose per-head attention contributions and the
quantities needed to propagate them through the MLP (RMSNorm scale, gate and
up-projection activations at the probing position).

The attention, MLP, RMSNorm and decoder-layer classes are adapted from
Hugging Face transformers (Apache License 2.0); `get_model` swaps them in before
loading the pretrained weights.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
    ACT2FN,
    GradientCheckpointingLayer,
)

# Force eager attention globally
Qwen3Config._attn_implementation = "eager"


# ---------------------------------------------------------------------
# RMSNorm (explicit, no kernels)
# ---------------------------------------------------------------------

class PatchedQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------
# Patched Attention
# ---------------------------------------------------------------------

class PatchedQwen3Attention(nn.Module):
    """
    Qwen3 attention that additionally records:
      • attention weights
      • values
      • per-head residual writes via W_o slices
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self._save_auxiliary_cpu_tensors = True

        # Qwen3-specific per-head RMSNorm
        self.q_norm = PatchedQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = PatchedQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position}
            )

        # ----- attention -----
        attn_output, attn_weights = eager_attention_forward(
            self,
            q, k, v,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # ----- per-head z_h -----
        v_rep = repeat_kv(v, self.num_key_value_groups)
        z = torch.matmul(attn_weights, v_rep)
        z = z.permute(0, 2, 1, 3).contiguous()

        # ----- per-head W_o writes -----
        W = self.o_proj.weight
        head_writes = []
        for h in range(self.num_heads):
            W_h = W[:, h * self.head_dim : (h + 1) * self.head_dim]
            head_writes.append(z[:, :, h, :] @ W_h.T)

        self._last_head_writes = torch.stack(head_writes, dim=2)
        if self._save_auxiliary_cpu_tensors:
            self._last_attn_weights = attn_weights.detach().cpu()
            self._last_value = v_rep.detach().cpu()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ---------------------------------------------------------------------
# Patched MLP
# ---------------------------------------------------------------------

class PatchedQwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        gate_pre = self.gate_proj(x)
        gate_act = self.act_fn(gate_pre)
        up_pre = self.up_proj(x)

        activ = gate_act * up_pre

        self._last_gate_pre = gate_pre.detach()
        self._last_gate_act = gate_act.detach()
        self._last_up_pre = up_pre.detach()
        self._last_mlp_activations = activ.detach()

        return self.down_proj(activ)


# ---------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------

class PatchedQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = PatchedQwen3Attention(config, layer_idx)
        self.mlp = PatchedQwen3MLP(config)

        self.input_layernorm = PatchedQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PatchedQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        self._last_post_attn_residual = hidden_states.detach()

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        self._last_mlp_input = hidden_states.detach()

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------
# Linearized propagation of a residual contribution into MLP activations
# ---------------------------------------------------------------------

def propagate_residual_to_mlp_activations_for_token(
    layer,
    token_idx: int,
    delta_residual: torch.Tensor,
):
    # batch index 0 (single-sequence forward)
    r_post = layer._last_post_attn_residual[0, token_idx]   # (d_model,)

    mlp = layer.mlp
    gate_pre = mlp._last_gate_pre[0, token_idx]             # (d_ff,)
    gate_act = mlp._last_gate_act[0, token_idx]
    up_pre   = mlp._last_up_pre[0, token_idx]

    # --- RMSNorm linearization ---
    # RMSNorm(x) = w * x / rms(x); with rms(x) frozen at the full-pass value:
    # δx_norm = w * δx / rms(x)
    rms = torch.sqrt((r_post.float() ** 2).mean() + layer.post_attention_layernorm.variance_epsilon)
    delta_residual = delta_residual.to(layer._last_post_attn_residual.device)
    delta_x = layer.post_attention_layernorm.weight * delta_residual / rms   # (d_model,)

    # --- project residual delta into MLP branches ---
    delta_gate = F.linear(delta_x, mlp.gate_proj.weight)        # (d_ff,)
    delta_up   = F.linear(delta_x, mlp.up_proj.weight)          # (d_ff,)

    # --- SiLU derivative at frozen point ---
    # SiLU'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    sig = torch.sigmoid(gate_pre)
    silu_prime = sig * (1 + gate_pre * (1 - sig))               # (d_ff,)

    # --- combine paths ---
    delta_activ = (
        silu_prime * delta_gate * up_pre
        + gate_act * delta_up
    )                                                           # (d_ff,)

    return delta_activ.detach().cpu()


# ---------------------------------------------------------------------
# Monkey-patch loader
# ---------------------------------------------------------------------

def get_model(model_name: str, device_map=None):
    import transformers
    from transformers import AutoModelForCausalLM

    transformers.models.qwen3.modeling_qwen3.Qwen3Attention = PatchedQwen3Attention
    transformers.models.qwen3.modeling_qwen3.Qwen3MLP = PatchedQwen3MLP
    transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm = PatchedQwen3RMSNorm
    transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer = PatchedQwen3DecoderLayer

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    model.config._attn_implementation = "eager"
    return model
