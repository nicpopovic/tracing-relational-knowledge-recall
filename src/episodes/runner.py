"""
Episode runner for the online 16-way experiments: AP expertise, RelSpec
selection and the linear probe on flattened [L, H, D] features, plus batched
extraction of Δ_att,h (and, for TokenScore, attention weights and values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.probing import compute_ap_from_sortidx

from .prompts import build_orderfix_prompts_subset


@dataclass
class FastSelectedNeurons:
    """Selected RelSpec features as dict entries and as flat/head indices."""

    entries: List[Dict]
    flat_indices: torch.Tensor
    head_indices: torch.Tensor


class FastFewShotEpisodeRunner:
    """RelSpec selection and probe training for one few-shot episode."""

    def __init__(self, model, tokenizer, device, expertise_chunk_size: int | None = 16384):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.expertise_chunk_size = None if expertise_chunk_size is None else int(expertise_chunk_size)

    @staticmethod
    def infer_feature_shape(rkr_features: torch.Tensor) -> Tuple[int, int, int]:
        if rkr_features.ndim != 4:
            raise ValueError("Expected rkr_features with shape [N, L, H, D]")
        return tuple(int(x) for x in rkr_features.shape[1:])

    @staticmethod
    def flatten_feature_tensor(rkr_features: torch.Tensor) -> torch.Tensor:
        """Flatten `[N, L, H, D]` to `[N, F]` (row-major, as `.reshape(-1)`)."""
        if rkr_features.ndim != 4:
            raise ValueError("Expected rkr_features with shape [N, L, H, D]")
        return rkr_features.reshape(rkr_features.shape[0], -1)

    @staticmethod
    def decode_flat_indices(flat_indices: torch.Tensor, feature_shape: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode flattened `[L, H, D]` coordinates back to `(layer, head, dim)`."""
        _, num_heads, head_dim = (int(feature_shape[0]), int(feature_shape[1]), int(feature_shape[2]))
        flat_indices = flat_indices.long()
        layers = flat_indices // (num_heads * head_dim)
        rem = flat_indices % (num_heads * head_dim)
        heads = rem // head_dim
        dims = rem % head_dim
        return layers, heads, dims

    def compute_expertise_flat(
        self,
        support_flat: torch.Tensor,
        labels_np: np.ndarray,
        episode_classes: Sequence[int],
    ) -> torch.Tensor:
        """
        Compute RelSpec AP scores in flattened feature space.

        Returns:
            Tensor of shape [C, F] on CPU.
        """
        labels = torch.as_tensor(labels_np, dtype=torch.long, device=self.device)
        support_flat = support_flat.to(self.device).float()

        num_features = int(support_flat.shape[1])
        chunk_size = num_features if self.expertise_chunk_size is None else min(self.expertise_chunk_size, num_features)
        class_masks = [(labels == int(cls)).long() for cls in episode_classes]
        expertise_chunks = [[] for _ in episode_classes]

        # Chunk over feature columns to cap peak memory; AP is computed
        # independently per column, so chunking does not change the result.
        for start in range(0, num_features, chunk_size):
            end = min(start + chunk_size, num_features)
            support_chunk = support_flat[:, start:end]  # [N_support, F_chunk]
            _, sort_idx = torch.sort(support_chunk, dim=0, descending=True)
            for class_idx, labels_binary in enumerate(class_masks):
                expertise_chunks[class_idx].append(compute_ap_from_sortidx(sort_idx, labels_binary))
            del support_chunk, sort_idx
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        expertise_rows = [torch.cat(chunks, dim=0) for chunks in expertise_chunks]
        return torch.stack(expertise_rows, dim=0).detach().cpu()

    def compute_expertise(
        self,
        rkr_features: torch.Tensor,
        labels_np: np.ndarray,
        episode_classes: Sequence[int],
    ) -> torch.Tensor:
        """AP expertise per episode class and feature, shape `[C, L, H, D]` on CPU."""
        feature_shape = self.infer_feature_shape(rkr_features)
        support_flat = self.flatten_feature_tensor(rkr_features)
        expertise_flat = self.compute_expertise_flat(support_flat, labels_np, episode_classes)
        return expertise_flat.reshape((len(episode_classes),) + feature_shape)

    def select_relspec_neurons_flat(
        self,
        expertise_flat: torch.Tensor,
        episode_classes: Sequence[int],
        n_relspec: int,
        feature_shape: Sequence[int],
    ) -> FastSelectedNeurons:
        """Top-`n_relspec` features per class by AP (class-major order)."""
        k = min(int(n_relspec), int(expertise_flat.shape[1]))
        top_vals, top_idxs = torch.topk(expertise_flat, k=k, dim=1)
        layers, heads, dims = self.decode_flat_indices(top_idxs.reshape(-1), feature_shape)

        entries: List[Dict] = []
        for class_idx, cls in enumerate(episode_classes):
            for rank in range(k):
                flat_pos = class_idx * k + rank
                entries.append(
                    {
                        "class": int(cls),
                        "rank": int(rank),
                        "i": int(layers[flat_pos].item()),
                        "j": int(heads[flat_pos].item()),
                        "d": int(dims[flat_pos].item()),
                        "ap": float(top_vals[class_idx, rank].item()),
                    }
                )

        num_heads = int(feature_shape[1])
        head_indices = layers * num_heads + heads
        return FastSelectedNeurons(
            entries=entries,
            flat_indices=top_idxs.reshape(-1).detach().cpu(),
            head_indices=head_indices.detach().cpu(),
        )

    def select_relspec_neurons(
        self,
        expertise: torch.Tensor,
        episode_classes: Sequence[int],
        n_relspec: int,
    ) -> List[Dict]:
        """
        Select RelSpec features from `[C, L, H, D]` expertise. Returns dicts with
        keys class, rank, i (layer), j (head), d (dim), ap. Features selected by
        several classes are kept once per class.
        """
        if expertise.ndim == 4:
            feature_shape = tuple(int(x) for x in expertise.shape[1:])
            expertise_flat = expertise.reshape(expertise.shape[0], -1)
        elif expertise.ndim == 2:
            raise ValueError("Flat expertise requires explicit feature metadata; use select_relspec_neurons_flat")
        else:
            raise ValueError("Unsupported expertise shape")
        return self.select_relspec_neurons_flat(expertise_flat, episode_classes, n_relspec, feature_shape).entries

    def build_probe_features_from_flat(
        self,
        flat_features: torch.Tensor,
        selected_flat_indices: torch.Tensor,
    ) -> torch.Tensor:
        if selected_flat_indices.numel() == 0:
            return torch.empty((flat_features.shape[0], 0), dtype=flat_features.dtype, device=self.device)
        return flat_features.to(self.device).index_select(1, selected_flat_indices.to(self.device))

    def build_probe_features(
        self,
        rkr_features: torch.Tensor,
        indices: np.ndarray,
        selected: List[Dict],
    ) -> torch.Tensor:
        """Probe input `[len(indices), len(selected)]` from the selected features."""
        subset = rkr_features[indices]
        flat_subset = self.flatten_feature_tensor(subset)
        feature_shape = self.infer_feature_shape(rkr_features)
        head_dim = int(feature_shape[2])
        num_heads = int(feature_shape[1])
        flat_indices = torch.tensor(
            [
                int(sel["i"]) * num_heads * head_dim
                + int(sel["j"]) * head_dim
                + int(sel["d"])
                for sel in selected
            ],
            dtype=torch.long,
        )
        return self.build_probe_features_from_flat(flat_subset, flat_indices)

    def train_probe(self, X_support: torch.Tensor, y_support: torch.Tensor, n_way: int, num_epochs: int = 200):
        probe = nn.Linear(X_support.shape[1], n_way).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            opt.zero_grad()
            loss = loss_fn(probe(X_support.float()), y_support)
            loss.backward()
            opt.step()

        probe.eval()
        return probe


def _set_auxiliary_cpu_tensor_saving(model, enabled: bool):
    previous = []
    for layer in model.model.layers:
        attn = layer.self_attn
        prev = getattr(attn, "_save_auxiliary_cpu_tensors", True)
        previous.append(prev)
        attn._save_auxiliary_cpu_tensors = bool(enabled)
    return previous


def _restore_auxiliary_cpu_tensor_saving(model, previous):
    for layer, prev in zip(model.model.layers, previous):
        layer.self_attn._save_auxiliary_cpu_tensors = bool(prev)


@torch.no_grad()
def extract_att_h_orderfix_batched(
    model,
    tokenizer,
    prompts: Sequence[Dict[str, Any]],
    batch_size: int = 16,
) -> tuple[torch.Tensor, np.ndarray]:
    """Δ_att,h at each prompt's probing position, shape `[N, L, H, D]`."""
    if len(prompts) == 0:
        return torch.empty((0, 0, 0, 0), dtype=torch.float32, device=model.device), np.empty((0,), dtype=np.int64)

    all_features: List[torch.Tensor] = []
    all_labels: List[int] = []
    layers = model.model.layers
    previous_aux = _set_auxiliary_cpu_tensor_saving(model, enabled=False)
    try:
        for start in range(0, len(prompts), int(batch_size)):
            batch_prompts = prompts[start : start + int(batch_size)]
            encodings = [
                {
                    "input_ids": p["tokenized"]["input_ids"][0].tolist(),
                    "attention_mask": p["tokenized"]["attention_mask"][0].tolist(),
                }
                for p in batch_prompts
            ]
            padded = tokenizer.pad(encodings, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(model.device)
            attention_mask = padded["attention_mask"].to(model.device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )

            token_ids = torch.tensor(
                [int(p["token_id"]) for p in batch_prompts],
                dtype=torch.long,
                device=model.device,
            )
            batch_ids = torch.arange(len(batch_prompts), dtype=torch.long, device=model.device)
            writes_t = torch.stack(
                [
                    layer.self_attn._last_head_writes[batch_ids, token_ids, :, :].detach().float().cpu()
                    for layer in layers
                ],
                dim=0,
            ).permute(1, 0, 2, 3)
            all_features.append(writes_t)
            all_labels.extend(int(p["label"]) for p in batch_prompts)

            if model.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        _restore_auxiliary_cpu_tensor_saving(model, previous_aux)

    return torch.cat(all_features, dim=0), np.asarray(all_labels, dtype=np.int64)


@torch.no_grad()
def extract_features_with_attention_orderfix_batched(
    model,
    tokenizer,
    prompts: Sequence[Dict[str, Any]],
    batch_size: int = 16,
    store_attention: bool = True,
):
    """
    Δ_att,h at the probing position plus, per prompt, the attention weights
    from that position `[L, H, S]` and the values `[L, H, S, Hd]` needed for
    TokenScore.
    """
    if len(prompts) == 0:
        return (
            torch.empty((0, 0, 0, 0), dtype=torch.float32, device=model.device),
            np.empty((0,), dtype=np.int64),
            [],
            [],
        )

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise ValueError("Tokenizer does not define a pad token or eos token for batched padding.")
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id

    features: List[torch.Tensor] = []
    labels: List[int] = []
    attn_at_t_list: List[torch.Tensor] = []
    value_list: List[torch.Tensor] = []

    layers = model.model.layers
    previous_aux = None
    if not store_attention:
        previous_aux = _set_auxiliary_cpu_tensor_saving(model, enabled=False)

    try:
        for start in range(0, len(prompts), int(batch_size)):
            batch_prompts = prompts[start : start + int(batch_size)]
            encodings = [
                {
                    "input_ids": p["tokenized"]["input_ids"][0].tolist(),
                    "attention_mask": p["tokenized"]["attention_mask"][0].tolist(),
                }
                for p in batch_prompts
            ]
            padded = tokenizer.pad(encodings, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(model.device)
            attention_mask = padded["attention_mask"].to(model.device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )

            token_ids = torch.tensor(
                [int(p["token_id"]) for p in batch_prompts],
                dtype=torch.long,
                device=model.device,
            )
            batch_ids = torch.arange(len(batch_prompts), dtype=torch.long, device=model.device)

            batch_features = torch.stack(
                [
                    layer.self_attn._last_head_writes[
                        batch_ids.to(layer.self_attn._last_head_writes.device),
                        token_ids.to(layer.self_attn._last_head_writes.device),
                        :,
                        :,
                    ].detach().float().cpu()
                    for layer in layers
                ],
                dim=0,
            ).permute(1, 0, 2, 3)
            features.append(batch_features)
            labels.extend(int(p["label"]) for p in batch_prompts)

            if store_attention:
                for local_idx, prompt in enumerate(batch_prompts):
                    token_id = int(prompt["token_id"])
                    seq_len = int(prompt["tokenized"]["input_ids"].shape[1])
                    attn_layers = []
                    value_layers = []
                    for layer in layers:
                        attn = layer.self_attn._last_attn_weights[local_idx].detach().float().cpu()
                        value = layer.self_attn._last_value[local_idx].detach().float().cpu()
                        attn_layers.append(attn[:, token_id, :seq_len])
                        value_layers.append(value[:, :seq_len, :])

                    attn_at_t_list.append(torch.stack(attn_layers, dim=0))
                    value_list.append(torch.stack(value_layers, dim=0))

            if model.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if previous_aux is not None:
            _restore_auxiliary_cpu_tensor_saving(model, previous_aux)

    return torch.cat(features, dim=0), np.asarray(labels, dtype=np.int64), attn_at_t_list, value_list


class OrderfixFastFewShotEpisodeRunner(FastFewShotEpisodeRunner):
    """Runner that also builds order-fix prompts and extracts Δ_att,h online."""

    def __init__(self, model, tokenizer, device, expertise_chunk_size: int | None = 16384, extraction_batch_size: int = 16):
        super().__init__(model, tokenizer, device, expertise_chunk_size=expertise_chunk_size)
        self.extraction_batch_size = int(extraction_batch_size)

    def extract_online_att_h(self, ds, indices: Sequence[int]) -> tuple[torch.Tensor, np.ndarray, List[Dict[str, Any]]]:
        prompts = build_orderfix_prompts_subset(ds, indices, self.tokenizer)
        features, labels = extract_att_h_orderfix_batched(
            self.model,
            self.tokenizer,
            prompts,
            batch_size=self.extraction_batch_size,
        )
        return features, labels, prompts
