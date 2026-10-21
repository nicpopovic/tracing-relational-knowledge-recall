"""
Few-shot episode runner used by the Gradio apps (entity order "ht", one
forward pass per prompt, attention weights and values kept for TokenScore).
"""

import torch
import torch.nn as nn
import numpy as np
import re
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple

from src.util.data import mask_text
from src.probing import compute_ap_from_sortidx


def build_prompts_subset(ds, indices, tokenizer, mark_entities: str = "head"):
    """
    Build prompts for a subset of dataset indices.

    Args:
        ds: HuggingFace dataset
        indices: Dataset indices to process
        tokenizer: Tokenizer for the model
        mark_entities: Which entities to mark for TokenScore analysis
            - "head": only head entity spans (semantic head)
            - "all": all entity spans in text order

    Returns:
        List of prompt dictionaries with tokenization and metadata
    """
    prompts = []

    for idx in indices:
        item = ds[int(idx)]
        text = item["text"]

        # Order entities by how they appear in the text
        # A = first block in text, B = second block in text
        if item["entity_order"] == "ht":
            A = item["h"]
            B = item["t"]
            A_id, B_id = "h", "t"
        else:  # "th"
            A = item["t"]
            B = item["h"]
            A_id, B_id = "t", "h"

        item["entities_pos"] = A + B
        item["entities_ids"] = [A_id] * len(A) + [B_id] * len(B)
        item["entities_sf"] = [text[x:y] for (x, y) in item["entities_pos"]]

        masked_text = mask_text(item)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Fill in the blanks in the following text:\n\n"
                    f"{masked_text}\n\n"
                    "Provide only the completed text as your answer."
                ),
            },
            {"role": "assistant", "content": text},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        match = re.search(re.escape(text), prompt)
        if not match:
            continue

        start_in_prompt = match.start()

        # Semantic names (independent of entity_order): first mention of h and t
        if len(item["h"]) > 0:
            hs, he = item["h"][0]
            head_name = text[hs:he]
        else:
            head_name = ""

        if len(item["t"]) > 0:
            ts, te = item["t"][0]
            tail_name = text[ts:te]
        else:
            tail_name = ""

        # Target entity span: first mention of second block (B) in ordered text
        if len(A) == 0 or len(B) == 0:
            continue

        target_start_text, target_end_text = item["entities_pos"][len(A)]
        target_start = target_start_text + start_in_prompt
        target_end = target_end_text + start_in_prompt

        # RKR position: token before second-block entity
        rel_start = target_start

        tokenized = tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Map character position to token ID
        token_id = None
        for i, (os, oe) in enumerate(tokenized["offset_mapping"][0]):
            if os <= rel_start < oe:
                token_id = i - 1
                break

        if token_id is None or token_id < 0:
            continue

        # Entity spans (prompt coordinates) for TokenScore analysis
        entity_spans: List[Tuple[int, int]] = []
        if mark_entities == "head":
            spans = A  # semantic head, regardless of entity_order
        elif mark_entities == "all":
            spans = item["entities_pos"]  # ordered by entity_order
        else:
            raise ValueError("mark_entities must be 'head' or 'all'")

        for (s, e) in spans:
            entity_spans.append((int(s + start_in_prompt), int(e + start_in_prompt)))

        prompts.append({
            "dataset_idx": int(idx),
            "prompt": prompt,
            "tokenized": tokenized,
            "token_id": int(token_id),
            "label": int(item["label"][1:].split(" -")[0]),
            "tail_char_span": (int(target_start), int(target_end)),
            "entity_char_spans": entity_spans,
            "head_name": head_name,
            "tail_name": tail_name,
            "entity_order": item["entity_order"],
        })

    return prompts


@torch.no_grad()
def extract_features_with_attention(
    model,
    prompts,
    store_attention: bool = True,
):
    """
    Extract RKR features, and optionally attention/value tensors.

    This helper is used in two slightly different regimes:

    1. Support-set extraction
       Only the RKR features and labels are needed for RelSpec and probe
       training. In this case, retaining attention/value tensors would be pure
       overhead, so callers should pass ``store_attention=False``.

    2. Query-set attribution
       The head-contribution-share and TokenScore-style analyses require the
       attention weights at the RKR position together with the per-head value
       tensors. In this case, callers should use the default
       ``store_attention=True``.

    Keeping this switch inside the extractor avoids duplicating model-forward
    logic while letting memory-hungry analyses stream query examples more
    safely.

    Args:
        model:
            Patched LLM model.
        prompts:
            List of prompt dictionaries from ``build_prompts_subset``.
        store_attention:
            Whether to retain per-example attention weights and value tensors.
            Set this to ``False`` for support examples when only RKR features
            are needed.

    Returns:
        Tuple of:
            - rkr_features: [N, L, H, D] per-head attention contributions
            - labels: [N] labels array
            - attn_at_t_list:
                List of [L, H, S] attention weights per example when
                ``store_attention=True``; otherwise an empty list
            - value_list:
                List of [L, H, S, Hd] value tensors per example when
                ``store_attention=True``; otherwise an empty list
    """
    feats = []
    labels = []
    attn_at_t_list = []
    value_list = []

    layers = model.model.layers

    for p in tqdm(prompts, desc="Extracting features"):
        input_ids = p["tokenized"]["input_ids"].to(model.device)
        attention_mask = p["tokenized"]["attention_mask"].to(model.device)
        t = int(p["token_id"])

        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        # Extract per-head writes at token t
        head_writes = torch.stack([layer.self_attn._last_head_writes.cpu() for layer in layers])
        writes_t = head_writes[:, 0, t, :, :]  # [L, H, D]
        feats.append(writes_t)
        labels.append(p["label"])

        if store_attention:
            # Extract attention weights and values only when attribution
            # requires them. These tensors are by far the largest objects
            # produced per prompt.
            attn_layers = []
            val_layers = []
            for layer in layers:
                a = layer.self_attn._last_attn_weights  # [1, H, S, S]
                a0 = a[0].detach().float().cpu()  # [H, S, S]
                attn_layers.append(a0[:, t, :])  # [H, S]

                v = layer.self_attn._last_value  # [1, H, S, Hd]
                v0 = v[0].detach().float().cpu()  # [H, S, Hd]
                val_layers.append(v0)

            attn_t = torch.stack(attn_layers, dim=0)  # [L, H, S]
            vals = torch.stack(val_layers, dim=0)  # [L, H, S, Hd]

            attn_at_t_list.append(attn_t)
            value_list.append(vals)

        torch.cuda.empty_cache()

    rkr_features = torch.stack(feats, dim=0)  # [N, L, H, D]
    return rkr_features, np.array(labels), attn_at_t_list, value_list


class FewShotEpisodeRunner:
    """
    Manages few-shot episode creation, RelSpec selection, and probe training.
    """

    def __init__(self, model, tokenizer, device):
        """
        Initialize episode runner.

        Args:
            model: Patched LLM model
            tokenizer: Tokenizer for the model
            device: torch.device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def sample_episode(self, n_way: int, k_shot: int, q_queries: int, seed: int):
        """
        Sample a few-shot episode from FewRel dataset.

        Args:
            n_way: Number of classes
            k_shot: Number of support examples per class
            q_queries: Number of query examples per class
            seed: Random seed

        Returns:
            Tuple of (support_idx, query_idx, episode_classes, prompts)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        ds = load_dataset("nicpopovic/fewrel-parsed", split="val_wiki")
        ds = ds.filter(lambda x: x["entity_order"] == "ht")

        # Build label->indices mapping
        label_to_idx = {}
        for i, item in enumerate(ds):
            y = int(item["label"][1:].split(" -")[0])
            label_to_idx.setdefault(y, []).append(i)

        # Filter classes with enough examples
        valid = {
            y: idxs
            for y, idxs in label_to_idx.items()
            if len(idxs) >= k_shot + q_queries
        }

        # Sample episode classes
        rng = np.random.default_rng(seed)
        episode_classes = rng.choice(
            list(valid.keys()), size=n_way, replace=False
        ).tolist()

        # Sample support and query examples
        support_idx, query_idx = [], []
        for cls in episode_classes:
            idxs = valid[cls].copy()
            rng.shuffle(idxs)
            support_idx.extend(idxs[:k_shot])
            query_idx.extend(idxs[k_shot : k_shot + q_queries])

        support_idx = np.array(support_idx, dtype=np.int64)
        query_idx = np.array(query_idx, dtype=np.int64)
        episode_indices = np.concatenate([support_idx, query_idx])

        # Build prompts
        prompts = build_prompts_subset(ds, episode_indices, self.tokenizer)

        return support_idx, query_idx, episode_classes, prompts

    def compute_expertise(self, rkr_features, labels_np, episode_classes):
        """
        Compute RelSpec expertise scores (Average Precision).

        Args:
            rkr_features: [N, L, H, D] feature tensor
            labels_np: [N] labels array
            episode_classes: List of class IDs in episode

        Returns:
            expertise: [C, L, H, D] AP scores per class
        """
        labels = torch.from_numpy(labels_np)
        device = self.device
        rkr_features = rkr_features.to(device).float()
        labels = labels.to(device)

        C = len(episode_classes)
        feature_shape = rkr_features.shape[1:]  # [L, H, D]

        # Flatten all non-sample dimensions once so we sort the support set only
        # a single time for the whole episode instead of once per (layer, head).
        support_feats_flat = rkr_features.reshape(rkr_features.shape[0], -1)  # [N, L*H*D]
        _, idx = torch.sort(support_feats_flat, dim=0, descending=True)  # [N, L*H*D]

        expertise = torch.empty((C,) + feature_shape, dtype=torch.float32)
        for ci, cls in enumerate(episode_classes):
            labels_binary = (labels == cls).long()  # [N]
            ap = compute_ap_from_sortidx(idx, labels_binary)  # [L*H*D]
            expertise[ci] = ap.reshape(feature_shape).cpu()

        return expertise

    def select_relspec_neurons(self, expertise, episode_classes, n_relspec: int):
        """
        Select top RelSpec neurons for each class.

        Args:
            expertise: [C, L, H, D] AP scores
            episode_classes: List of class IDs
            n_relspec: Number of neurons to select per class

        Returns:
            List of dicts with keys: class, rank, i (layer), j (head), d (dim), ap (score)
        """
        selected = []
        C, L, H, D = expertise.shape

        for ci, cls in enumerate(episode_classes):
            flat = expertise[ci].reshape(-1)
            top_vals, top_idxs = torch.topk(flat, n_relspec)

            for rank in range(n_relspec):
                idx = int(top_idxs[rank].item())
                val = float(top_vals[rank].item())

                # Decode flat index to (layer, head, dim)
                l = idx // (H * D)
                rem = idx % (H * D)
                h = rem // D
                d = rem % D

                selected.append({
                    "class": int(cls),
                    "rank": int(rank),
                    "i": int(l),      # layer
                    "j": int(h),      # head
                    "d": int(d),      # dimension
                    "ap": val         # AP score
                })
        return selected

    def build_probe_features(self, rkr_features, indices, selected):
        """
        Build probe feature matrix from selected neurons.

        Args:
            rkr_features: [N, L, H, D] feature tensor
            indices: Indices to extract
            selected: List of dicts with keys: i (layer), j (head), d (dim), class

        Returns:
            Feature matrix [len(indices), M] where M = len(selected)
        """
        subset = rkr_features[indices]  # [N_subset, L, H, D]
        if len(selected) == 0:
            return torch.empty((subset.shape[0], 0), dtype=subset.dtype, device=self.device)

        layers = torch.tensor([int(sel["i"]) for sel in selected], dtype=torch.long)
        heads = torch.tensor([int(sel["j"]) for sel in selected], dtype=torch.long)
        dims = torch.tensor([int(sel["d"]) for sel in selected], dtype=torch.long)

        # Advanced indexing gathers all selected neuron dimensions in one shot.
        X = subset[:, layers, heads, dims]  # [N_subset, M]
        return X.to(self.device)

    def train_probe(self, X_support, y_support, n_way: int, num_epochs: int = 200):
        """
        Train a linear probe on support set.

        Args:
            X_support: [N_support, M] support features
            y_support: [N_support] support labels (local indices)
            n_way: Number of classes
            num_epochs: Training epochs

        Returns:
            Trained probe (nn.Linear)
        """
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
