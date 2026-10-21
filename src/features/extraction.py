"""
Main feature extraction orchestration.

This module provides the FeatureExtractor class that handles the full pipeline
of loading data, running models, and extracting features.
"""

import os
import re
import torch
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.util.data import mask_text
from .attention import (
    extract_per_head_per_token_attention,
    extract_layer_token_attention,
)
from .mlp import (
    extract_mlp_per_head_per_token,
    extract_mlp_layer_token,
    extract_mlp_total,
)


class FeatureExtractor:
    """
    Unified feature extractor for relational knowledge recall experiments.

    Handles data loading, prompt construction, model inference, and feature extraction
    for various feature types defined in the paper.
    """

    FEATURE_TYPES = {
        "att_h": "Per-head attention contribution (Δ_att,h)",
        "att_e1_h": "Per-head per-token attention contribution (Δ_att,e1,h)",
        "att_total": "Total attention contribution (baseline)",
        "att_e1": "Total token contribution (Δ_att,e1)",
        "mlp_h": "MLP per-head contribution (Δ_MLP,h)",
        "mlp_e1_h": "MLP per-head per-token contribution (Δ_MLP,e1,h)",
        "mlp_e1": "MLP layer-token contribution (Δ_MLP,e1)",
        "mlp_total": "MLP state at the recall position (baseline)",
    }

    def __init__(
        self,
        model_name: str,
        dataset_name: str = "nicpopovic/fewrel-parsed",
        split: str = "val_wiki",
        device: Optional[str] = None,
        device_map_auto: bool = False,
    ):
        """
        Initialize the feature extractor.

        Only examples in which the first-mentioned entity is the subject
        (entity order "ht") are used. Prompts covering both entity orders are
        built by ``src.episodes.prompts``.

        Args:
            model_name: HuggingFace model identifier
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            device: Device to run model on (None = auto-detect)
            device_map_auto: Spread the model over all visible GPUs (device_map="auto")
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.device_map_auto = device_map_auto

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = self._load_model(
            model_name,
            device_map="auto" if self.device_map_auto else None,
        )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if not self.device_map_auto:
            self.model.to(self.device)

        # Load dataset
        self.dataset = self._load_and_filter_dataset()

        # Get relation labels
        self.relations = sorted(list(set(self.dataset["label"])))

    def _load_model(self, model_name: str, device_map=None):
        """Load the appropriate model based on model name."""
        if "llama" in model_name.lower():
            from src.probingmods.llama import get_model
        elif "qwen3" in model_name.lower():
            from src.probingmods.qwen3 import get_model
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

        model = get_model(model_name, device_map=device_map)
        model.eval()
        return model

    def _load_and_filter_dataset(self):
        """Load the dataset, keeping examples where e1 is the subject (order "ht")."""
        ds = load_dataset(self.dataset_name, split=self.split)
        return ds.filter(lambda x: x["entity_order"] == "ht")

    def _create_prompts(self) -> List[Dict[str, Any]]:
        """
        Create prompts with RKR position annotations.

        Returns:
            List of prompt dictionaries containing:
                - prompt: Full text prompt
                - tokenized: Tokenized prompt with offset mapping
                - rkr_pos: List of RKR positions with metadata
        """
        prompts = []

        for i, item in enumerate(self.dataset):
            text = item["text"]

            # Prepare entity positions
            item["entities_pos"] = item["h"] + item["t"]
            item["entities_ids"] = ["h"] * len(item["h"]) + ["t"] * len(item["t"])
            item["entities_sf"] = [text[x:y] for (x, y) in item["entities_pos"]]

            # Create masked text
            masked_text = mask_text(item)

            # Construct messages for chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Fill in the blanks in the following text:\n\n"
                        f"{masked_text}\n\n"
                        f"Provide only the completed text as your answer."
                    ),
                },
                {"role": "assistant", "content": text},
            ]

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Localize text position in prompt
            escaped_text = re.escape(text)
            match = re.search(escaped_text, prompt)
            if not match:
                continue

            start, end = match.start(), match.end()

            # Determine RKR position: right before the second-mentioned entity
            # (the tail t); e1 is the last mention of the head h before it.
            rel_start, rel_end = item["entities_pos"][len(item["h"])]
            head_end = item["entities_pos"][len(item["h"]) - 1][1]

            # Adjust to prompt positions
            rel_start += start
            rel_end += start
            head_end += start

            rkr_pos = [
                {
                    "start": rel_start,
                    "head_end": head_end,
                    "property": item["label"],
                }
            ]

            # Tokenize with offset mapping
            tokenized = self.tokenizer(
                prompt, return_offsets_mapping=True, return_tensors="pt"
            )

            # Map character positions to token IDs
            for rkr in rkr_pos:
                char_start = rkr["start"]
                for idx, (offset_start, offset_end) in enumerate(
                    tokenized["offset_mapping"][0]
                ):
                    if offset_start <= char_start < offset_end:
                        # -1 because RKR happens right before entity generation
                        rkr["token_id"] = idx - 1
                        break

                # Last token of e1: entity spans are end-exclusive, so look up
                # the token containing the span's final character.
                char_head_last = rkr["head_end"] - 1
                for idx, (offset_start, offset_end) in enumerate(
                    tokenized["offset_mapping"][0]
                ):
                    if offset_start <= char_head_last < offset_end:
                        rkr["token_id_head_end"] = idx
                        break

            # Verify token positions
            valid_rkrs = []
            for rkr in rkr_pos:
                if "token_id" not in rkr:
                    continue

                token_id = rkr["token_id"]
                _, offset_end = tokenized["offset_mapping"][0][token_id]

                try:
                    assert (
                        prompt[:offset_end].strip()[-10:]
                        == prompt[: rkr["start"]].strip()[-10:]
                    ), "Mismatch in RKR char and token positions"
                    valid_rkrs.append(rkr)
                except (AssertionError, IndexError):
                    continue

            if not valid_rkrs:
                continue

            prompts.append(
                {
                    "prompt": prompt,
                    "tokenized": tokenized,
                    "rkr_pos": valid_rkrs,
                }
            )

        return prompts

    def extract_features(
        self,
        feature_type: str,
        output_dir: Optional[str] = None,
        save: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Extract features of the specified type.

        Args:
            feature_type: Type of feature to extract (see FEATURE_TYPES)
            output_dir: Directory to save features (auto-generated if None)
            save: Whether to save features to disk

        Returns:
            Tuple of (features_tensor, labels_list)
        """
        if feature_type not in self.FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature type: {feature_type}. "
                f"Valid types: {list(self.FEATURE_TYPES.keys())}"
            )

        print(f"Extracting features: {self.FEATURE_TYPES[feature_type]}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name} (split: {self.split})")
        print("Entity order: ht")

        # Create prompts
        print("Creating prompts...")
        prompts = self._create_prompts()
        print(f"Created {len(prompts)} valid prompts")
        expected_feature_rows = sum(len(prompt["rkr_pos"]) for prompt in prompts)
        print(f"Expected feature rows: {expected_feature_rows}")

        # Extract features
        print(f"Extracting {feature_type} features...")
        rkr_features = None
        rkr_labels = []
        feature_write_idx = 0

        for prompt in tqdm(prompts, desc="Processing prompts"):
            input_ids = prompt["tokenized"]["input_ids"].to(self.device)
            attention_mask = prompt["tokenized"]["attention_mask"].to(self.device)

            # Run model
            with torch.no_grad():
                _ = self.model(
                    input_ids, attention_mask, output_attentions=True, use_cache=True
                )

            # For attention and MLP per-head features, extract all head writes once per prompt
            # to avoid re-stacking the large tensor for each RKR position
            if feature_type in ("att_h", "att_total", "mlp_h"):
                all_head_writes = torch.stack([
                    layer.self_attn._last_head_writes.cpu()
                    for layer in self.model.model.layers
                ])  # (num_layers, batch_size, seq_len, num_heads, head_dim)

            # Clear CUDA cache immediately after forward pass
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Extract features for each RKR position
            for rkr in prompt["rkr_pos"]:
                token_id = rkr["token_id"]
                source_token_id = rkr.get("token_id_head_end")
                prop = rkr["property"]

                # Extract based on feature type
                if feature_type == "att_h":
                    # Select head writes at RKR position: (num_layers, num_heads, head_dim)
                    # Clone so the large all_head_writes tensor can be freed
                    features = all_head_writes[:, 0, token_id, :, :].clone()
                elif feature_type == "att_e1_h":
                    if source_token_id is None:
                        continue
                    features = extract_per_head_per_token_attention(
                        self.model, token_id, source_token_id
                    )
                elif feature_type == "att_total":
                    # Sum head writes across heads: (num_layers, head_dim)
                    # Sum creates a new tensor, no clone needed
                    features = all_head_writes[:, 0, token_id, :, :].sum(dim=1)
                elif feature_type == "att_e1":
                    if source_token_id is None:
                        continue
                    features = extract_layer_token_attention(
                        self.model, token_id, source_token_id
                    )
                elif feature_type == "mlp_h":
                    # Extract MLP per-head features inline to avoid re-stacking
                    from .mlp import _get_propagate_function
                    propagate_fn = _get_propagate_function(self.model)

                    mlp_features_list = []
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        # Clone so the large all_head_writes tensor can be freed
                        writes_layer = all_head_writes[layer_idx, 0, token_id, :, :].clone()
                        num_heads = writes_layer.shape[0]

                        per_head_mlp = []
                        for h in range(num_heads):
                            delta_mlp_t = propagate_fn(layer, token_id, writes_layer[h])
                            per_head_mlp.append(delta_mlp_t)

                        mlp_features_list.append(torch.stack(per_head_mlp))

                    features = torch.stack(mlp_features_list)  # (num_layers, heads, d_ff)
                elif feature_type == "mlp_e1_h":
                    if source_token_id is None:
                        continue
                    features = extract_mlp_per_head_per_token(
                        self.model, token_id, source_token_id
                    )
                elif feature_type == "mlp_e1":
                    if source_token_id is None:
                        continue
                    features = extract_mlp_layer_token(
                        self.model, token_id, source_token_id
                    )
                elif feature_type == "mlp_total":
                    features = extract_mlp_total(self.model, token_id)
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")

                features_cpu = features.cpu()
                if rkr_features is None:
                    # Preallocate the full output tensor as soon as we know the
                    # per-example feature shape. This avoids building a long
                    # Python list of tensors and, more importantly, avoids the
                    # large end-of-run `torch.stack(...)` memory spike that can
                    # OOM larger models right after the final prompt completes.
                    rkr_features = torch.empty(
                        (expected_feature_rows, *features_cpu.shape),
                        dtype=features_cpu.dtype,
                    )
                rkr_features[feature_write_idx].copy_(features_cpu)
                feature_write_idx += 1
                # Extract numeric property ID (e.g., "P26" -> 26)
                rkr_labels.append(int(prop[1:].split(" -")[0]))

            # Clear CUDA cache after processing prompt
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if rkr_features is None:
            raise RuntimeError("No features were extracted from the provided prompts")

        # Each prompt contributes one feature row; trim if any were skipped.
        if feature_write_idx != expected_feature_rows:
            rkr_features = rkr_features[:feature_write_idx]

        print(f"Extracted features shape: {rkr_features.shape}")
        print(f"Number of labels: {len(rkr_labels)}")

        # Save if requested
        if save:
            self._save_features(rkr_features, rkr_labels, feature_type, output_dir)

        return rkr_features, rkr_labels

    def _save_features(
        self,
        features: torch.Tensor,
        labels: List[int],
        feature_type: str,
        output_dir: Optional[str] = None,
    ):
        """Save extracted features and labels to disk."""
        if output_dir is None:
            # Auto-generate output directory
            output_dir = (
                f"data_parsed/{self.dataset_name.split('/')[-1]}-"
                f"{self.split}-ht/"
                f"{self.model_name.replace('/', '_')}"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Save label map
        label_map_path = os.path.join(output_dir, "label_map.txt")
        if not os.path.exists(label_map_path):
            with open(label_map_path, "w") as f:
                for relation in self.relations:
                    f.write(relation + "\n")

        # Save features
        model_name_safe = self.model_name.replace("/", "_")
        save_path = os.path.join(
            output_dir, f"rkr_features_{feature_type}_{model_name_safe}.pt"
        )

        torch.save((features, labels), save_path)
        print(f"Saved features to: {save_path}")

    @staticmethod
    def list_feature_types():
        """Print available feature types."""
        print("Available feature types:")
        for key, desc in FeatureExtractor.FEATURE_TYPES.items():
            print(f"  {key:12s} - {desc}")
