"""
Expertise-based neuron selection for relation classification.

Implements the RelSpec (Relation-Specific) neuron selection method based on
Average Precision (AP) scores, following Liu et al. 2025.
"""

import torch
from typing import List, Dict, Any


def compute_ap_from_sortidx(
    idx: torch.Tensor, labels_binary: torch.Tensor
) -> torch.Tensor:
    """
    Compute Average Precision from sorted indices.

    AP = sum_i P(i) * (R(i) - R(i-1)) with R(0) = 0, i.e. the same quantity as
    ``sklearn.metrics.average_precision_score`` (up to tie handling: tied
    activations are ranked in sort order rather than grouped).

    Args:
        idx: [N_support, D] sort indices per neuron (descending activation)
        labels_binary: [N_support] binary labels in {0,1}

    Returns:
        [D] AP score for each neuron
    """
    sorted_labels = labels_binary[idx]  # [N_support, D]

    tp = sorted_labels.cumsum(dim=0)
    fp = (1 - sorted_labels).cumsum(dim=0)

    precision = tp / (tp + fp + 1e-12)
    n_pos = labels_binary.sum() + 1e-12
    recall = tp / n_pos

    recall_prev = torch.cat([torch.zeros_like(recall[:1]), recall[:-1]], dim=0)
    ap = (precision * (recall - recall_prev)).sum(dim=0)
    return ap


def compute_expertise_episode(
    rkr_features: torch.Tensor,
    support_indices: torch.Tensor,
    support_labels: torch.Tensor,
    episode_classes: List[int],
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    Compute expertise scores (AP) for neurons in an episode.

    For each neuron and each class in the episode, computes how well that neuron
    ranks positive examples of that class above negative examples.

    Args:
        rkr_features: [N, ...] Feature tensor on CPU (can be 3D or 4D)
        support_indices: Indices used as support in this episode
        support_labels: [N] Labels for all samples (global)
        episode_classes: List of global class IDs for this episode (length = N_way)
        use_gpu: Whether to use GPU for computation

    Returns:
        expertise: Tensor of shape [N_way, ...] with same structure as features,
                   containing AP per neuron per episode class
    """
    # Get feature dimensions
    N_total = rkr_features.shape[0]
    feature_shape = rkr_features.shape[1:]  # Remaining dimensions

    # Select support features and labels
    support_feats = rkr_features[support_indices]  # [N_support, ...]
    support_labels_episode = support_labels[support_indices]  # [N_support]

    # Setup device
    if use_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    support_labels_gpu = support_labels_episode.to(dev)
    classes = torch.tensor(episode_classes, dtype=torch.long, device=dev)
    C_epi = len(episode_classes)

    # Initialize expertise tensor
    expertise = torch.zeros((C_epi,) + feature_shape, dtype=torch.float32)

    # Flatten all dimensions except the first (samples)
    support_feats_flat = support_feats.reshape(support_feats.shape[0], -1)  # [N_support, D]
    D = support_feats_flat.shape[1]

    # Move to GPU
    support_feats_gpu = support_feats_flat.float().to(dev)

    # Sort features by activation value
    _, idx = torch.sort(support_feats_gpu, dim=0, descending=True)  # [N_support, D]

    # Compute AP for each class
    for ci, cls in enumerate(classes):
        labels_binary = (support_labels_gpu == cls).long()  # [N_support]
        ap = compute_ap_from_sortidx(idx, labels_binary)  # [D]

        # Reshape back to the feature shape
        ap_reshaped = ap.cpu().reshape(feature_shape)
        expertise[ci] = ap_reshaped

    return expertise


def select_relspec_neurons(
    expertise: torch.Tensor, n_relspec: int, episode_classes: List[int]
) -> List[Dict[str, Any]]:
    """
    Select top relation-specific neurons per class based on expertise scores.

    Args:
        expertise: Expertise tensor of shape [N_way, ...] (can be 3D or 4D)
        n_relspec: Number of neurons to select per class
        episode_classes: List of global class IDs (length = N_way)

    Returns:
        List of dictionaries describing selected neurons:
            {
              "class": int,      # Global class ID
              "rank": int,       # Rank within that class (0 to n_relspec-1)
              "coords": tuple,   # Neuron coordinates in feature tensor
              "ap": float        # AP score
            }
    """
    C_epi = expertise.shape[0]
    feature_shape = expertise.shape[1:]

    selected = []

    for ci in range(C_epi):
        # Flatten expertise for this class
        scores = expertise[ci].reshape(-1)  # [D]
        top_vals, top_idx = torch.topk(scores, k=min(n_relspec, len(scores)))
        cls_global = int(episode_classes[ci])

        for rank in range(len(top_vals)):
            idx = int(top_idx[rank].item())
            val = float(top_vals[rank].item())

            # Convert flat index back to multidimensional coordinates
            coords = []
            remaining = idx
            for dim_size in reversed(feature_shape):
                coords.insert(0, remaining % dim_size)
                remaining = remaining // dim_size

            selected.append(
                {
                    "class": cls_global,
                    "rank": int(rank),
                    "coords": tuple(coords),
                    "ap": val,
                }
            )

    return selected


def dedupe_selected_neurons(selected_neurons: List[Dict]) -> List[Dict]:
    """
    Remove duplicate neurons (by coordinates) across classes in an episode.

    When a neuron is selected by multiple classes, keeps only the first occurrence.

    Args:
        selected_neurons: List of neuron dictionaries

    Returns:
        Deduplicated list of neuron dictionaries
    """
    seen = set()
    deduped = []

    for entry in selected_neurons:
        key = entry["coords"]
        if key not in seen:
            seen.add(key)
            deduped.append(entry)

    return deduped
