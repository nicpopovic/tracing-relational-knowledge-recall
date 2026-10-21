"""
Few-shot training utilities for relation classification.

Implements episodic sampling and RelSpec-based few-shot training.
"""

import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .expertise import (
    compute_expertise_episode,
    select_relspec_neurons,
    dedupe_selected_neurons,
)
from .probe import LinearProbe, train_probe, evaluate_probe


def build_label_index(
    labels_np: np.ndarray,
    min_count: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Build mapping from label to sample indices.

    Args:
        labels_np: Array of labels [N]
        min_count: Minimum samples required per class

    Returns:
        Dictionary mapping label -> array of indices
    """
    label_to_indices = {}
    unique_labels = np.unique(labels_np)

    for y in unique_labels:
        idxs = np.where(labels_np == y)[0]
        if min_count is not None and len(idxs) < min_count:
            continue
        label_to_indices[int(y)] = idxs

    return label_to_indices


def sample_episode_indices(
    label_to_indices: Dict[int, np.ndarray],
    n_way: int,
    k_shot: int,
    q_queries: int,
    rng: np.random.Generator,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Sample indices for a single episode.

    Args:
        label_to_indices: Mapping from label to indices
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        q_queries: Number of query examples per class
        rng: Random number generator

    Returns:
        Tuple of (episode_classes, support_indices, query_indices)
    """
    classes = list(label_to_indices.keys())
    assert len(classes) >= n_way, "Not enough classes for n-way sampling"

    episode_classes = rng.choice(classes, size=n_way, replace=False).tolist()

    support_indices = []
    query_indices = []

    for cls in episode_classes:
        idxs = label_to_indices[cls].copy()
        assert (
            len(idxs) >= k_shot + q_queries
        ), f"Class {cls} too small for episode"

        rng.shuffle(idxs)
        support_cls = idxs[:k_shot]
        query_cls = idxs[k_shot : k_shot + q_queries]

        support_indices.extend(support_cls.tolist())
        query_indices.extend(query_cls.tolist())

    return (
        episode_classes,
        np.array(support_indices, dtype=np.int64),
        np.array(query_indices, dtype=np.int64),
    )


def build_relspec_features(
    rkr_features: torch.Tensor,
    indices: np.ndarray,
    selected_neurons: List[Dict],
    device: torch.device,
) -> torch.Tensor:
    """
    Build feature vectors from selected relation-specific neurons.

    Args:
        rkr_features: [N_total, ...] Feature tensor on CPU
        indices: Sample indices (support or query)
        selected_neurons: List of neuron dictionaries with coords
        device: Target device

    Returns:
        X: [N_subset, M] feature matrix where M = len(selected_neurons)
    """
    subset_feats = rkr_features[indices]  # [N_subset, ...]
    N_subset = subset_feats.shape[0]
    M = len(selected_neurons)

    X = torch.empty((N_subset, M), dtype=subset_feats.dtype)

    for m, spec in enumerate(selected_neurons):
        coords = spec["coords"]
        # Index into tensor using coordinates
        X[:, m] = subset_feats[(slice(None),) + coords]

    return X.to(device)


class FewShotTrainer:
    """
    Trainer for few-shot relation classification using RelSpec neurons.
    """

    def __init__(
        self,
        rkr_features: torch.Tensor,
        labels: np.ndarray,
        n_way: int = 5,
        k_shot: int = 5,
        q_queries: int = 15,
        n_relspec: int = 3000,
        remove_duplicates: bool = True,
        use_gpu: bool = True,
        lr: float = 1e-2,
        epochs: int = 200,
        seed: int = 42,
    ):
        """
        Initialize the few-shot trainer.

        Args:
            rkr_features: Feature tensor [N, ...] on CPU
            labels: Label array [N]
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            q_queries: Number of query examples per class
            n_relspec: Number of RelSpec neurons per class
            remove_duplicates: Whether to deduplicate selected neurons
            use_gpu: Whether to use GPU
            lr: Learning rate for probe training
            epochs: Number of epochs for probe training
            seed: Random seed
        """
        self.rkr_features = rkr_features
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.n_relspec = n_relspec
        self.remove_duplicates = remove_duplicates
        self.use_gpu = use_gpu
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

        # Set device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build class index
        min_per_class = k_shot + q_queries
        self.label_to_indices = build_label_index(labels, min_count=min_per_class)

        print(f"Eligible classes for episodes: {len(self.label_to_indices)}")

    def run_single_episode(
        self, episode_classes: List[int], support_idx: np.ndarray, query_idx: np.ndarray
    ) -> Dict:
        """
        Run a single episode: compute expertise, train probe, evaluate.

        Args:
            episode_classes: List of class IDs for this episode
            support_idx: Support set indices
            query_idx: Query set indices

        Returns:
            Dictionary with episode results
        """
        support_labels_all = torch.from_numpy(self.labels)
        query_labels = self.labels[query_idx]

        # (a) Compute expertise on support set
        expertise_epi = compute_expertise_episode(
            self.rkr_features,
            torch.from_numpy(support_idx),
            support_labels_all,
            episode_classes,
            use_gpu=self.use_gpu,
        )

        # (b) Select RelSpec neurons
        selected_neurons = select_relspec_neurons(
            expertise_epi, n_relspec=self.n_relspec, episode_classes=episode_classes
        )

        if self.remove_duplicates:
            selected_neurons = dedupe_selected_neurons(selected_neurons)

        # (c) Build RelSpec features
        X_support = build_relspec_features(
            self.rkr_features, support_idx, selected_neurons, self.device
        )
        X_query = build_relspec_features(
            self.rkr_features, query_idx, selected_neurons, self.device
        )

        # (d) Map global labels to local [0..n_way-1]
        episode_classes_sorted = sorted(episode_classes)
        label2local = {cls: i for i, cls in enumerate(episode_classes_sorted)}

        y_support_local = torch.tensor(
            [label2local[int(self.labels[i])] for i in support_idx],
            dtype=torch.long,
            device=self.device,
        )
        y_query_local = torch.tensor(
            [label2local[int(y)] for y in query_labels],
            dtype=torch.long,
            device=self.device,
        )

        # (e) Train linear probe
        M = X_support.shape[1]
        probe = LinearProbe(M, self.n_way).to(self.device)
        train_probe(
            probe, X_support, y_support_local, num_epochs=self.epochs, lr=self.lr
        )

        # (f) Evaluate on query set
        results = evaluate_probe(probe, X_query, y_query_local)
        results["num_selected_neurons"] = len(selected_neurons)

        return results

    def run_episodes(self, num_episodes: int) -> Dict:
        """
        Run multiple episodes and aggregate results.

        Args:
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with aggregated results
        """
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        rng = np.random.default_rng(self.seed)

        global_correct = 0
        global_total = 0
        per_episode_acc = []

        for ep in tqdm(range(num_episodes), desc="Few-shot episodes"):
            # Sample episode
            episode_classes, support_idx, query_idx = sample_episode_indices(
                self.label_to_indices, self.n_way, self.k_shot, self.q_queries, rng
            )

            # Run episode
            results = self.run_single_episode(episode_classes, support_idx, query_idx)

            # Aggregate
            per_episode_acc.append(results["accuracy"])
            global_correct += results["correct"]
            global_total += results["total"]

        overall_acc = global_correct / global_total

        return {
            "overall_accuracy": overall_acc,
            "mean_episode_accuracy": np.mean(per_episode_acc),
            "std_episode_accuracy": np.std(per_episode_acc),
            "per_episode_accuracies": per_episode_acc,
            "num_episodes": num_episodes,
        }

    def run_multi_seed(
        self, num_episodes: int, num_runs: int = 5, base_seed: int = 42
    ) -> Dict:
        """
        Run multiple trials with different seeds.

        Args:
            num_episodes: Number of episodes per run
            num_runs: Number of runs with different seeds
            base_seed: Base seed (will use base_seed+i for run i)

        Returns:
            Dictionary with aggregated results across runs
        """
        all_accuracies = []

        for i in range(num_runs):
            self.seed = base_seed + i
            print(f"\n=== Run {i+1}/{num_runs} (seed={self.seed}) ===")

            results = self.run_episodes(num_episodes)
            acc = results["overall_accuracy"]
            all_accuracies.append(acc)

            print(
                f"Run {i+1}: Accuracy = {acc:.4f}, "
                f"Episode mean = {results['mean_episode_accuracy']:.4f} "
                f"± {results['std_episode_accuracy']:.4f}"
            )

        return {
            "mean_accuracy": float(np.mean(all_accuracies)),
            "std_accuracy": float(np.std(all_accuracies)),
            "all_accuracies": all_accuracies,
            "num_runs": num_runs,
        }
