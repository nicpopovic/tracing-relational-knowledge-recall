"""Episode sampling shared by the online 16-way experiments (Sec. 4.2, 4.3)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_valid_class_index(ds, k_shot: int, q_queries: int) -> Dict[int, List[int]]:
    """
    Build a class -> dataset-index mapping with episode-size filtering.

    Only classes with at least `k_shot + q_queries` examples can be sampled for
    an episode, because each episode needs both support and query members.
    """
    label_to_idx: Dict[int, List[int]] = {}
    for i, item in enumerate(ds):
        y = int(item["label"][1:].split(" -")[0])
        label_to_idx.setdefault(y, []).append(i)

    min_needed = k_shot + q_queries
    return {y: idxs for y, idxs in label_to_idx.items() if len(idxs) >= min_needed}


def sample_episode_indices(
    valid: Dict[int, List[int]],
    n_way: int,
    k_shot: int,
    q_queries: int,
    rng: np.random.Generator,
):
    """
    Sample one few-shot episode from the already filtered class index.

    Returns:
        episode_classes, support_idx, query_idx
    """
    classes_all = sorted(valid.keys())
    if len(classes_all) < n_way:
        raise RuntimeError(f"Not enough classes: have {len(classes_all)}, need {n_way}")

    episode_classes = rng.choice(classes_all, size=n_way, replace=False).tolist()

    support_idx = []
    query_idx = []
    for cls in episode_classes:
        idxs = np.array(valid[cls], dtype=np.int64)
        rng.shuffle(idxs)
        support_idx.extend(idxs[:k_shot].tolist())
        query_idx.extend(idxs[k_shot : k_shot + q_queries].tolist())

    return (
        episode_classes,
        np.array(support_idx, dtype=np.int64),
        np.array(query_idx, dtype=np.int64),
    )
