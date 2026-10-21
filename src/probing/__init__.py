"""
AP expertise and RelSpec selection, linear probe and episodic few-shot
training (Sec. 4.1, App. A).
"""

from .expertise import (
    compute_ap_from_sortidx,
    compute_expertise_episode,
    select_relspec_neurons,
    dedupe_selected_neurons,
)
from .probe import LinearProbe
from .training import (
    build_label_index,
    sample_episode_indices,
    build_relspec_features,
    FewShotTrainer,
)

__all__ = [
    "compute_ap_from_sortidx",
    "compute_expertise_episode",
    "select_relspec_neurons",
    "dedupe_selected_neurons",
    "LinearProbe",
    "build_label_index",
    "sample_episode_indices",
    "build_relspec_features",
    "FewShotTrainer",
]
