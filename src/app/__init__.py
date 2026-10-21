"""Support code for the Gradio apps (`app_tokenscore*.py`, Sec. 4.3 figures)."""

from .rendering import BASE_CSS, TokenScoreRenderer
from .runner import FewShotEpisodeRunner, build_prompts_subset, extract_features_with_attention

__all__ = [
    "BASE_CSS",
    "TokenScoreRenderer",
    "FewShotEpisodeRunner",
    "build_prompts_subset",
    "extract_features_with_attention",
]
