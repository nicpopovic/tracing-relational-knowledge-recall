"""HTML rendering of token-level scores for the Gradio apps."""

import torch
import numpy as np
from typing import Tuple, Optional


# Base CSS for all visualizations
BASE_CSS = """
<style>
.query {
  margin-bottom: 22px;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 10px;
}
.row2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.panel {
  border: 1px solid #eee;
  border-radius: 10px;
  padding: 10px;
}
.panel h4 {
  margin: 0 0 8px 0;
  font-size: 13px;
  font-weight: 600;
}
.prompt {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  white-space: pre-wrap;
  line-height: 1.55;
}
.tok { border-radius: 3px; }
.meta {
  font-size: 12px;
  margin-bottom: 10px;
  opacity: 0.9;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid #ddd;
}
</style>
"""


class TokenScoreRenderer:
    """
    Renders token scores as colored HTML spans.

    Supports different visualization modes:
    - TokenScore (red for positive, blue for negative)
    - Attention weights (red gradient for positive values only)
    """

    @staticmethod
    def score_to_rgba(
        score_norm: float,
        squared: bool = False,
        hide_negatives: bool = True
    ) -> str:
        """
        Convert normalized score to RGBA color string.

        Args:
            score_norm: Normalized score in [-1, 1]
            squared: Whether to square the score (preserving sign)
            hide_negatives: Whether to hide negative scores

        Returns:
            RGBA color string like "rgba(255, 80, 80, 0.5)"
        """
        score_norm = float(max(-1.0, min(1.0, score_norm)))

        if squared:
            score_norm = np.sign(score_norm) * (score_norm ** 2)

        a = min(0.85, 1.0 * abs(score_norm))

        if score_norm >= 0:
            # Positive scores: red
            return f"rgba(255, 80, 80, {a:.3f})"

        if hide_negatives:
            a = 0.0

        # Negative scores: blue
        return f"rgba(80, 120, 255, {a:.3f})"

    @staticmethod
    def find_skip_block(prompt: str) -> Tuple[int, int]:
        """
        Find the "Cutting Knowledge Date" and "Today Date" block to skip in rendering.

        Args:
            prompt: Full prompt text

        Returns:
            Tuple of (skip_start, skip_end) character positions, or (-1, -1) if not found
        """
        cut_pat = "Cutting Knowledge Date:"
        today_pat = "Today Date:"

        # Find start of "Cutting Knowledge Date:" line
        s0 = prompt.find(cut_pat)
        if s0 == -1:
            return (-1, -1)

        # Find end of that line
        e0 = prompt.find("\n", s0)
        if e0 == -1:
            e0 = len(prompt)

        # Find "Today Date:" line after cutting line
        s1 = prompt.find(today_pat, e0)
        if s1 == -1:
            return (-1, -1)

        e1 = prompt.find("\n", s1)
        if e1 == -1:
            e1 = len(prompt)

        # Consume any blank lines after Today Date
        e2 = e1
        while e2 < len(prompt) and prompt[e2] == "\n":
            e2 += 1

        return (s0, e2)

    def render_tokens(
        self,
        prompt: str,
        tokenized: dict,
        tail_char_span: Tuple[int, int],
        token_scores: torch.Tensor,
        exclude_first_token: bool = True,
        max_abs: Optional[float] = None,
        squared: bool = False,
        hide_negatives: bool = True,
    ) -> str:
        """
        Render prompt with token-level score highlighting.

        Args:
            prompt: Full prompt text
            tokenized: Tokenizer output with offset_mapping
            tail_char_span: (start, end) character positions of tail entity
            token_scores: [S] token scores
            exclude_first_token: Exclude first token from normalization
            max_abs: Max absolute score for normalization (auto-computed if None)
            squared: Square scores for visualization
            hide_negatives: Hide negative scores

        Returns:
            HTML string with colored token spans
        """
        tail_start, tail_end = tail_char_span
        offsets = tokenized["offset_mapping"][0]
        scores = token_scores.detach().cpu().numpy()

        # Find metadata block to skip
        skip_start, skip_end = self.find_skip_block(prompt)

        def in_skip_block(s, e):
            if skip_start == -1:
                return False
            return not (e <= skip_start or s >= skip_end)

        # Compute normalization factor if not provided
        if max_abs is None:
            vals = []
            for tid, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if s >= tail_end:
                    break
                if in_skip_block(s, e):
                    continue
                vals.append(float(scores[tid]) if tid < len(scores) else 0.0)

            if not vals:
                max_abs = 1.0
            else:
                if exclude_first_token and len(vals) > 1:
                    max_abs = max(1e-8, max(abs(v) for v in vals[1:]))
                else:
                    max_abs = max(1e-8, max(abs(v) for v in vals))
        else:
            max_abs = float(max_abs)
            if max_abs < 1e-8:
                max_abs = 1.0

        # Build HTML spans
        spans = []
        for tid, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s >= tail_end:
                break
            if in_skip_block(s, e):
                continue

            tok_text = prompt[s:e]
            is_tail = not (e <= tail_start or s >= tail_end)

            if is_tail:
                # Highlight tail entity in green
                bg = "rgba(140, 255, 140, 0.75)"
            else:
                sc = float(scores[tid]) if tid < len(scores) else 0.0
                bg = self.score_to_rgba(
                    sc / max_abs, squared=squared, hide_negatives=hide_negatives
                )

            tok_text_esc = (
                tok_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            spans.append(
                f'<span class="tok" data-token-id="{tid}" '
                f'style="background-color: {bg};" '
                f'title="tok={tid} score={float(scores[tid]) if tid < len(scores) else 0.0:.4g}">'
                f"{tok_text_esc}"
                f"</span>"
            )

        return f'<div class="prompt">{"".join(spans)}</div>'
