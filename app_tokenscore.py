"""
Gradio app for RelSpec few-shot probing with TokenScore + GxA comparison.

Displays three panels side-by-side for each query example:
  - TokenScore  (probe-based, relation type classification)
  - GxA         (Geva et al. 2023, object/entity prediction)
  - Avg Attention (baseline)

Used for the TokenScore / GxA / attention comparison figures in Sec. 4.3 and
the appendix: TokenScore and GxA answer *different* questions.
"""

import argparse
import gradio as gr
import torch
import numpy as np
import random
from transformers import AutoTokenizer

from src.app import (
    BASE_CSS,
    FewShotEpisodeRunner,
    TokenScoreRenderer,
    extract_features_with_attention,
)
from src.attribution import compute_avg_attention, compute_tokenscore_value_aware
from src.attribution.gxa import compute_gxa_layerwise
from src.util.data import PROPERTY_NAMES


EXTRA_CSS = """
<style>
.row3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}
.note {
  font-size: 11px;
  color: #666;
  margin-top: 4px;
  font-style: italic;
}
</style>
"""


def load_model(model_name: str):
    if "llama" in model_name.lower():
        from src.probingmods.llama import get_model
    elif "qwen3" in model_name.lower():
        from src.probingmods.qwen3 import get_model
    else:
        raise NotImplementedError(f"Model {model_name} not supported")
    # device_map="auto" splits layers across all available GPUs (requires
    # `accelerate`).  Do NOT call .to(device) afterwards.
    device_map = "auto" if torch.cuda.is_available() else None
    model = get_model(model_name, device_map=device_map)
    model.eval()
    return model


def run_episode(
    n_way: int,
    k_shot: int,
    q_queries: int,
    n_relspec: int,
    seed: int,
    contrast: str = "logsumexp_other",
    tau: float = 1.0,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))

    n_way = int(n_way)
    k_shot = int(k_shot)
    q_queries = int(q_queries)
    n_relspec = int(n_relspec)

    support_idx, query_idx, episode_classes, prompts = runner.sample_episode(
        n_way=n_way, k_shot=k_shot, q_queries=q_queries, seed=seed
    )

    rkr_features, labels_np, attn_at_t_list, value_list = extract_features_with_attention(
        model, prompts
    )

    # Release cached GPU memory before the grad-enabled GxA forward passes.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset_indices = np.array([p["dataset_idx"] for p in prompts])
    support_mask = np.isin(dataset_indices, support_idx)
    query_mask = np.isin(dataset_indices, query_idx)

    X_support_full = rkr_features[support_mask]
    X_query_full = rkr_features[query_mask]
    y_support = labels_np[support_mask]

    attn_query_list = [a for a, m in zip(attn_at_t_list, query_mask) if m]
    value_query_list = [v for v, m in zip(value_list, query_mask) if m]
    query_prompts = [p for p, m in zip(prompts, query_mask) if m]

    expertise = runner.compute_expertise(X_support_full, y_support, episode_classes)
    selected = runner.select_relspec_neurons(expertise, episode_classes, n_relspec)

    Xs = runner.build_probe_features(
        X_support_full, np.arange(len(X_support_full)), selected
    )
    Xq = runner.build_probe_features(
        X_query_full, np.arange(len(X_query_full)), selected
    )

    cls_sorted = sorted(episode_classes)
    label2local = {c: i for i, c in enumerate(cls_sorted)}
    ys = torch.tensor([label2local[int(y)] for y in y_support], device=device)

    probe = runner.train_probe(Xs, ys, n_way=n_way)

    with torch.no_grad():
        preds_local = torch.argmax(probe(Xq.float()), dim=1).cpu().numpy()

    html_blocks = [BASE_CSS, EXTRA_CSS]
    renderer = TokenScoreRenderer()

    for qi, (p, pred_loc) in enumerate(zip(query_prompts, preds_local)):
        gold_prop = f'P{int(p["label"])}'
        pred_prop = f'P{int(cls_sorted[int(pred_loc)])}'
        gold_name = PROPERTY_NAMES.get(gold_prop, gold_prop)
        pred_name = PROPERTY_NAMES.get(pred_prop, pred_prop)
        correct = gold_prop == pred_prop
        mark = "✅" if correct else "❌"

        t = int(p["token_id"])

        # --- TokenScore (aggregated across layers) ---
        token_scores, _, _ = compute_tokenscore_value_aware(
            Xq_row=Xq[qi],
            attn_t=attn_query_list[qi],
            values_by_layer=value_query_list[qi],
            probe=probe,
            selected=selected,
            model=model,
            contrast=contrast,
            tau=float(tau),
            aggregate_layers=True,
        )

        # --- Average attention baseline ---
        attn_mean = compute_avg_attention(attn_query_list[qi])

        # --- GxA (separate grad-enabled forward pass) ---
        input_ids = p["tokenized"]["input_ids"].to(device)
        attn_mask = p["tokenized"]["attention_mask"].to(device)
        S = input_ids.shape[1]
        target_tok_id = int(input_ids[0, t + 1].item()) if t + 1 < S else -1

        with torch.enable_grad():
            gxa_by_layer, used_target = compute_gxa_layerwise(
                model=model,
                input_ids=input_ids,
                attention_mask=attn_mask,
                query_pos=t,
                target_token_id=target_tok_id,
            )  # [L, S]

        gxa_scores = gxa_by_layer.sum(dim=0)  # [S]
        total = gxa_scores.sum()
        if total > 1e-8:
            gxa_scores = gxa_scores / total

        try:
            target_tok_text = tokenizer.decode([used_target]).strip()
        except Exception:
            target_tok_text = str(used_target)

        # --- Render three panels ---
        badges = [
            f'<span class="badge">{mark}</span>',
            f'<span class="badge">({p["head_name"]}, {gold_name}, {p["tail_name"]})</span>',
            f'<span class="badge">pred: {pred_name}</span>',
        ]
        header = f'<div class="meta">{"".join(badges)}</div>'

        html_ts = renderer.render_tokens(
            prompt=p["prompt"], tokenized=p["tokenized"],
            tail_char_span=p["tail_char_span"], token_scores=token_scores,
            exclude_first_token=False, squared=False, hide_negatives=False,
        )
        html_gxa = renderer.render_tokens(
            prompt=p["prompt"], tokenized=p["tokenized"],
            tail_char_span=p["tail_char_span"], token_scores=gxa_scores,
            exclude_first_token=False, squared=False, hide_negatives=True,
        )
        html_attn = renderer.render_tokens(
            prompt=p["prompt"], tokenized=p["tokenized"],
            tail_char_span=p["tail_char_span"], token_scores=attn_mean,
            exclude_first_token=True, squared=False, hide_negatives=True,
        )

        card = f"""
        <div class="query">
          {header}
          <div class="row3">
            <div class="panel">
              <h4>TokenScore (relation type)</h4>
              <div class="note">"Which tokens identify the <em>{gold_name}</em> relation?"
              Blue = pushing toward a competing relation.</div>
              {html_ts}
            </div>
            <div class="panel">
              <h4>GxA — Geva et al. (2023) (object prediction)</h4>
              <div class="note">"Which tokens are needed to predict <em>{target_tok_text!r}</em>?"
              Unsigned sensitivity; cannot show negative contributions.</div>
              {html_gxa}
            </div>
            <div class="panel">
              <h4>Avg. Attention (baseline)</h4>
              {html_attn}
            </div>
          </div>
        </div>
        """
        html_blocks.append(card)

        del gxa_by_layer, gxa_scores

    return "".join(html_blocks)


# ---------------------------------------------------------------------------
# Model / runner setup
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
_parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                     help="HuggingFace model identifier (LLaMA or Qwen3)")
_parser.add_argument("--share", action="store_true",
                     help="Create a public Gradio share link (off by default)")
ARGS = _parser.parse_args()
MODEL_NAME = ARGS.model_name
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = load_model(MODEL_NAME)
# With device_map="auto" the model is already placed; derive the input
# device from the embedding layer (always on the first device).
# Fall back to MPS or CPU when CUDA is unavailable.
if torch.cuda.is_available():
    device = model.model.embed_tokens.weight.device
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
runner = FewShotEpisodeRunner(model, tokenizer, device)

print("Starting Gradio app (GxA variant)...")

with gr.Blocks(title="RelSpec Few-Shot Probing — TokenScore vs GxA") as app:
    gr.Markdown("## RelSpec Few-Shot Probing — TokenScore vs GxA")
    gr.Markdown(
        """
        Side-by-side comparison of TokenScore and GxA (Geva et al., 2023).

        These methods answer **different questions**:
        - **TokenScore** (left): "Which tokens identify the *relation type*?" — emphasises relation-context words. Blue = negative contribution.
        - **GxA** (centre): "Which tokens are needed to predict *this specific entity*?" — emphasises subject entity. Unsigned; no negatives.
        - **Avg Attention** (right): Baseline.

        🔴 Red: Positive  |  🔵 Blue: Negative (TokenScore only)  |  🟢 Green: Tail entity
        """
    )

    with gr.Row():
        n_way = gr.Number(value=5, label="n-way", info="Number of classes")
        k_shot = gr.Number(value=5, label="k-shot", info="Support examples per class")
        q_queries = gr.Number(value=1, label="q-queries", info="Query examples per class")
        n_relspec = gr.Number(value=3000, label="n_relspec", info="RelSpec neurons per class")
        seed = gr.Number(value=42, label="seed", info="Random seed")
        contrast = gr.Dropdown(
            choices=["runner_up", "mean_other", "logsumexp_other"],
            value="logsumexp_other",
            label="contrast",
            info="Probe weight contrast method",
        )
        tau = gr.Number(value=1.0, label="tau", info="Temperature for logsumexp")

    run_btn = gr.Button("Run Episode", variant="primary")
    output = gr.HTML(label="TokenScore | GxA | Avg Attention")

    run_btn.click(
        fn=run_episode,
        inputs=[n_way, k_shot, q_queries, n_relspec, seed, contrast, tau],
        outputs=output,
    )

if __name__ == "__main__":
    app.launch(share=ARGS.share)
