"""
Gradio app for per-layer TokenScore + GxA visualization.

Displays three columns for each layer:
  - TokenScore  (probe-based, relation type classification)
  - GxA         (Geva et al. 2023, object/entity prediction)
  - Avg Attention (per-layer head mean)

Allows navigating between query examples within a few-shot episode.

This is the layerwise variant of `app_tokenscore.py` (TokenScore vs GxA).
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
from src.attribution import compute_tokenscore_value_aware
from src.attribution.gxa import compute_gxa_layerwise
from src.util.data import PROPERTY_NAMES


EXTRA_CSS = """
<style>
.row3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
.layer-row {
  margin-bottom: 4px;
}
.col-header {
  font-size: 13px;
  font-weight: 700;
  margin: 0 0 6px 0;
  padding: 4px 8px;
  border-radius: 6px;
  background: #f4f4f4;
}
.note {
  font-size: 11px;
  color: #666;
  font-style: italic;
  margin-bottom: 8px;
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


def run_episode_and_store(n_way, k_shot, q_queries, n_relspec, seed):
    """
    Run episode and store per-layer TokenScore, GxA, and attention for all queries.

    Returns a state dict with all data needed for interactive rendering.
    """
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

    queries = []
    for qi, p in enumerate(query_prompts):
        gold_prop = f'P{int(p["label"])}'
        pred_prop = f'P{int(cls_sorted[int(preds_local[qi])])}'

        t = int(p["token_id"])

        # --- TokenScore per layer: [L, S] ---
        ts_by_layer, _, _ = compute_tokenscore_value_aware(
            Xq_row=Xq[qi],
            attn_t=attn_query_list[qi],
            values_by_layer=value_query_list[qi],
            probe=probe,
            selected=selected,
            model=model,
            contrast="logsumexp_other",
            tau=1.0,
            aggregate_layers=False,
        )  # [L, S]

        # --- Average attention per layer: [L, S] ---
        attn_by_layer = attn_query_list[qi].float().mean(dim=1).cpu()  # [L, S]

        # --- GxA per layer: [L, S] (grad-enabled forward pass) ---
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

        try:
            target_tok_text = tokenizer.decode([used_target]).strip()
        except Exception:
            target_tok_text = str(used_target)

        queries.append({
            "prompt": p["prompt"],
            "tokenized": p["tokenized"],
            "tail_char_span": p["tail_char_span"],
            "token_id": t,                       # RKR position (last token before tail)
            "gold_prop": gold_prop,
            "pred_prop": pred_prop,
            "gold_name": PROPERTY_NAMES.get(gold_prop, gold_prop),
            "pred_name": PROPERTY_NAMES.get(pred_prop, pred_prop),
            "head_name": p["head_name"],
            "tail_name": p["tail_name"],
            "ts_by_layer": ts_by_layer,          # [L, S] CPU
            "gxa_by_layer": gxa_by_layer,        # [L, S] CPU
            "attn_by_layer": attn_by_layer,      # [L, S] CPU
            "target_tok_text": target_tok_text,
        })

    state = {
        "queries": queries,
        "num_layers": int(len(model.model.layers)),
        "query_count": int(len(queries)),
    }
    return state


def _build_top_token_table(q, L):
    """
    Build an HTML table and a Markdown string showing the top-scored token
    per layer for each of the three methods.

    'Top' = argmax over non-special, non-tail tokens (raw score, so for
    TokenScore the most positively contributing token is selected).

    Returns:
        table_html: str   — styled HTML <table>
        md_text:    str   — pipe-delimited Markdown table
    """
    prompt = q["prompt"]
    offsets = q["tokenized"]["offset_mapping"][0].tolist()
    tail_end = q["tail_char_span"][1]
    t = q["token_id"]  # RKR position: last token before the tail entity

    # Build token-text lookup; None = excluded (BOS/EOS or at/after tail).
    tok_texts = []
    for s, e in offsets:
        if s == e or s >= tail_end:
            tok_texts.append(None)
        else:
            tok_texts.append(prompt[s:e])

    def top_token_with_context(scores_1d):
        """
        Return (html_phrase, md_phrase, score) for the highest-scored token,
        flanked by its immediate left and right neighbours.
        The top token is bolded; neighbours are shown as plain text.

        Right neighbour: if it falls past the RKR position t (i.e. into the
        tail entity), show "[next token]" instead of the entity text.
        """
        scored = [
            (tid, sc)
            for tid, sc in enumerate(scores_1d.tolist())
            if tok_texts[tid] is not None
        ]
        if not scored:
            return "—", "—", 0.0

        scored.sort(key=lambda x: x[1], reverse=True)
        top_tid, best_sc = scored[0]
        top_text = tok_texts[top_tid]

        # Nearest non-excluded neighbour to the left
        left_text = None
        for prev in range(top_tid - 1, -1, -1):
            if tok_texts[prev] is not None:
                left_text = tok_texts[prev]
                break

        # Nearest non-excluded neighbour to the right.
        # Any position past t is the tail entity — show "[next token]" there.
        right_text = None
        for nxt in range(top_tid + 1, len(tok_texts)):
            if tok_texts[nxt] is not None:
                right_text = "[next token]" if nxt > t else tok_texts[nxt]
                break

        parts_html = []
        parts_md = []
        if left_text:
            parts_html.append(left_text)
            parts_md.append(left_text)
        parts_html.append(f"<b>{top_text}</b>")
        # Markdown bold (**text**) doesn't render if there's a leading/trailing
        # space inside the markers.  Move any whitespace outside the **.
        _inner = top_text.strip()
        _lead  = top_text[: len(top_text) - len(top_text.lstrip())]
        _trail = top_text[len(top_text.rstrip()):]
        parts_md.append(f"{_lead}**{_inner}**{_trail}")
        if right_text:
            parts_html.append(right_text)
            parts_md.append(right_text)

        return "".join(parts_html), "".join(parts_md), best_sc

    def esc_md(t):
        return t.replace("|", "\\|").replace("`", "'")

    # --- Extract the raw query sentence from the full prompt ---
    # LLaMA instruct format: <|start_header_id|>assistant<|end_header_id|>\n\nTEXT
    # Find the last assistant-response start before the tail entity.
    tail_start = q["tail_char_span"][0]
    marker = "<|end_header_id|>\n\n"
    last_marker = q["prompt"].rfind(marker, 0, tail_start)
    if last_marker >= 0:
        sent_start = last_marker + len(marker)
    else:
        last_nl = q["prompt"].rfind("\n", 0, tail_start)
        sent_start = last_nl + 1 if last_nl >= 0 else 0
    sentence = q["prompt"][sent_start:tail_start] + "[next token]"

    md_header = (
        f"Prompt:\n```\n{q['prompt']}\n```\n"
        f'Sentence: "{sentence}"\n'
        f'Triple: ({q["head_name"]}, {q["gold_name"]}, {q["tail_name"]})\n'
        f"Relation type: {q['gold_name']}\n\n"
    )

    html_rows = []
    md_rows = [
        "| Layer | TokenScore | GxA | Attn |",
        "|------:|:-----------|:----|:-----|",
    ]

    for l in range(L):
        ts_h,  ts_md,  ts_sc  = top_token_with_context(q["ts_by_layer"][l])
        gxa_h, gxa_md, gxa_sc = top_token_with_context(q["gxa_by_layer"][l])
        at_h,  at_md,  at_sc  = top_token_with_context(q["attn_by_layer"][l])

        html_rows.append(
            f"<tr>"
            f"<td style='text-align:right;padding:3px 8px;'>{l+1}</td>"
            f"<td style='padding:3px 8px;font-family:monospace'>{ts_h}</td>"
            f"<td style='text-align:right;padding:3px 8px;color:#b33'>{ts_sc:.3f}</td>"
            f"<td style='padding:3px 8px;font-family:monospace'>{gxa_h}</td>"
            f"<td style='text-align:right;padding:3px 8px;color:#b70'>{gxa_sc:.3f}</td>"
            f"<td style='padding:3px 8px;font-family:monospace'>{at_h}</td>"
            f"<td style='text-align:right;padding:3px 8px;color:#448'>{at_sc:.3f}</td>"
            f"</tr>"
        )

        md_rows.append(
            f"| {l+1} | {esc_md(ts_md)} | {esc_md(gxa_md)} | {esc_md(at_md)} |"
        )

    table_html = (
        "<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;'>"
        "<thead><tr style='background:#f0f0f0;'>"
        "<th style='padding:4px 8px;'>Layer</th>"
        "<th colspan='2' style='padding:4px 8px;color:#b33;'>TokenScore — top token</th>"
        "<th colspan='2' style='padding:4px 8px;color:#b70;'>GxA — top token</th>"
        "<th colspan='2' style='padding:4px 8px;color:#448;'>Attn — top token</th>"
        "</tr></thead>"
        "<tbody>" + "".join(html_rows) + "</tbody>"
        "</table>"
    )

    return table_html, md_header + "\n".join(md_rows)


def render_query_layerwise(state, q_idx):
    """Render selected query with all layers in three columns + top-token table."""
    if state is None or not state.get("queries"):
        return (0, "", "", "", "", "", "")

    q_idx = int(q_idx) % int(state["query_count"])
    q = state["queries"][q_idx]
    L = int(state["num_layers"])

    renderer = TokenScoreRenderer()
    correct = q["gold_prop"] == q["pred_prop"]
    mark = "✅" if correct else "❌"

    header_html = (
        BASE_CSS + EXTRA_CSS +
        f'<div class="meta">'
        f'<span class="badge">{mark}</span>'
        f'<span class="badge">({q["head_name"]}, {q["gold_name"]}, {q["tail_name"]})</span>'
        f'<span class="badge">pred: {q["pred_name"]}</span>'
        f'</div>'
    )

    # Global normalization across all layers (exclude token 0 for stability)
    max_ts = float(torch.abs(q["ts_by_layer"][:, 1:]).max())
    max_gxa = float(torch.abs(q["gxa_by_layer"][:, 1:]).max())
    max_attn = float(torch.abs(q["attn_by_layer"][:, 1:]).max())

    rows_ts = []
    rows_gxa = []
    rows_attn = []

    for l in range(L):
        lbl = f"<b>Layer {l+1:02d}:</b> "

        html_ts = lbl + renderer.render_tokens(
            prompt=q["prompt"], tokenized=q["tokenized"],
            tail_char_span=q["tail_char_span"],
            token_scores=q["ts_by_layer"][l],
            max_abs=max_ts, exclude_first_token=False,
            squared=False, hide_negatives=False,
        ).split("\n</span>")[-1]

        html_gxa = lbl + renderer.render_tokens(
            prompt=q["prompt"], tokenized=q["tokenized"],
            tail_char_span=q["tail_char_span"],
            token_scores=q["gxa_by_layer"][l],
            max_abs=max_gxa, exclude_first_token=False,
            squared=False, hide_negatives=True,
        ).split("\n</span>")[-1]

        html_attn = lbl + renderer.render_tokens(
            prompt=q["prompt"], tokenized=q["tokenized"],
            tail_char_span=q["tail_char_span"],
            token_scores=q["attn_by_layer"][l],
            max_abs=max_attn, exclude_first_token=False,
            squared=False, hide_negatives=True,
        ).split("\n</span>")[-1]

        rows_ts.append(html_ts)
        rows_gxa.append(html_gxa)
        rows_attn.append(html_attn)

    ts_out = (
        f'<div class="col-header">TokenScore (relation type)</div>'
        f'<div class="note">"Which tokens identify the {q["gold_name"]!r} relation?"</div>'
        + "<br>".join(rows_ts)
    )
    gxa_out = (
        f'<div class="col-header">GxA — Geva et al. 2023 (object prediction)</div>'
        f'<div class="note">"Which tokens predict {q["target_tok_text"]!r}?"</div>'
        + "<br>".join(rows_gxa)
    )
    attn_out = (
        f'<div class="col-header">Avg. Attention (baseline)</div>'
        + "<br>".join(rows_attn)
    )

    table_html, md_text = _build_top_token_table(q, L)

    return (q_idx, header_html, ts_out, gxa_out, attn_out, table_html, md_text)


_EMPTY = (0, "", "", "", "", "", "")


def next_query(state, q_idx):
    if not state or not state.get("query_count"):
        return _EMPTY
    return render_query_layerwise(state, int(q_idx) + 1)


def prev_query(state, q_idx):
    if not state or not state.get("query_count"):
        return _EMPTY
    return render_query_layerwise(state, int(q_idx) - 1)


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

NUM_LAYERS = len(model.model.layers)

print("Starting Gradio app (layerwise GxA variant)...")

with gr.Blocks(title="RelSpec Layer-wise — TokenScore vs GxA") as demo:
    gr.Markdown("## RelSpec Few-Shot Probing — Per-Layer TokenScore vs GxA")
    gr.Markdown(
        """
        Three columns per layer: **TokenScore** (relation type) | **GxA** (object prediction) | **Avg Attention** (baseline).

        - 🔴 **Red**: Positive contribution
        - 🔵 **Blue**: Negative / competing-class contribution (TokenScore only)
        - 🟢 **Green**: Tail entity

        GxA measures unsigned sensitivity to entity prediction; TokenScore measures signed contribution to relation classification.
        """
    )

    with gr.Row():
        n_way = gr.Number(value=5, label="n-way")
        k_shot = gr.Number(value=5, label="k-shot")
        q_queries = gr.Number(value=1, label="q-queries")
        n_relspec = gr.Number(value=3000, label="n_relspec")
        seed = gr.Number(value=42, label="seed")

    run_btn = gr.Button("Run Episode", variant="primary")

    state = gr.State(None)
    q_idx_state = gr.State(0)

    with gr.Row():
        prev_btn = gr.Button("◀ Previous Query")
        next_btn = gr.Button("Next Query ▶")

    header_html = gr.HTML()

    with gr.Row():
        ts_html = gr.HTML(label="TokenScore by Layer")
        gxa_html = gr.HTML(label="GxA by Layer")
        attn_html = gr.HTML(label="Avg Attention by Layer")

    gr.Markdown("### Top-scored token per layer")
    table_html = gr.HTML()

    with gr.Row():
        md_box = gr.Textbox(
            label="Markdown table",
            lines=6,
            interactive=False,
        )
        copy_btn = gr.Button("📋 Copy Markdown", scale=0)

    outputs = [q_idx_state, header_html, ts_html, gxa_html, attn_html, table_html, md_box]

    def run_and_render(n_way, k_shot, q_queries, n_relspec, seed):
        st = run_episode_and_store(n_way, k_shot, q_queries, n_relspec, seed)
        rendered = render_query_layerwise(st, 0)
        return (st,) + rendered

    run_btn.click(
        fn=run_and_render,
        inputs=[n_way, k_shot, q_queries, n_relspec, seed],
        outputs=[state] + outputs,
    )
    next_btn.click(fn=next_query, inputs=[state, q_idx_state], outputs=outputs)
    prev_btn.click(fn=prev_query, inputs=[state, q_idx_state], outputs=outputs)

    copy_btn.click(
        fn=None,
        inputs=[md_box],
        outputs=[],
        js="(md) => { navigator.clipboard.writeText(md); }",
    )

if __name__ == "__main__":
    demo.launch(share=ARGS.share)
