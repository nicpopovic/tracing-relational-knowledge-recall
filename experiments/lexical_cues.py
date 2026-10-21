#!/usr/bin/env python3
"""
TokenScore vs episode-local lexical cues (Sec. 4.3, Table 3), 16-way 5-shot
over both entity orders. Lexical profiles are TF-IDF contrasts fit on the
support prompts; each query's TokenScore over its visible prompt prefix is
compared with the predicted relation's profile (rho, Mass, StrongAlign).

Outputs (in --output_dir): query_metrics.jsonl, summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attribution import compute_tokenscore_value_aware  # noqa: E402
from src.episodes import (  # noqa: E402
    FastFewShotEpisodeRunner,
    build_orderfix_prompts_subset,
    ensure_padding_token,
    extract_att_h_orderfix_batched,
    extract_features_with_attention_orderfix_batched,
    load_model,
)
from experiments.common import build_valid_class_index, sample_episode_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenScore vs episode-local lexical cues")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model id or local path (LLaMA or Qwen3)")
    parser.add_argument("--dataset_name", type=str, default="nicpopovic/fewrel-parsed", help="FewRel dataset (Hugging Face id or local path)")
    parser.add_argument("--eval_split", type=str, default="val_wiki", help="One split or a comma-separated list")
    parser.add_argument("--n_way", type=int, default=16, help="Relations per episode")
    parser.add_argument("--k_shot", type=int, default=5, help="Support examples per relation")
    parser.add_argument("--q_queries", type=int, default=15, help="Query examples per relation")
    parser.add_argument("--n_relspec", type=int, default=3000, help="RelSpec features selected per relation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Seed for episode sampling (numpy) and probe initialization (torch)")
    parser.add_argument("--extraction_batch_size", type=int, default=1, help="Prompts per forward pass during feature extraction")
    parser.add_argument("--expertise_chunk_size", type=int, default=4096, help="Feature columns per chunk in the AP computation (memory only)")
    parser.add_argument("--contrast", type=str, default="logsumexp_other", choices=["logsumexp_other", "runner_up", "mean_other"], help="Probe contrast; logsumexp_other is the paper's softmax-weighted contrast")
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for --contrast logsumexp_other")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available, else cpu)")
    parser.add_argument("--device_map_auto", action="store_true", help="Split the model across GPUs with device_map='auto'")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--top_fraction", type=float, default=0.10, help="Fraction of visible tokens compared in the top-k overlap metric")
    parser.add_argument("--alignment_spearman_threshold", type=float, default=0.30, help="StrongAlign: minimum Spearman rho")
    parser.add_argument("--alignment_mass_threshold", type=float, default=0.50, help="StrongAlign: minimum Mass")
    parser.add_argument("--alignment_margin_threshold", type=float, default=0.0, help="Lexical-shortcut flag: TF-IDF similarity margin must exceed this")
    return parser.parse_args()


def visible_prefix_token_ids(prompt: dict) -> list[int]:
    input_ids = prompt["tokenized"]["input_ids"][0].tolist()
    token_id = int(prompt["token_id"])
    return [int(tok) for tok in input_ids[: token_id + 1]]


def sparse_mean(dicts: Sequence[dict[int, float]]) -> dict[int, float]:
    if not dicts:
        return {}
    acc: dict[int, float] = defaultdict(float)
    for row in dicts:
        for key, value in row.items():
            acc[int(key)] += float(value)
    denom = float(len(dicts))
    return {key: value / denom for key, value in acc.items()}


def sparse_subtract(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    keys = set(a) | set(b)
    out = {}
    for key in keys:
        out[int(key)] = float(a.get(key, 0.0) - b.get(key, 0.0))
    return out


def sparse_norm(a: dict[int, float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in a.values()))


def sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for key, value in a.items():
        dot += float(value) * float(b.get(key, 0.0))
    denom = sparse_norm(a) * sparse_norm(b)
    if denom <= 0.0:
        return 0.0
    return float(dot / denom)


def build_episode_tfidf_model(support_prompts: Sequence[dict]) -> dict[str, Any]:
    docs = []
    for prompt in support_prompts:
        token_ids = visible_prefix_token_ids(prompt)
        docs.append(
            {
                "label": int(prompt["label"]),
                "dataset_idx": int(prompt["dataset_idx"]),
                "token_ids": token_ids,
            }
        )

    num_docs = len(docs)
    doc_freq: Counter[int] = Counter()
    for doc in docs:
        doc_freq.update(set(doc["token_ids"]))

    idf = {
        int(token_id): float(math.log((1.0 + num_docs) / (1.0 + df)) + 1.0)
        for token_id, df in doc_freq.items()
    }

    doc_vectors = []
    for doc in docs:
        counts = Counter(int(tok) for tok in doc["token_ids"])
        length = max(1, len(doc["token_ids"]))
        tfidf = {
            int(token_id): float((count / length) * idf[int(token_id)])
            for token_id, count in counts.items()
            if int(token_id) in idf
        }
        row = dict(doc)
        row["tfidf"] = tfidf
        doc_vectors.append(row)

    labels = sorted({int(doc["label"]) for doc in doc_vectors})
    docs_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for doc in doc_vectors:
        docs_by_label[int(doc["label"])].append(doc)

    class_profiles = {
        int(label): sparse_mean([doc["tfidf"] for doc in docs_by_label[int(label)]])
        for label in labels
    }

    all_class_sum: dict[int, float] = defaultdict(float)
    for profile in class_profiles.values():
        for token_id, value in profile.items():
            all_class_sum[int(token_id)] += float(value)

    contrast_profiles: dict[int, dict[int, float]] = {}
    num_labels = len(labels)
    for label in labels:
        this_profile = class_profiles[int(label)]
        if num_labels <= 1:
            other_mean = {}
        else:
            other_mean = {
                int(token_id): float((all_class_sum[int(token_id)] - this_profile.get(int(token_id), 0.0)) / (num_labels - 1))
                for token_id in all_class_sum
            }
        contrast_profiles[int(label)] = sparse_subtract(this_profile, other_mean)

    return {
        "idf": idf,
        "doc_vectors": doc_vectors,
        "docs_by_label": docs_by_label,
        "class_profiles": class_profiles,
        "contrast_profiles": contrast_profiles,
    }


def build_query_tfidf_vector(token_ids: Sequence[int], idf: dict[int, float]) -> dict[int, float]:
    counts = Counter(int(tok) for tok in token_ids if int(tok) in idf)
    length = max(1, len(token_ids))
    return {
        int(token_id): float((count / length) * idf[int(token_id)])
        for token_id, count in counts.items()
    }


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, float | None]:
    if len(x) < 2 or len(y) < 2:
        return None, None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None, None
    corr, p_value = spearmanr(x_arr, y_arr)
    if np.isnan(corr):
        return None, None
    return float(corr), float(p_value)


def top_overlap_fraction(
    lexical_scores: np.ndarray,
    token_scores: np.ndarray,
    top_fraction: float,
) -> tuple[float | None, int]:
    lexical_pos = np.flatnonzero(lexical_scores > 0.0)
    token_pos = np.flatnonzero(token_scores > 0.0)
    if len(lexical_pos) == 0 or len(token_pos) == 0:
        return None, 0

    k = max(1, int(math.ceil(len(token_scores) * float(top_fraction))))
    actual_k = min(k, len(lexical_pos), len(token_pos))
    if actual_k <= 0:
        return None, 0

    lex_ranked = lexical_pos[np.argsort(lexical_scores[lexical_pos])[::-1][:actual_k]]
    tok_ranked = token_pos[np.argsort(token_scores[token_pos])[::-1][:actual_k]]
    overlap = len(set(int(x) for x in lex_ranked.tolist()) & set(int(x) for x in tok_ranked.tolist()))
    return float(overlap / actual_k), int(actual_k)


def positive_mass_fraction(token_scores: np.ndarray, lexical_scores: np.ndarray) -> float | None:
    positive = np.clip(token_scores, a_min=0.0, a_max=None)
    denom = float(np.sum(positive))
    if denom <= 0.0:
        return None
    mask = lexical_scores > 0.0
    return float(np.sum(positive[mask]) / denom)


def similarity_stats_for_predicted_relation(
    query_vector: dict[int, float],
    predicted_label: int,
    docs_by_label: dict[int, list[dict[str, Any]]],
) -> dict[str, float | int | None]:
    pred_docs = docs_by_label.get(int(predicted_label), [])
    if not pred_docs:
        return {
            "pred_mean_cosine": None,
            "pred_max_cosine": None,
            "best_other_mean_cosine": None,
            "best_other_max_cosine": None,
            "mean_margin": None,
            "max_margin": None,
        }

    pred_cosines = [sparse_cosine(query_vector, doc["tfidf"]) for doc in pred_docs]
    pred_mean = float(mean(pred_cosines)) if pred_cosines else None
    pred_max = float(max(pred_cosines)) if pred_cosines else None

    other_means = []
    other_maxes = []
    for label, docs in docs_by_label.items():
        if int(label) == int(predicted_label):
            continue
        cosines = [sparse_cosine(query_vector, doc["tfidf"]) for doc in docs]
        if not cosines:
            continue
        other_means.append(float(mean(cosines)))
        other_maxes.append(float(max(cosines)))

    best_other_mean = max(other_means) if other_means else None
    best_other_max = max(other_maxes) if other_maxes else None
    mean_margin = None if pred_mean is None or best_other_mean is None else float(pred_mean - best_other_mean)
    max_margin = None if pred_max is None or best_other_max is None else float(pred_max - best_other_max)

    return {
        "pred_mean_cosine": pred_mean,
        "pred_max_cosine": pred_max,
        "best_other_mean_cosine": best_other_mean,
        "best_other_max_cosine": best_other_max,
        "mean_margin": mean_margin,
        "max_margin": max_margin,
    }


def token_texts(tokenizer, token_ids: Sequence[int]) -> list[str]:
    return [tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in token_ids]


def top_token_rows(
    tokenizer,
    token_ids: Sequence[int],
    scores: Sequence[float],
    max_items: int = 8,
    positive_only: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    indexed = list(enumerate(zip(token_ids, scores)))
    if positive_only:
        indexed = [(pos, pair) for pos, pair in indexed if float(pair[1]) > 0.0]
    indexed.sort(key=lambda item: float(item[1][1]), reverse=True)
    for pos, (token_id, score) in indexed[:max_items]:
        rows.append(
            {
                "position": int(pos),
                "token_id": int(token_id),
                "token_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                "score": float(score),
            }
        )
    return rows


def summarize_metric(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(len(vals)),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
    }


def fraction_where(rows: Sequence[dict[str, Any]], predicate) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "fraction": None}
    hits = sum(1 for row in rows if predicate(row))
    return {
        "count": int(len(rows)),
        "fraction": float(hits / len(rows)),
    }


def summarize_subset(
    rows: Sequence[dict[str, Any]],
    *,
    spearman_threshold: float,
    mass_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    rows = list(rows)
    return {
        "count": int(len(rows)),
        "lexical_tokenscore_spearman": summarize_metric(rows, "lexical_tokenscore_spearman"),
        "positive_tokenscore_mass_on_lexical_tokens": summarize_metric(rows, "positive_tokenscore_mass_on_lexical_tokens"),
        "topk_overlap_fraction": summarize_metric(rows, "topk_overlap_fraction"),
        "similarity_pred_mean_cosine": summarize_metric(rows, "similarity_pred_mean_cosine"),
        "similarity_pred_max_cosine": summarize_metric(rows, "similarity_pred_max_cosine"),
        "similarity_mean_margin": summarize_metric(rows, "similarity_mean_margin"),
        "similarity_max_margin": summarize_metric(rows, "similarity_max_margin"),
        "positive_spearman_fraction": fraction_where(
            [row for row in rows if row.get("lexical_tokenscore_spearman") is not None],
            lambda row: float(row["lexical_tokenscore_spearman"]) > 0.0,
        ),
        "strong_alignment_fraction": fraction_where(
            [
                row
                for row in rows
                if row.get("lexical_tokenscore_spearman") is not None
                and row.get("positive_tokenscore_mass_on_lexical_tokens") is not None
            ],
            lambda row: (
                float(row["lexical_tokenscore_spearman"]) >= spearman_threshold
                and float(row["positive_tokenscore_mass_on_lexical_tokens"]) >= mass_threshold
            ),
        ),
        "positive_margin_fraction": fraction_where(
            [row for row in rows if row.get("similarity_max_margin") is not None],
            lambda row: float(row["similarity_max_margin"]) > margin_threshold,
        ),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_padding_token(tokenizer)

    print("Loading model...")
    model = load_model(args.model_name, device_map_auto=args.device_map_auto)
    if not args.device_map_auto:
        model.to(device)

    runner = FastFewShotEpisodeRunner(model, tokenizer, device, expertise_chunk_size=args.expertise_chunk_size)

    print("Loading dataset...")
    ds = load_dataset(args.dataset_name, split=args.eval_split)
    valid = build_valid_class_index(ds, k_shot=args.k_shot, q_queries=args.q_queries)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    query_metrics_path = output_dir / "query_metrics.jsonl"
    query_metrics_file = query_metrics_path.open("w", encoding="utf-8", buffering=1)

    all_query_rows: list[dict[str, Any]] = []
    episode_runtimes: list[float] = []

    try:
        for episode_idx in range(int(args.episodes)):
            episode_start = time.time()
            print(f"[Episode {episode_idx + 1}/{args.episodes}] sampling...", flush=True)
            episode_classes, support_idx, query_idx = sample_episode_indices(
                valid=valid,
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_queries=args.q_queries,
                rng=rng,
            )

            support_prompts = build_orderfix_prompts_subset(ds, support_idx, tokenizer)
            query_prompts = build_orderfix_prompts_subset(ds, query_idx, tokenizer)
            if len(support_prompts) != len(support_idx):
                raise RuntimeError(
                    f"Orderfix support prompt extraction dropped examples: expected {len(support_idx)}, got {len(support_prompts)}"
                )
            if len(query_prompts) != len(query_idx):
                raise RuntimeError(
                    f"Orderfix query prompt extraction dropped examples: expected {len(query_idx)}, got {len(query_prompts)}"
                )

            print(f"[Episode {episode_idx + 1}/{args.episodes}] extracting support/query features...", flush=True)
            X_support_full, y_support = extract_att_h_orderfix_batched(
                model,
                tokenizer,
                support_prompts,
                batch_size=args.extraction_batch_size,
            )
            X_query_full, y_query, attn_query_list, value_query_list = extract_features_with_attention_orderfix_batched(
                model,
                tokenizer,
                query_prompts,
                batch_size=args.extraction_batch_size,
                store_attention=True,
            )

            print(f"[Episode {episode_idx + 1}/{args.episodes}] training probe...", flush=True)
            expertise = runner.compute_expertise(X_support_full, y_support, episode_classes)
            selected = runner.select_relspec_neurons(expertise, episode_classes, args.n_relspec)

            Xs = runner.build_probe_features(X_support_full, np.arange(len(X_support_full)), selected)
            Xq = runner.build_probe_features(X_query_full, np.arange(len(X_query_full)), selected)

            cls_sorted = sorted(int(cls) for cls in episode_classes)
            label2local = {int(cls): idx for idx, cls in enumerate(cls_sorted)}
            ys = torch.tensor([label2local[int(y)] for y in y_support], device=device, dtype=torch.long)

            probe = runner.train_probe(Xs, ys, n_way=args.n_way)
            with torch.no_grad():
                preds_local = torch.argmax(probe(Xq.float()), dim=1).cpu().numpy()

            lexical_model = build_episode_tfidf_model(support_prompts)

            for query_pos, prompt in enumerate(query_prompts):
                token_scores, _, _ = compute_tokenscore_value_aware(
                    Xq_row=Xq[query_pos],
                    attn_t=attn_query_list[query_pos],
                    values_by_layer=value_query_list[query_pos],
                    probe=probe,
                    selected=selected,
                    model=model,
                    contrast=args.contrast,
                    tau=float(args.tau),
                    aggregate_layers=True,
                )

                token_id = int(prompt["token_id"])
                visible_ids = visible_prefix_token_ids(prompt)
                visible_scores = token_scores[: token_id + 1].detach().cpu().numpy().astype(float)
                predicted_label = int(cls_sorted[int(preds_local[query_pos])])
                gold_label = int(y_query[query_pos])
                lexical_profile = lexical_model["contrast_profiles"][int(predicted_label)]
                lexical_scores = np.asarray(
                    [float(lexical_profile.get(int(token_id_value), 0.0)) for token_id_value in visible_ids],
                    dtype=float,
                )

                spearman_corr, spearman_p = safe_spearman(lexical_scores.tolist(), visible_scores.tolist())
                positive_mass = positive_mass_fraction(visible_scores, lexical_scores)
                overlap_fraction, overlap_k = top_overlap_fraction(
                    lexical_scores,
                    visible_scores,
                    top_fraction=args.top_fraction,
                )

                query_vector = build_query_tfidf_vector(visible_ids, lexical_model["idf"])
                sim_stats = similarity_stats_for_predicted_relation(
                    query_vector,
                    predicted_label,
                    lexical_model["docs_by_label"],
                )

                has_strong_alignment = (
                    spearman_corr is not None
                    and positive_mass is not None
                    and float(spearman_corr) >= float(args.alignment_spearman_threshold)
                    and float(positive_mass) >= float(args.alignment_mass_threshold)
                )
                has_positive_margin = (
                    sim_stats["max_margin"] is not None
                    and float(sim_stats["max_margin"]) > float(args.alignment_margin_threshold)
                )
                lexical_shortcut_suspected = (
                    not bool(gold_label == predicted_label)
                    and has_strong_alignment
                    and has_positive_margin
                )

                token_text_list = token_texts(tokenizer, visible_ids)
                row = {
                    "episode": int(episode_idx),
                    "dataset_idx": int(prompt["dataset_idx"]),
                    "gold_label": int(gold_label),
                    "predicted_label": int(predicted_label),
                    "correct": bool(gold_label == predicted_label),
                    "visible_token_count": int(len(visible_ids)),
                    "lexical_tokenscore_spearman": spearman_corr,
                    "lexical_tokenscore_spearman_p": spearman_p,
                    "positive_tokenscore_mass_on_lexical_tokens": positive_mass,
                    "topk_overlap_fraction": overlap_fraction,
                    "topk_overlap_k": int(overlap_k),
                    "similarity_pred_mean_cosine": sim_stats["pred_mean_cosine"],
                    "similarity_pred_max_cosine": sim_stats["pred_max_cosine"],
                    "similarity_best_other_mean_cosine": sim_stats["best_other_mean_cosine"],
                    "similarity_best_other_max_cosine": sim_stats["best_other_max_cosine"],
                    "similarity_mean_margin": sim_stats["mean_margin"],
                    "similarity_max_margin": sim_stats["max_margin"],
                    "has_strong_lexical_alignment": bool(has_strong_alignment),
                    "has_positive_similarity_margin": bool(has_positive_margin),
                    "lexical_shortcut_suspected": bool(lexical_shortcut_suspected),
                    "visible_prefix_text": tokenizer.decode(visible_ids, skip_special_tokens=False),
                    "top_lexical_tokens": top_token_rows(tokenizer, visible_ids, lexical_scores.tolist(), positive_only=True),
                    "top_tokenscore_tokens": top_token_rows(tokenizer, visible_ids, visible_scores.tolist(), positive_only=True),
                    "visible_token_texts": token_text_list,
                }
                query_metrics_file.write(json.dumps(row) + "\n")
                all_query_rows.append(row)

            episode_runtime = float(time.time() - episode_start)
            episode_runtimes.append(episode_runtime)
            print(f"[Episode {episode_idx + 1}/{args.episodes}] done in {episode_runtime:.1f}s", flush=True)
    finally:
        query_metrics_file.close()

    summary = {
        "config": {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "eval_split": args.eval_split,
            "n_way": int(args.n_way),
            "k_shot": int(args.k_shot),
            "q_queries": int(args.q_queries),
            "n_relspec": int(args.n_relspec),
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "contrast": args.contrast,
            "tau": float(args.tau),
            "extraction_batch_size": int(args.extraction_batch_size),
            "expertise_chunk_size": int(args.expertise_chunk_size),
            "text_unit": "actual_model_visible_orderfix_prompt_prefix_up_to_token_id",
            "tfidf_fit_scope": "support_only",
            "lexical_target": "predicted_relation_only",
            "alignment_spearman_threshold": float(args.alignment_spearman_threshold),
            "alignment_mass_threshold": float(args.alignment_mass_threshold),
            "alignment_margin_threshold": float(args.alignment_margin_threshold),
        },
        "query_count": int(len(all_query_rows)),
        "episode_count": int(len(episode_runtimes)),
        "runtime_seconds_total": float(sum(episode_runtimes)),
        "runtime_seconds_mean_per_episode": float(mean(episode_runtimes)) if episode_runtimes else None,
        "projected_runtime_seconds_100episodes": float(mean(episode_runtimes) * 100.0) if episode_runtimes else None,
        "all_queries": summarize_subset(
            all_query_rows,
            spearman_threshold=float(args.alignment_spearman_threshold),
            mass_threshold=float(args.alignment_mass_threshold),
            margin_threshold=float(args.alignment_margin_threshold),
        ),
        "correct_queries": summarize_subset(
            [row for row in all_query_rows if bool(row["correct"])],
            spearman_threshold=float(args.alignment_spearman_threshold),
            mass_threshold=float(args.alignment_mass_threshold),
            margin_threshold=float(args.alignment_margin_threshold),
        ),
        "incorrect_queries": summarize_subset(
            [row for row in all_query_rows if not bool(row["correct"])],
            spearman_threshold=float(args.alignment_spearman_threshold),
            mass_threshold=float(args.alignment_mass_threshold),
            margin_threshold=float(args.alignment_margin_threshold),
        ),
        "lexical_shortcut_suspected_queries": summarize_subset(
            [row for row in all_query_rows if bool(row["lexical_shortcut_suspected"])],
            spearman_threshold=float(args.alignment_spearman_threshold),
            mass_threshold=float(args.alignment_mass_threshold),
            margin_threshold=float(args.alignment_margin_threshold),
        ),
        "lexical_shortcut_suspected_fraction": fraction_where(
            all_query_rows,
            lambda row: bool(row["lexical_shortcut_suspected"]),
        ),
        "valid_counts": {
            "lexical_tokenscore_spearman": int(sum(row.get("lexical_tokenscore_spearman") is not None for row in all_query_rows)),
            "positive_tokenscore_mass_on_lexical_tokens": int(sum(row.get("positive_tokenscore_mass_on_lexical_tokens") is not None for row in all_query_rows)),
            "topk_overlap_fraction": int(sum(row.get("topk_overlap_fraction") is not None for row in all_query_rows)),
            "similarity_max_margin": int(sum(row.get("similarity_max_margin") is not None for row in all_query_rows)),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote query metrics to {query_metrics_path}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
