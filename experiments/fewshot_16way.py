#!/usr/bin/env python3
"""
16-way 5-shot relation classification over both entity orders, with per-query
HeadScore concentration k_X (Sec. 4.2: Table 2, per-property appendix tables,
input to Fig. 3). Δ_att,h is extracted online per episode with the order-fix
prompt; RelSpec features are selected on the support set and a linear probe is
trained and evaluated on the queries.

Outputs (in --output_dir):
    rkr_features_orderfix_online_<n>way_<k>shot_n<m>_classification_report.json
    head_contribution_share_query_metrics.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attribution import compute_head_contribution_share_batch, heads_needed_for_share  # noqa: E402
from src.episodes import OrderfixFastFewShotEpisodeRunner, ensure_padding_token, load_model  # noqa: E402
from experiments.common import build_valid_class_index, sample_episode_indices  # noqa: E402


def parse_thresholds(text: str) -> List[float]:
    """Parse a comma-separated threshold string like '0.5,0.8,0.9,0.95'."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Invalid threshold {value}; expected 0 < x <= 1")
        values.append(value)
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def build_global_relation_report(gold_global: list[int], pred_global: list[int]) -> dict:
    """Per-class precision/recall/F1 over all pooled queries, keyed by "P<id>"."""
    labels = sorted({int(x) for x in gold_global} | {int(x) for x in pred_global})
    report: dict[str, dict] = {}
    for label in labels:
        tp = fp = fn = 0
        support = 0
        for pred, gold in zip(pred_global, gold_global):
            pred = int(pred)
            gold = int(gold)
            if gold == label:
                support += 1
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        report[f"P{label}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return report


def collect_episode_head_contribution_stats(
    *,
    runner,
    X_support_full: torch.Tensor,
    y_support: np.ndarray,
    X_query_full: torch.Tensor,
    query_prompts: Sequence[Dict],
    episode_classes: Sequence[int],
    n_way: int,
    k_shot: int,
    q_queries: int,
    n_relspec: int,
    contrast: str,
    tau: float,
    thresholds: Sequence[float],
    device: torch.device,
    episode_idx: int,
    query_metrics_file,
    pooled_gold: list[int],
    pooled_pred: list[int],
) -> None:
    """RelSpec selection, probe training and per-query prediction + k_X for one episode."""
    expertise = runner.compute_expertise(X_support_full, y_support, episode_classes)
    selected = runner.select_relspec_neurons(expertise, episode_classes, n_relspec)

    Xs = runner.build_probe_features(
        X_support_full,
        np.arange(len(X_support_full)),
        selected,
    )
    cls_sorted = sorted(episode_classes)
    label2local = {c: i for i, c in enumerate(cls_sorted)}
    ys = torch.tensor([label2local[int(y)] for y in y_support], device=device)
    probe = runner.train_probe(Xs, ys, n_way=n_way)

    Xq_all = runner.build_probe_features(
        X_query_full,
        np.arange(len(X_query_full)),
        selected,
    )

    _, head_shares_batch, pred_local_batch, _ = compute_head_contribution_share_batch(
        X_rows=Xq_all,
        probe=probe,
        selected=selected,
        contrast=contrast,
        tau=tau,
        aggregate_mode="net_abs",
    )
    pred_local_np = pred_local_batch.detach().cpu().numpy().astype(np.int64)
    head_shares_np = head_shares_batch.detach().cpu().numpy()

    for qi, prompt in enumerate(query_prompts):
        pred_local = int(pred_local_np[qi])
        head_shares = head_shares_np[qi]

        gold = int(prompt["label"])
        pred = int(cls_sorted[pred_local])
        is_correct = bool(gold == pred)

        pooled_gold.append(gold)
        pooled_pred.append(pred)

        row = {
            "episode": int(episode_idx),
            "dataset_idx": int(prompt["dataset_idx"]),
            "gold_label": int(gold),
            "pred_label": int(pred),
            "is_correct": is_correct,
            "n_way": int(n_way),
            "k_shot": int(k_shot),
            "q_queries": int(q_queries),
            "n_relspec": int(n_relspec),
            "contrast": contrast,
            "tau": float(tau),
            "num_layers": int(head_shares.shape[0]),
            "num_heads": int(head_shares.shape[1]),
            "total_selected_features": int(len(selected)),
            # k_X: smallest number of heads whose share of |HeadScore| reaches X.
            "threshold_counts": {
                f"{thr:.2f}": int(heads_needed_for_share(torch.as_tensor(head_shares), thr))
                for thr in thresholds
            },
        }
        query_metrics_file.write(json.dumps(row) + "\n")

    del Xq_all, Xs, expertise, selected, ys, probe
    del head_shares_batch, pred_local_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="16-way 5-shot classification with per-query HeadScore k_X")
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
    parser.add_argument("--thresholds", type=str, default="0.5,0.8,0.9,0.95,0.99", help="Comma-separated fractions X for k_X")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available, else cpu)")
    parser.add_argument("--device_map_auto", action="store_true", help="Split the model across GPUs with device_map='auto'")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_padding_token(tokenizer)

    print("Loading model...")
    model = load_model(args.model_name, device_map_auto=args.device_map_auto)
    if not args.device_map_auto:
        model.to(device)
    runner = OrderfixFastFewShotEpisodeRunner(
        model,
        tokenizer,
        device,
        expertise_chunk_size=args.expertise_chunk_size,
        extraction_batch_size=args.extraction_batch_size,
    )

    print("Loading dataset...")
    split_names = [part.strip() for part in str(args.eval_split).split(",") if part.strip()]
    if not split_names:
        raise ValueError("At least one eval split is required")
    if len(split_names) == 1:
        ds = load_dataset(args.dataset_name, split=split_names[0])
    else:
        ds = concatenate_datasets([load_dataset(args.dataset_name, split=name) for name in split_names])
    valid = build_valid_class_index(ds, k_shot=args.k_shot, q_queries=args.q_queries)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    job_start_time = time.time()

    pooled_gold: list[int] = []
    pooled_pred: list[int] = []

    query_metrics_path = out_dir / "head_contribution_share_query_metrics.jsonl"
    query_metrics_file = open(query_metrics_path, "w", encoding="utf-8", buffering=1)

    try:
        for ep in range(args.episodes):
            print(f"[Episode {ep+1}/ {args.episodes}] Sampling and running episode...", flush=True)
            episode_classes, support_idx, query_idx = sample_episode_indices(
                valid=valid,
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_queries=args.q_queries,
                rng=rng,
            )

            X_support_full, y_support, _ = runner.extract_online_att_h(ds, support_idx)
            X_query_full, _, query_prompts = runner.extract_online_att_h(ds, query_idx)

            collect_episode_head_contribution_stats(
                runner=runner,
                X_support_full=X_support_full,
                y_support=y_support,
                X_query_full=X_query_full,
                query_prompts=query_prompts,
                episode_classes=episode_classes,
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_queries=args.q_queries,
                n_relspec=args.n_relspec,
                contrast=args.contrast,
                tau=args.tau,
                thresholds=thresholds,
                device=device,
                episode_idx=ep,
                query_metrics_file=query_metrics_file,
                pooled_gold=pooled_gold,
                pooled_pred=pooled_pred,
            )

            query_metrics_file.flush()
            os.fsync(query_metrics_file.fileno())

            elapsed = time.time() - job_start_time
            eta = elapsed / (ep + 1) * (args.episodes - ep - 1)
            print(f"[Episode {ep+1}/ {args.episodes}] done | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m", flush=True)
    finally:
        query_metrics_file.close()

    report_path = out_dir / f"rkr_features_orderfix_online_{args.n_way}way_{args.k_shot}shot_n{args.n_relspec}_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(build_global_relation_report(pooled_gold, pooled_pred), f, indent=2)


if __name__ == "__main__":
    main()
