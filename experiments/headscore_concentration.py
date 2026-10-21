#!/usr/bin/env python3
"""
HeadScore concentration per relation (Sec. 4.2, Fig. 3). Averages the per-query
k_X written by `experiments/fewshot_16way.py` per gold relation and reports its
Spearman correlation with probe precision, F1 and Wikidata output range, plus
mean k_X for correct vs incorrect predictions.

Output: --output_dir/head_concentration_summary_threshold_<100*X>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.util.data import PROPERTY_NAMES  # noqa: E402


def threshold_key(value: float) -> str:
    """Threshold key format used in `threshold_counts`."""
    return f"{value:.2f}"


def load_query_metrics(path: str) -> List[dict]:
    """Read the JSONL per-query metrics file into memory."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ranges(path: str) -> Dict[str, int]:
    """Load relation output ranges ("P<id>" -> number of distinct objects)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items() if v is not None}


def compute_relation_means(rows: List[dict], threshold: float) -> Dict[int, float]:
    """Mean k_X per gold relation."""
    key = threshold_key(threshold)
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["gold_label"])].append(float(row["threshold_counts"][key]))
    return {label: float(np.mean(vals)) for label, vals in grouped.items()}


def correct_vs_incorrect(rows: List[dict], threshold: float) -> dict:
    """Query-level k_X for correct vs incorrect predictions."""
    key = threshold_key(threshold)
    correct_vals = [float(r["threshold_counts"][key]) for r in rows if bool(r["is_correct"])]
    incorrect_vals = [float(r["threshold_counts"][key]) for r in rows if not bool(r["is_correct"])]
    return {
        "correct_count": len(correct_vals),
        "incorrect_count": len(incorrect_vals),
        "correct_mean": float(np.mean(correct_vals)) if correct_vals else None,
        "incorrect_mean": float(np.mean(incorrect_vals)) if incorrect_vals else None,
    }


def spearman_summary(xs: List[float], ys: List[float]) -> dict:
    rho, p = spearmanr(xs, ys) if len(xs) >= 2 else (None, None)
    return {
        "rho": None if rho is None else float(rho),
        "p": None if p is None else float(p),
        "n": len(xs),
    }


def main():
    parser = argparse.ArgumentParser(description="HeadScore concentration (k_X) per relation")
    parser.add_argument("--query_metrics_path", type=str, required=True,
                        help="head_contribution_share_query_metrics.jsonl from fewshot_16way.py")
    parser.add_argument("--classification_report_path", type=str, required=True,
                        help="Per-class report JSON from fewshot_16way.py")
    parser.add_argument("--range_json", type=str, default="experiments/wikidata_ranges.json",
                        help="Wikidata output range per property (default: the shipped wikidata_ranges.json)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Contribution-share threshold X for k_X (paper: 0.95)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_query_metrics(args.query_metrics_path)
    if not rows:
        raise RuntimeError("No rows found in query metrics file")

    key = threshold_key(args.threshold)
    if key not in rows[0]["threshold_counts"]:
        raise KeyError(
            f"Threshold {key} not present in metrics file. "
            "Re-run fewshot_16way.py with a matching --thresholds argument."
        )

    relation_means = compute_relation_means(rows, args.threshold)
    with open(args.classification_report_path, "r", encoding="utf-8") as f:
        per_class = json.load(f)
    range_by_prop = load_ranges(args.range_json)

    # Mean k_X vs probe precision / F1 (percentages).
    metric_correlations = {}
    for metric in ("precision", "f1"):
        xs_m, ys_m = [], []
        for rel_label, mean_k in sorted(relation_means.items()):
            prop_id = f"P{rel_label}"
            if prop_id not in per_class:
                continue
            xs_m.append(mean_k)
            ys_m.append(float(per_class[prop_id][metric]) * 100.0)
        metric_correlations[metric] = spearman_summary(xs_m, ys_m)

    # Output range vs mean k_X.
    xs_range, ys_range = [], []
    for rel_label, mean_k in sorted(relation_means.items()):
        prop_id = f"P{rel_label}"
        if prop_id not in range_by_prop:
            continue
        xs_range.append(float(range_by_prop[prop_id]))
        ys_range.append(mean_k)

    summary = {
        "threshold": float(args.threshold),
        "query_count": len(rows),
        "relation_count": len(relation_means),
        "spearman_vs_precision": metric_correlations["precision"],
        "spearman_vs_f1": metric_correlations["f1"],
        "spearman_vs_output_range": spearman_summary(xs_range, ys_range),
        "correct_vs_incorrect": correct_vs_incorrect(rows, args.threshold),
        "relation_mean_heads_needed": {
            f"P{label}": {
                "name": PROPERTY_NAMES.get(f"P{label}", f"P{label}"),
                "avg_heads_needed": float(value),
            }
            for label, value in sorted(relation_means.items())
        },
    }

    summary_path = out_dir / f"head_concentration_summary_threshold_{int(round(args.threshold * 100))}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
