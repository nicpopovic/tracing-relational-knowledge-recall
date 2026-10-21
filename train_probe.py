#!/usr/bin/env python3
"""
Few-shot relation classification with RelSpec feature selection and linear probes.

For each episode, the top ``--n_relspec`` features per class are selected by AP
expertise on the support set, a linear probe is trained on the support set and
evaluated on the query set. Accuracy is pooled over all episodes of a run and
reported as mean/std over ``--num_runs`` seeds (base_seed + i).

Usage:
    python train_probe.py --data_path features.pt --n_way 5 --k_shot 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.probing import FewShotTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train few-shot relation classification probes"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to features .pt file (features, labels) from extract_features.py"
    )

    # Episode configuration
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support examples per class")
    parser.add_argument("--q_queries", type=int, default=15, help="Number of query examples per class")
    parser.add_argument("--n_episodes", type=int, default=500, help="Number of episodes per run")

    # RelSpec configuration
    parser.add_argument(
        "--n_relspec", type=int, default=3000, help="Number of RelSpec features per class"
    )
    parser.add_argument(
        "--no_remove_duplicates",
        action="store_false",
        dest="remove_duplicates",
        help="Keep features selected by several classes multiple times (default: de-duplicate)"
    )

    # Probe training
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (Adam)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")

    # Multi-seed evaluation
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs with different seeds")
    parser.add_argument(
        "--base_seed", type=int, default=42, help="Base random seed (run i uses base_seed + i)"
    )

    parser.add_argument(
        "--no_gpu", action="store_false", dest="use_gpu", help="Force CPU usage"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON (default: results_<data_stem>_<n>way_<k>shot_n<m>.json)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Few-Shot Relation Classification Training")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Episode config: {args.n_way}-way {args.k_shot}-shot, {args.q_queries} queries")
    print(f"Number of episodes: {args.n_episodes}")
    print(f"RelSpec features: {args.n_relspec} (de-duplicate: {args.remove_duplicates})")
    print(f"Probe training: lr={args.lr}, epochs={args.epochs}")
    print(f"Multi-seed: {args.num_runs} runs, base_seed={args.base_seed}")
    print(f"Device: {'GPU' if args.use_gpu else 'CPU'}")
    print("=" * 80)

    print("\nLoading features...")
    rkr_features, rkr_labels = torch.load(args.data_path, map_location="cpu")

    # Add head dimension for features without one (e.g. att_total, mlp_total)
    if len(rkr_features.shape) == 3:
        rkr_features = rkr_features.unsqueeze(1)

    if isinstance(rkr_labels, torch.Tensor):
        labels_np = rkr_labels.cpu().numpy()
    else:
        labels_np = np.asarray(rkr_labels, dtype=np.int64)

    print(f"Features shape: {rkr_features.shape}")
    print(f"Number of samples: {len(labels_np)}")
    print(f"Number of unique classes: {len(np.unique(labels_np))}")

    trainer = FewShotTrainer(
        rkr_features=rkr_features,
        labels=labels_np,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        n_relspec=args.n_relspec,
        remove_duplicates=args.remove_duplicates,
        use_gpu=args.use_gpu,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.base_seed,
    )

    print("\n" + "=" * 80)
    print("Starting multi-seed evaluation...")
    print("=" * 80)

    results = trainer.run_multi_seed(
        num_episodes=args.n_episodes,
        num_runs=args.num_runs,
        base_seed=args.base_seed,
    )

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{args.n_way}-way {args.k_shot}-shot:")
    print(f"  Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"  Std accuracy:  {results['std_accuracy']:.4f}")
    print(f"  All runs: {[f'{acc:.4f}' for acc in results['all_accuracies']]}")
    print("=" * 80)

    if args.output_path is None:
        data_stem = Path(args.data_path).stem
        args.output_path = (
            f"results_{data_stem}_{args.n_way}way_{args.k_shot}shot_"
            f"n{args.n_relspec}.json"
        )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "config": {
            "data_path": args.data_path,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "q_queries": args.q_queries,
            "n_episodes": args.n_episodes,
            "n_relspec": args.n_relspec,
            "remove_duplicates": args.remove_duplicates,
            "lr": args.lr,
            "epochs": args.epochs,
            "num_runs": args.num_runs,
            "base_seed": args.base_seed,
        },
        "results": results,
    }

    with open(args.output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
