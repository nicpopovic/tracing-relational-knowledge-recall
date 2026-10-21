#!/usr/bin/env python3
"""
Feature extraction for relational knowledge recall.

Extracts the representations defined in Section 3 of "Tracing
Relational Knowledge Recall in Large Language Models" at the relational
knowledge recall position, for FewRel examples where the first-mentioned
entity is the subject.

Usage:
    python extract_features.py --model_name meta-llama/Llama-3.1-8B-Instruct --feature_type att_h
    python extract_features.py --list_features
"""

import argparse
from src.features import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for relational knowledge recall experiments"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nicpopovic/fewrel-parsed",
        help="HuggingFace dataset identifier"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_wiki",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=list(FeatureExtractor.FEATURE_TYPES.keys()),
        help="Type of feature to extract"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save features (auto-generated if not specified)"
    )
    parser.add_argument(
        "--list_features",
        action="store_true",
        help="List available feature types and exit"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if not specified)"
    )
    parser.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Use Hugging Face device_map=auto to spread the model across visible GPUs"
    )

    args = parser.parse_args()

    if args.list_features:
        FeatureExtractor.list_feature_types()
        return

    if not args.model_name:
        parser.error("--model_name is required")
    if not args.feature_type:
        parser.error("--feature_type is required")

    print("=" * 80)
    print("Feature Extraction for Relational Knowledge Recall")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name} (split: {args.split})")
    print(f"Feature type: {args.feature_type}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("=" * 80)

    extractor = FeatureExtractor(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        device=args.device,
        device_map_auto=args.device_map_auto,
    )

    features, labels = extractor.extract_features(
        feature_type=args.feature_type,
        output_dir=args.output_dir,
        save=True,
    )

    print("\n" + "=" * 80)
    print("Feature extraction complete!")
    print(f"Features shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
