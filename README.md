# Tracing Relational Knowledge Recall in Large Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2604.19934-b31b1b.svg)](https://arxiv.org/abs/2604.19934)
[![ACL](https://img.shields.io/badge/ACLAnthology-2026.findings--acl.2160-b31b1b.svg)](https://aclanthology.org/2026.findings-acl.2160/)
[![Website](https://img.shields.io/badge/🔗_Project_Page-Tracing-009688)](https://nicpopovic.com/publications/tracing)

Code for the paper "Tracing Relational Knowledge Recall in Large Language Models".

## Layout

- `src/probingmods/`: patched LLaMA / Qwen3 modules exposing per-head attention writes and MLP propagation (Sec. 3.1, 3.2)
- `src/features/`: features at the recall position: `att_h`, `att_e1_h`, `att_e1`, `att_total`, `mlp_h`, `mlp_e1_h`, `mlp_e1`, `mlp_total`
- `src/probing/`: AP expertise and RelSpec selection, linear probe, episodic training (Sec. 4.1)
- `src/attribution/`: TokenScore, HeadScore and k_X (Sec. 3.3), GxA baseline
- `src/episodes/`: order-fix prompts and the online 16-way episode runner (Sec. 4.2, 4.3)
- `src/app/`: episode runner and HTML rendering for the Gradio apps
- `extract_features.py`, `train_probe.py`: Sec. 4.1; `experiments/`: Sec. 4.2, 4.3

## Sec. 4.1: Few-shot relation classification (Table 1, appendix)

Defaults are the paper settings: m=3000 RelSpec features per class, q=15 queries per class, 500 episodes × 5 seeds (42..46), Adam lr 1e-2, 200 epochs.

```bash
M=meta-llama/Llama-3.1-8B-Instruct; MS=meta-llama_Llama-3.1-8B-Instruct
python extract_features.py --model_name $M --feature_type att_h   # any type from --list_features
python train_probe.py --n_way 5 --k_shot 5 \
  --data_path data_parsed/fewrel-parsed-val_wiki-ht/$MS/rkr_features_att_h_$MS.pt \
  --output_path results/fewshot/$MS/att_h_5way_5shot_n3000.json
```

Run every feature type with `--n_way {5,10} --k_shot {1,5}`; the RelSpec-size ablation uses `att_h` with `--n_relspec {200,1000,3000,5000}`. Accuracy: `results.mean_accuracy` / `results.std_accuracy`.

## Sec. 4.2: What affects classification accuracy? (Table 2, Fig. 3)

16-way 5-shot over both entity orders, q=15, 100 episodes, seed 42 (defaults). The order-fix prompt is the fill-in-the-blanks prompt, except that when the object precedes the subject only the object is blanked, so the probing position always directly precedes the object. `--contrast logsumexp_other` (default) is the paper's softmax-weighted contrast, with τ = `--tau`.

```bash
OUT=results/fewshot_16way/llama31_8b
python experiments/fewshot_16way.py --model_name meta-llama/Llama-3.1-8B-Instruct --output_dir $OUT
python experiments/headscore_concentration.py --threshold 0.95 \
  --query_metrics_path $OUT/head_contribution_share_query_metrics.jsonl \
  --classification_report_path $OUT/rkr_features_orderfix_online_16way_5shot_n3000_classification_report.json \
  --output_dir $OUT/concentration_095
python experiments/wikidata_connection_counts.py --samples_per_property 1000 --max_offset 50000 \
  --sleep_seconds 0.25 --output_dir results/wikidata_connection_counts   # CPU, public WDQS endpoint
```

- Table 2 / appendix: the classification report holds per-property precision, recall, F1 (`P<id>` keys); macro scores are their unweighted means.
- Fig. 3: `head_concentration_summary_threshold_95.json` has k95 per relation (`relation_mean_heads_needed`) and its Spearman correlation with precision, F1 and output range (`experiments/wikidata_ranges.json`); per-query k_X is in `threshold_counts["0.95"]` of the query metrics. Mean connection count: `properties.<P>.mean_connection_count` in `property_stats.json`.

## Sec. 4.3: Token-level attribution (Table 3, figures)

```bash
python experiments/lexical_cues.py --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/lexical_cues/llama31_8b
```

16-way 5-shot, q=15, 100 episodes, seed 42 (defaults). In `summary.json`: ρ = `all_queries.lexical_tokenscore_spearman.mean`, Mass = `all_queries.positive_tokenscore_mass_on_lexical_tokens.mean`, StrongAlign× = `incorrect_queries.strong_alignment_fraction.fraction` (ρ ≥ 0.30 and Mass ≥ 0.50). Per-query values: `query_metrics.jsonl`.

Interactive TokenScore vs GxA vs attention (aggregated / per layer). The apps sample episodes (default 5-way 5-shot, one query per relation) from examples where the subject precedes the object (two-blank prompt); n-way, k-shot, q and the seed are set in the UI:

```bash
python app_tokenscore.py --model_name meta-llama/Llama-3.2-1B-Instruct
python app_tokenscore_layerwise.py --model_name meta-llama/Llama-3.2-1B-Instruct
```
