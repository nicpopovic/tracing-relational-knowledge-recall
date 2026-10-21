#!/usr/bin/env python3
"""
Mean Wikidata connection count per relation (Sec. 4.2, Fig. 3): sample
subject-object pairs (A, P, B) for each property P and count the direct truthy
Wikidata properties connecting A and B in either direction.

Sampling is approximate: seeded pseudo-random LIMIT/OFFSET batches over
(?a wdt:P ?b) with both endpoints Wikidata entities, keeping unique pairs. It
can be biased by the reachable offset range and WDQS result ordering.

Outputs (in --output_dir): samples.jsonl, property_stats.json, metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# The 16 FewRel validation relations evaluated in the paper (Sec. 4.2).
PROPERTIES = {
    "P59": "constellation",
    "P177": "crosses",
    "P2094": "competition class",
    "P364": "original language of film or TV show",
    "P412": "voice type",
    "P413": "position played on team / speciality",
    "P206": "located in or next to body of water",
    "P410": "military, police or special rank",
    "P641": "sport",
    "P155": "follows",
    "P921": "main subject",
    "P463": "member of",
    "P25": "mother",
    "P40": "child",
    "P26": "spouse",
    "P361": "part of",
}

ENTITY_PREFIX = "http://www.wikidata.org/entity/"
DIRECT_PREFIX = "http://www.wikidata.org/prop/direct/"
WDQS_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "WDPairConnectionCounts/1.0 (tracing-relational-knowledge-recall research code)"


@dataclass
class SampledPair:
    subject_id: str
    object_id: str


@dataclass
class PairSample:
    property_id: str
    property_label: str
    subject_id: str
    subject_label: str
    object_id: str
    object_label: str
    forward_properties: list[str]
    reverse_properties: list[str]
    connection_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "property_label": self.property_label,
            "subject_id": self.subject_id,
            "subject_label": self.subject_label,
            "object_id": self.object_id,
            "object_label": self.object_label,
            "forward_properties": self.forward_properties,
            "reverse_properties": self.reverse_properties,
            "all_properties": sorted(set(self.forward_properties + self.reverse_properties)),
            "connection_count": self.connection_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--properties", type=str, default=",".join(PROPERTIES.keys()))
    parser.add_argument("--samples_per_property", type=int, default=10)
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=3,
        help="Controls LIMIT size relative to requested samples.",
    )
    parser.add_argument(
        "--max_offset",
        type=int,
        default=3000,
        help="Maximum OFFSET explored by the approximate sampler.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--sleep_seconds", type=float, default=0.5)
    parser.add_argument("--retry_base_seconds", type=float, default=4.0)
    parser.add_argument(
        "--count_batch_size",
        type=int,
        default=25,
        help="How many sampled pairs to count in one grouped WDQS query.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def property_to_sampling_pattern(prop_id: str) -> str:
    return f"?a wdt:{prop_id} ?b"


def run_sparql_json(
    sparql: str,
    *,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query": sparql, "format": "json"})
    req = urllib.request.Request(
        f"{WDQS_URL}?{params}",
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            retry_after_raw = exc.headers.get("Retry-After")
            retry_after = float(retry_after_raw) if retry_after_raw else None
        except Exception as exc:  # pragma: no cover
            last_error = exc
            retry_after = None

        if attempt == max_retries - 1:
            break
        wait_seconds = retry_after if retry_after is not None else retry_base_seconds * (2 ** attempt)
        time.sleep(wait_seconds)
    raise RuntimeError(f"WDQS query failed after {max_retries} attempts: {last_error}")


def extract_entity_id(uri: str) -> str:
    if not uri.startswith(ENTITY_PREFIX):
        raise ValueError(f"Expected entity URI, got {uri}")
    return uri.rsplit("/", 1)[-1]


def extract_label(binding: dict[str, Any], key: str) -> str:
    return binding.get(key, {}).get("value", "")


def sample_property_pairs(
    prop_id: str,
    *,
    samples_per_property: int,
    candidate_multiplier: int,
    max_offset: int,
    seed: int,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> list[SampledPair]:
    pattern = property_to_sampling_pattern(prop_id)
    batch_limit = max(10, candidate_multiplier * samples_per_property)
    offset_step = batch_limit
    offset_candidates = list(range(0, max_offset + 1, offset_step))
    random.Random(f"{prop_id}:{seed}").shuffle(offset_candidates)

    results: list[SampledPair] = []
    seen: set[tuple[str, str]] = set()
    failed_batches = 0

    total_offsets = len(offset_candidates)
    for offset_index, offset in enumerate(offset_candidates, start=1):
        if offset_index == 1 or offset_index % 10 == 0:
            print(
                f"  sampling progress: property={prop_id} offset_batch={offset_index}/{total_offsets} "
                f"collected={len(results)}/{samples_per_property}",
                flush=True,
            )
        sparql = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT ?a ?b WHERE {{
          {pattern} .
          FILTER(STRSTARTS(STR(?a), "{ENTITY_PREFIX}"))
          FILTER(STRSTARTS(STR(?b), "{ENTITY_PREFIX}"))
        }}
        LIMIT {batch_limit}
        OFFSET {offset}
        """
        try:
            data = run_sparql_json(
                sparql,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
            )
        except Exception as exc:
            failed_batches += 1
            print(f"  WARNING: sampling batch failed at OFFSET {offset}: {exc}", flush=True)
            if failed_batches >= 3 and not results:
                raise
            continue

        for row in data["results"]["bindings"]:
            subject_id = extract_entity_id(row["a"]["value"])
            object_id = extract_entity_id(row["b"]["value"])
            key = (subject_id, object_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(SampledPair(subject_id=subject_id, object_id=object_id))
            if len(results) >= samples_per_property:
                return results
        time.sleep(0.1)

    return results


def count_pair_connections_batch(
    pairs: list[SampledPair],
    *,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> list[dict[str, list[str]]]:
    values_rows = " ".join(
        f'("{index}" wd:{pair.subject_id} wd:{pair.object_id})'
        for index, pair in enumerate(pairs)
    )
    sparql = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT DISTINCT ?pair_idx ?dir ?prop WHERE {{
      {{
        VALUES (?pair_idx ?a ?b) {{ {values_rows} }}
        ?a ?p ?b .
        FILTER(STRSTARTS(STR(?p), "{DIRECT_PREFIX}"))
        BIND("forward" AS ?dir)
        BIND(REPLACE(STR(?p), "^{DIRECT_PREFIX}", "") AS ?prop)
      }}
      UNION
      {{
        VALUES (?pair_idx ?a ?b) {{ {values_rows} }}
        ?b ?p ?a .
        FILTER(STRSTARTS(STR(?p), "{DIRECT_PREFIX}"))
        BIND("reverse" AS ?dir)
        BIND(REPLACE(STR(?p), "^{DIRECT_PREFIX}", "") AS ?prop)
      }}
    }}
    ORDER BY ?pair_idx ?dir ?prop
    """
    data = run_sparql_json(
        sparql,
        timeout=timeout,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
    )
    connections = [{"forward_properties": [], "reverse_properties": []} for _ in pairs]
    for row in data["results"]["bindings"]:
        pair_idx = int(row["pair_idx"]["value"])
        prop_id = row["prop"]["value"]
        if row["dir"]["value"] == "forward":
            connections[pair_idx]["forward_properties"].append(prop_id)
        else:
            connections[pair_idx]["reverse_properties"].append(prop_id)
    return connections


def fetch_entity_labels(
    entity_ids: list[str],
    *,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for start in range(0, len(entity_ids), 100):
        batch = entity_ids[start : start + 100]
        values = " ".join(f"wd:{entity_id}" for entity_id in batch)
        sparql = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        SELECT ?entity ?entityLabel WHERE {{
          VALUES ?entity {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        try:
            data = run_sparql_json(
                sparql,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
            )
        except Exception as exc:
            print(f"WARNING: label lookup failed for batch {start // 100 + 1}: {exc}", flush=True)
            continue
        for row in data["results"]["bindings"]:
            entity_id = extract_entity_id(row["entity"]["value"])
            labels[entity_id] = extract_label(row, "entityLabel") or entity_id
        time.sleep(0.1)
    return labels


def summarize_property(samples: list[PairSample]) -> dict[str, Any]:
    counts = [sample.connection_count for sample in samples]
    if not counts:
        return {
            "n_samples": 0,
            "mean_connection_count": None,
            "std_connection_count": None,
            "min_connection_count": None,
            "median_connection_count": None,
            "max_connection_count": None,
        }
    return {
        "n_samples": len(samples),
        "mean_connection_count": statistics.fmean(counts),
        "std_connection_count": statistics.stdev(counts) if len(counts) >= 2 else 0.0,
        "min_connection_count": min(counts),
        "median_connection_count": statistics.median(counts),
        "max_connection_count": max(counts),
    }


def write_outputs(
    output_dir: Path,
    all_samples: list[PairSample],
    property_stats: dict[str, Any],
    failures: dict[str, str],
) -> None:
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=True) + "\n")

    payload = {
        "properties": property_stats,
        "failures": failures,
    }
    (output_dir / "property_stats.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_properties = [part.strip() for part in args.properties.split(",") if part.strip()]
    unknown = [prop_id for prop_id in requested_properties if prop_id not in PROPERTIES]
    if unknown:
        raise ValueError(f"Unknown properties: {', '.join(unknown)}")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "properties": requested_properties,
        "samples_per_property": args.samples_per_property,
        "candidate_multiplier": args.candidate_multiplier,
        "max_offset": args.max_offset,
        "seed": args.seed,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "sleep_seconds": args.sleep_seconds,
        "retry_base_seconds": args.retry_base_seconds,
        "count_batch_size": args.count_batch_size,
        "sampling_approximation": (
            "Seeded pseudo-random LIMIT/OFFSET sampling over a restricted result "
            "slice. This is only approximately uniform and may be biased by the "
            "reachable offset range and WDQS result ordering."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    all_samples: list[PairSample] = []
    property_stats: dict[str, Any] = {}
    failures: dict[str, str] = {}

    for prop_id in requested_properties:
        prop_label = PROPERTIES[prop_id]
        prop_position = requested_properties.index(prop_id) + 1
        print(
            f"[{prop_position}/{len(requested_properties)}] Sampling {prop_id} ({prop_label})...",
            flush=True,
        )
        try:
            sampled_pairs = sample_property_pairs(
                prop_id,
                samples_per_property=args.samples_per_property,
                candidate_multiplier=args.candidate_multiplier,
                max_offset=args.max_offset,
                seed=args.seed,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
            )
        except Exception as exc:
            failures[prop_id] = f"sampling_failed: {exc}"
            property_stats[prop_id] = {
                "property_label": prop_label,
                **summarize_property([]),
            }
            write_outputs(output_dir, all_samples, property_stats, failures)
            print(f"  FAILED during sampling: {exc}", flush=True)
            continue

        if len(sampled_pairs) < args.samples_per_property:
            print(
                f"  WARNING: only found {len(sampled_pairs)} unique pairs for requested "
                f"N={args.samples_per_property}",
                flush=True,
            )

        prop_samples: list[PairSample] = []
        try:
            for batch_start in range(0, len(sampled_pairs), args.count_batch_size):
                pair_batch = sampled_pairs[batch_start : batch_start + args.count_batch_size]
                batch_end = batch_start + len(pair_batch)
                print(
                    f"  counting progress: property={prop_id} pairs={batch_start + 1}-{batch_end}/{len(sampled_pairs)}",
                    flush=True,
                )
                connection_infos = count_pair_connections_batch(
                    pair_batch,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                )
                for inner_index, (pair, connection_info) in enumerate(zip(pair_batch, connection_infos), start=1):
                    connection_count = (
                        len(connection_info["forward_properties"])
                        + len(connection_info["reverse_properties"])
                    )
                    sample = PairSample(
                        property_id=prop_id,
                        property_label=prop_label,
                        subject_id=pair.subject_id,
                        subject_label=pair.subject_id,
                        object_id=pair.object_id,
                        object_label=pair.object_id,
                        forward_properties=connection_info["forward_properties"],
                        reverse_properties=connection_info["reverse_properties"],
                        connection_count=connection_count,
                    )
                    prop_samples.append(sample)
                    all_samples.append(sample)
                    print(
                        f"  [{batch_start + inner_index}/{len(sampled_pairs)}] "
                        f"{pair.subject_id} -> {pair.object_id}: "
                        f"{connection_count} direct truthy connections",
                        flush=True,
                    )
                write_outputs(output_dir, all_samples, property_stats, failures)
                time.sleep(args.sleep_seconds)
        except Exception as exc:
            failures[prop_id] = f"count_failed: {exc}"
            print(f"  FAILED during counting: {exc}", flush=True)

        property_stats[prop_id] = {
            "property_label": prop_label,
            **summarize_property(prop_samples),
        }
        write_outputs(output_dir, all_samples, property_stats, failures)

    entity_ids = sorted({sample.subject_id for sample in all_samples} | {sample.object_id for sample in all_samples})
    label_map = fetch_entity_labels(
        entity_ids,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    for sample in all_samples:
        sample.subject_label = label_map.get(sample.subject_id, sample.subject_id)
        sample.object_label = label_map.get(sample.object_id, sample.object_id)
    write_outputs(output_dir, all_samples, property_stats, failures)

    print(f"Wrote metadata to {output_dir / 'metadata.json'}", flush=True)
    print(f"Wrote per-sample results to {output_dir / 'samples.jsonl'}", flush=True)
    print(f"Wrote aggregate stats to {output_dir / 'property_stats.json'}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
