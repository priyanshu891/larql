#!/usr/bin/env python3
"""Re-run the canonical English WordNet probe with rich JSON output.

Why: per M2 in /Users/christopherhay/chris-source/larql/META_MODEL.md, canonical
feature_labels.json was generated under an earlier methodology iteration and
only string-form labels survived to deployment. The polysemy audit (Pilot 2a
v1) returned METRICS_INSUFFICIENT because L0 features have structurally-noisy
down_meta despite coherent gating — and we cannot use entity-side coherence
to discriminate them without the canonical-side entity sets.

This script regenerates rich JSON for the canonical 5 wn:* relations using
the SAME methodology as the multilingual + subword pilots, so the audit can
combine down_meta inspection with entity-coherence inspection on a uniform
substrate.

Output:
  feature_labels_canonical_rerun.json       — comparability labels (>=2 hits, conf>0.5)
  feature_labels_canonical_rerun_rich.json  — entities, outputs, relations, layer per feature
  feature_labels_canonical_rerun_drift.json — drift vs deployed feature_labels.json (M2 evidence)

This is NOT a re-merge into canonical. Canonical stays deployed as-is. This
artifact is parallel — it's the entity-rich version of canonical needed by
the audit and the M2 drift assessment.
"""

import argparse
import json
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import probe_mlx as pm  # noqa: E402

_KNOWLEDGE_DIR = _SCRIPT_DIR.parent
_DEFAULT_CANONICAL = _KNOWLEDGE_DIR / "data" / "wordnet_relations.json"


def load_canonical(path: Path) -> dict:
    """Load canonical wordnet_relations.json, keep only the 5 wn:* relations,
    wrap with wn: prefix."""
    with open(path) as f:
        raw = json.load(f)
    wn_relations = ["synonym", "hypernym", "antonym", "meronym", "derivation"]
    return {f"wn:{rel}": raw[rel] for rel in wn_relations if rel in raw}


def parse_args():
    p = argparse.ArgumentParser(description="Canonical wn:* probe with rich output")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--vindex", default=None)
    p.add_argument("--canonical", type=str, default=str(_DEFAULT_CANONICAL))
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-gate-score", type=float, default=5.0)
    p.add_argument("--offline", action="store_true", default=True)
    p.add_argument("--limit-subjects", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_id = args.model
    model_slug = pm._model_slug(model_id)

    vindex_path = args.vindex
    if vindex_path is None:
        output_root = _KNOWLEDGE_DIR.parent / "output"
        canonical_vindex = output_root / "gemma3-4b-v2.vindex"
        slug_default = output_root / f"{model_slug}.vindex"
        if canonical_vindex.exists():
            vindex_path = str(canonical_vindex)
        elif slug_default.exists():
            vindex_path = str(slug_default)
    if not vindex_path or not Path(vindex_path).exists():
        print(f"ERROR: vindex not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading vindex: {vindex_path}")
    config, gates, down_meta = pm.load_vindex_gates_and_meta(vindex_path)
    num_layers = config["num_layers"]
    print(f"  {num_layers} layers, {config['hidden_size']} hidden, {len(down_meta)} features")

    if "layer_bands" in config and config["layer_bands"]:
        bands = config["layer_bands"]
        syntax_end = bands.get("knowledge_start", num_layers * 2 // 5)
    else:
        syntax_end = num_layers * 2 // 5
    scan_layers = list(range(0, syntax_end))
    print(f"  Scanning syntax layers: L0-L{syntax_end - 1} ({len(scan_layers)} layers)")

    canonical_path = Path(args.canonical)
    if not canonical_path.exists():
        print(f"ERROR: canonical data not found at {canonical_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading canonical wn:* data: {canonical_path}")
    syntax_data = load_canonical(canonical_path)
    total_pairs = sum(len(d.get("pairs", [])) for d in syntax_data.values())
    print(f"  {len(syntax_data)} relations, {total_pairs} pairs")

    syntax_index = pm.build_match_index(syntax_data)
    print(f"  Match index: {len(syntax_index)} entries")

    TEMPLATES = {rel: ["{X}"] for rel in syntax_data}

    print(f"Loading MLX model: {model_id}...")
    import os
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_id)
    print("  Model loaded")

    start_time = time.time()

    print("Phase 1: collecting unique subjects...")
    rel_to_subjects = {}
    unique_subjects_set = set()
    for rel_name in TEMPLATES:
        if rel_name not in syntax_data:
            continue
        rel_subjs = list({
            pair[0] for pair in syntax_data[rel_name].get("pairs", [])
            if len(pair) >= 2 and 2 <= len(pair[0]) <= 30
        })
        if args.limit_subjects:
            rel_subjs = rel_subjs[: args.limit_subjects]
        rel_to_subjects[rel_name] = rel_subjs
        unique_subjects_set.update(rel_subjs)

    unique_subjects = sorted(unique_subjects_set, key=lambda s: (len(s.split()), len(s)))
    pair_level_probes = sum(len(s) for s in rel_to_subjects.values())
    print(f"  {len(unique_subjects)} unique subjects across {len(rel_to_subjects)} relations"
          f" (pair-level: {pair_level_probes} — cache saves {pair_level_probes - len(unique_subjects)} forward passes)")

    print("Phase 2: encoding...")
    residual_cache = {}
    encode_start = time.time()
    for i, subj in enumerate(unique_subjects):
        residuals, _ = pm.get_residuals_and_logits(model, tokenizer, subj)
        if residuals is None:
            continue
        residual_cache[subj] = {l: residuals[l] for l in scan_layers if l in residuals}
        if (i + 1) % 50 == 0 or (i + 1) == len(unique_subjects):
            el = time.time() - encode_start
            rate = (i + 1) / max(el, 0.1)
            eta = max(0, (len(unique_subjects) - i - 1) / max(rate, 0.1))
            sys.stdout.write(
                f"\r  encoded {i+1}/{len(unique_subjects)} ({rate:.1f}/s, ETA {eta:.0f}s)  "
            )
            sys.stdout.flush()
    print(f"\n  Encoded {len(residual_cache)} subjects in {time.time() - encode_start:.0f}s")

    print("Phase 3: gate matching...")
    feature_hits = defaultdict(lambda: defaultdict(int))
    feature_entities = defaultdict(lambda: defaultdict(set))
    feature_outputs = defaultdict(lambda: defaultdict(set))
    feature_first_layer = {}
    match_attempts = 0

    for rel_name, rel_subjs in rel_to_subjects.items():
        gate_matched = 0
        for subject in rel_subjs:
            if subject not in residual_cache:
                continue
            residuals = residual_cache[subject]
            match_attempts += 1
            subj_key = subject.lower().strip()

            for layer in scan_layers:
                if layer not in residuals or layer not in gates:
                    continue
                r = residuals[layer]
                scores = gates[layer] @ r
                top_indices = np.argsort(-np.abs(scores))[:args.top_k]

                for feat_idx in top_indices:
                    score = float(scores[feat_idx])
                    if abs(score) < args.min_gate_score:
                        continue
                    tokens = down_meta.get((layer, int(feat_idx)), [])
                    if not tokens:
                        continue

                    feat_key = f"L{layer}_F{feat_idx}"
                    for target in tokens:
                        if len(target) < 2:
                            continue
                        tgt_lower = target.lower().strip()
                        if syntax_index.get((subj_key, tgt_lower)) == rel_name:
                            feature_hits[feat_key][rel_name] += 1
                            feature_entities[feat_key][rel_name].add(subject)
                            feature_outputs[feat_key][rel_name].add(tgt_lower)
                            if feat_key not in feature_first_layer:
                                feature_first_layer[feat_key] = layer
                            gate_matched += 1
                            break

        print(f"  {rel_name:<25s} {len(rel_subjs):5d} subjects -> {gate_matched:5d} hits")

    elapsed = time.time() - start_time
    inference_count = len(residual_cache)
    print(f"\nTotal: {inference_count} forward passes (vs {pair_level_probes} naive),"
          f" {match_attempts} match attempts in {elapsed:.0f}s")
    print(f"Features with hits: {len(feature_hits)}")

    # Apply same threshold as ml + sw pilots for comparability
    rerun_labels = {}
    label_details = {}
    relation_totals = Counter()
    for feat_key, rel_counts in feature_hits.items():
        total_hits = sum(rel_counts.values())
        primary_rel = max(rel_counts, key=rel_counts.get)
        primary_count = rel_counts[primary_rel]
        confidence = primary_count / total_hits

        if primary_count >= 2 and confidence > 0.5:
            rerun_labels[feat_key] = primary_rel
            relation_totals[primary_rel] += 1
            entities = sorted(feature_entities[feat_key].get(primary_rel, set()))
            outputs = sorted(feature_outputs[feat_key].get(primary_rel, set()))
            label_details[feat_key] = {
                "primary": primary_rel,
                "confidence": round(confidence, 3),
                "hits": total_hits,
                "entity_count": len(entities),
                "entities": entities[:30],
                "outputs": outputs[:10],
                "relations": {r: c for r, c in sorted(rel_counts.items(), key=lambda x: -x[1])},
                "first_layer": feature_first_layer.get(feat_key),
            }
    print(f"Labeled (>=2 hits, conf>0.5): {len(rerun_labels)} features"
          f" ({len(feature_hits) - len(rerun_labels)} dropped)")

    if relation_totals:
        print(f"\nRelation distribution ({len(relation_totals)} relations):")
        for rel, count in relation_totals.most_common():
            print(f"  {rel:<25s} {count:4d}")

    rerun_path = Path(vindex_path) / "feature_labels_canonical_rerun.json"
    with open(rerun_path, "w") as f:
        json.dump(rerun_labels, f, indent=2, ensure_ascii=False)
    print(f"\nCanonical re-run labels -> {rerun_path}")

    rich_path = Path(vindex_path) / "feature_labels_canonical_rerun_rich.json"
    with open(rich_path, "w") as f:
        json.dump(label_details, f, indent=2, ensure_ascii=False)
    print(f"Canonical re-run rich -> {rich_path}")

    # M2 drift assessment vs deployed canonical
    deployed = json.load(open(Path(vindex_path) / "feature_labels.json"))
    deployed_wn = {k: v for k, v in deployed.items() if isinstance(v, str) and v.startswith("wn:")}
    rerun_wn = set(rerun_labels.keys())
    deployed_wn_keys = set(deployed_wn.keys())

    only_in_deployed = deployed_wn_keys - rerun_wn
    only_in_rerun = rerun_wn - deployed_wn_keys
    in_both = deployed_wn_keys & rerun_wn
    label_drift = []
    label_agree = []
    for k in sorted(in_both):
        if deployed_wn[k] != rerun_labels[k]:
            label_drift.append({"feature": k, "deployed": deployed_wn[k], "rerun": rerun_labels[k]})
        else:
            label_agree.append(k)

    drift_path = Path(vindex_path) / "feature_labels_canonical_rerun_drift.json"
    with open(drift_path, "w") as f:
        json.dump({
            "deployed_wn_count": len(deployed_wn),
            "rerun_wn_count": len(rerun_wn),
            "in_both": len(in_both),
            "only_in_deployed": sorted(only_in_deployed),
            "only_in_rerun": sorted(only_in_rerun)[:50],
            "label_drift": label_drift,
            "label_agree_count": len(label_agree),
            "drift_rate_among_overlap": round(len(label_drift) / max(len(in_both), 1), 3),
        }, f, indent=2, ensure_ascii=False)
    print(f"M2 drift assessment -> {drift_path}")

    print("\n" + "=" * 60)
    print("CANONICAL RE-RUN SUMMARY + M2 DRIFT")
    print("=" * 60)
    print(f"Deployed canonical wn:* count:  {len(deployed_wn)}")
    print(f"Re-run wn:* count:              {len(rerun_wn)}")
    print(f"  Overlap:                      {len(in_both)}")
    print(f"  Only in deployed:             {len(only_in_deployed)} (features re-run didn't reproduce)")
    print(f"  Only in re-run:               {len(only_in_rerun)} (features re-run found that deployed missed)")
    print(f"  Label drift among overlap:    {len(label_drift)}/{len(in_both)} ({len(label_drift)/max(len(in_both),1)*100:.1f}%)")

    if label_drift[:5]:
        print(f"\nSample drift cases (first 5):")
        for d in label_drift[:5]:
            print(f"  {d['feature']}: deployed={d['deployed']} → rerun={d['rerun']}")


if __name__ == "__main__":
    main()
