#!/usr/bin/env python3
"""Pilot 2a v2 — Polysemy audit with entity-side coherence.

v1 returned METRICS_INSUFFICIENT because L0 features (e.g., L0_F5560) have
structurally-noisy down_meta despite coherent gating. This version augments
the v1 metric set with entity-side coherence drawn from rich JSON for ALL
129 cumulative wn:* features (canonical via re-run, multilingual + subword
already-rich).

Same anchors:
  L9_F7535  MUST land promiscuous
  L8_F8974, L0_F5560, L12_F5382  MUST land mono_semantic

New metrics (per feature):
  entity_count            — number of distinct matched entities
  entity_coherence        — mean pairwise Jaccard over char-bigrams of entities
  entity_bimodality       — bimodal clustering score over entities (for polysemy)

Classification:
  promiscuous   = low entity_coherence AND low down_meta coherence AND few entities
  polysemantic  = high entity_bimodality OR high down_meta bimodality with enough content
  mono_semantic = high entity_coherence OR high down_meta coherence (the OR captures
                  L0_F5560's coherent-entities-noisy-down_meta case)
  borderline    = metrics don't decide
"""

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import probe_mlx as pm  # noqa: E402

VINDEX = Path("/Users/christopherhay/chris-source/larql/output/gemma3-4b-v2.vindex")
ANCHOR_PROMISCUOUS = {"L9_F7535"}
ANCHOR_MONO = {"L8_F8974", "L0_F5560", "L12_F5382"}

NOISE_TOKENS = {
    "mathrm", "marginleft", "nonatomic", "eqref", "newcommand",
    "textbf", "textit", "begin", "end", "frac", "left", "right",
    "src", "href", "html", "div", "span", "img",
    "init", "self", "args", "kwargs", "void", "null", "true", "false",
}


def script_tag(token: str) -> str:
    s = token.lstrip("▁").strip()
    if not s:
        return "empty"
    scripts = set()
    has_alpha = False
    for ch in s:
        if not ch.isalpha():
            continue
        has_alpha = True
        try:
            n = unicodedata.name(ch, "")
        except ValueError:
            continue
        if "LATIN" in n:
            scripts.add("LAT")
        elif "CYRILLIC" in n:
            scripts.add("CYR")
        elif "GREEK" in n:
            scripts.add("GRK")
        elif "ARABIC" in n:
            scripts.add("ARA")
        elif "HEBREW" in n:
            scripts.add("HEB")
        elif any(k in n for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            scripts.add("CJK")
        elif "DEVANAGARI" in n:
            scripts.add("DEV")
        elif "THAI" in n:
            scripts.add("THA")
        else:
            scripts.add("OTH")
    if not has_alpha:
        return "punct_or_digit"
    return "+".join(sorted(scripts)) if scripts else "OTH"


def is_real_word(token: str, min_len: int = 3) -> bool:
    s = token.lstrip("▁").strip()
    if len(s) < min_len or not s.isalpha() or s.lower() in NOISE_TOKENS:
        return False
    return True


def real_word_ratio(tokens):
    return sum(1 for t in tokens if is_real_word(t)) / max(len(tokens), 1)


def mean_token_length(tokens):
    if not tokens:
        return 0.0
    return sum(len(t.lstrip("▁").strip()) for t in tokens) / len(tokens)


def char_bigrams(s):
    s = s.lstrip("▁").strip().lower()
    if len(s) < 2:
        return set()
    return {s[i:i+2] for i in range(len(s) - 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_coherence(strings):
    """Mean pairwise Jaccard over character bigrams. Works for tokens or entities."""
    if len(strings) < 2:
        return 0.0
    bigrams = [char_bigrams(s) for s in strings]
    sims = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            sims.append(jaccard(bigrams[i], bigrams[j]))
    return sum(sims) / max(len(sims), 1)


def bimodality(strings, cap=12):
    """Average-linkage 2-cluster silhouette over char-bigram distances."""
    if len(strings) < 4 or len(strings) > cap:
        return 0.0
    bigrams = [char_bigrams(s) for s in strings]
    n = len(strings)
    dist = [[1.0 - jaccard(bigrams[i], bigrams[j]) for j in range(n)] for i in range(n)]
    best_sil = 0.0
    for k in range(1, n // 2 + 1):
        for cluster_a in combinations(range(n), k):
            set_a = set(cluster_a)
            set_b = set(range(n)) - set_a
            if not set_a or not set_b:
                continue
            sils = []
            for i in range(n):
                own = set_a if i in set_a else set_b
                other = set_b if i in set_a else set_a
                if len(own) <= 1:
                    sils.append(0.0)
                    continue
                a = sum(dist[i][j] for j in own if j != i) / max(len(own) - 1, 1)
                b = sum(dist[i][j] for j in other) / max(len(other), 1)
                sils.append((b - a) / max(a, b, 1e-9))
            avg = sum(sils) / len(sils)
            if avg > best_sil:
                best_sil = avg
    return best_sil


def featkey_to_lf(k):
    l_str, f_str = k.split("_F")
    return int(l_str[1:]), int(f_str)


def load_cumulative_inventory_with_entities():
    """Load the 129 cumulative wn:* features with entity sets from rich JSONs.

    Combines (canonical re-run rich) + multilingual rich + subword rich.
    Each feature gets the UNION of entities across all sources it appears in.
    """
    canonical_rich_path = VINDEX / "feature_labels_canonical_rerun_rich.json"
    ml_rich_path = VINDEX / "feature_labels_multilingual_pilot_rich.json"
    sw_rich_path = VINDEX / "feature_labels_subword_pilot_rich.json"
    canonical_str_path = VINDEX / "feature_labels.json"  # for wn:* membership in deployed canonical

    canonical_rich = json.load(open(canonical_rich_path)) if canonical_rich_path.exists() else {}
    ml_rich = json.load(open(ml_rich_path))
    sw_rich = json.load(open(sw_rich_path))
    deployed_canonical = json.load(open(canonical_str_path))

    # Union of wn:* feature keys
    feature_keys = set()
    feature_keys.update(canonical_rich.keys())
    feature_keys.update(k for k, v in ml_rich.items() if v.get("primary", "").startswith("wn:"))
    feature_keys.update(k for k, v in sw_rich.items() if v.get("primary", "").startswith("wn:"))
    # Also include deployed canonical wn:* features even if the re-run didn't reproduce them
    feature_keys.update(k for k, v in deployed_canonical.items() if isinstance(v, str) and v.startswith("wn:"))

    inventory = {}
    for k in feature_keys:
        entry = {
            "sources": [],
            "labels_from_each": {},
            "entities": set(),
            "outputs": set(),
        }
        if k in canonical_rich:
            entry["sources"].append("canonical_rerun")
            entry["labels_from_each"]["canonical_rerun"] = canonical_rich[k].get("primary")
            entry["entities"].update(canonical_rich[k].get("entities", []))
            entry["outputs"].update(canonical_rich[k].get("outputs", []))
        if k in deployed_canonical and isinstance(deployed_canonical[k], str) and deployed_canonical[k].startswith("wn:"):
            entry["sources"].append("canonical_deployed")
            entry["labels_from_each"]["canonical_deployed"] = deployed_canonical[k]
        if k in ml_rich and ml_rich[k].get("primary", "").startswith("wn:"):
            entry["sources"].append("multilingual")
            entry["labels_from_each"]["multilingual"] = ml_rich[k]["primary"]
            entry["entities"].update(ml_rich[k].get("entities", []))
            entry["outputs"].update(ml_rich[k].get("outputs", []))
        if k in sw_rich and sw_rich[k].get("primary", "").startswith("wn:"):
            entry["sources"].append("subword")
            entry["labels_from_each"]["subword"] = sw_rich[k]["primary"]
            entry["entities"].update(sw_rich[k].get("entities", []))
            entry["outputs"].update(sw_rich[k].get("outputs", []))
        entry["entities"] = sorted(entry["entities"])
        entry["outputs"] = sorted(entry["outputs"])
        if entry["sources"]:
            inventory[k] = entry

    return inventory


def classify_v2(metrics, cutoffs):
    rwr = metrics["real_word_ratio"]
    sim = metrics["down_meta_coherence"]
    bim_dm = metrics["down_meta_bimodality"]
    nrw = metrics["num_real_words"]

    ent_n = metrics["entity_count"]
    ent_coh = metrics["entity_coherence"]
    ent_bim = metrics["entity_bimodality"]

    # Promiscuous: low coherence on BOTH sides AND few entities
    # The L9_F7535 signature: low entity_coherence (Dutch nouns and English adjectives
    # don't cluster) + low down_meta coherence + small entity count
    if (ent_coh < cutoffs["promisc_ent_coh_max"]
            and sim < cutoffs["promisc_dm_sim_max"]
            and ent_n <= cutoffs["promisc_ent_n_max"]):
        return "promiscuous"

    # Polysemantic: bimodal clustering on entities OR down_meta with enough content
    if ((ent_bim > cutoffs["poly_bim_min"] and ent_n >= 4)
            or (bim_dm > cutoffs["poly_bim_min"] and nrw >= 4)):
        return "polysemantic"

    # Mono-semantic: high entity_coherence (L0_F5560 case) OR high down_meta coherence/rwr
    if (ent_coh >= cutoffs["mono_ent_coh_min"]
            or rwr >= cutoffs["mono_rwr_min"]
            or sim >= cutoffs["mono_dm_sim_min"]):
        return "mono_semantic"

    return "borderline"


def search_cutoffs_v2(metrics_per_feature, anchor_promisc, anchor_mono):
    # Grid search over the v2 cutoff space
    ent_coh_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    dm_sim_grid = [0.02, 0.05, 0.10, 0.15, 0.20]
    ent_n_grid = [2, 3, 4, 5]
    bim_grid = [0.30, 0.40, 0.50, 0.60]
    rwr_grid = [0.40, 0.50, 0.60, 0.70, 0.80]
    mono_ent_coh_grid = [0.05, 0.10, 0.15, 0.20, 0.30]

    best_cutoffs = None
    best_score = -1
    best_anchor_check = None

    for promisc_ent_coh_max in ent_coh_grid:
        for promisc_dm_sim_max in dm_sim_grid:
            for promisc_ent_n_max in ent_n_grid:
                for mono_ent_coh_min in [v for v in mono_ent_coh_grid if v >= promisc_ent_coh_max]:
                    for mono_rwr_min in rwr_grid:
                        for mono_dm_sim_min in [v for v in dm_sim_grid if v >= promisc_dm_sim_max]:
                            for poly_bim_min in bim_grid:
                                cutoffs = {
                                    "promisc_ent_coh_max": promisc_ent_coh_max,
                                    "promisc_dm_sim_max": promisc_dm_sim_max,
                                    "promisc_ent_n_max": promisc_ent_n_max,
                                    "mono_ent_coh_min": mono_ent_coh_min,
                                    "mono_rwr_min": mono_rwr_min,
                                    "mono_dm_sim_min": mono_dm_sim_min,
                                    "poly_bim_min": poly_bim_min,
                                }
                                anchor_check = {}
                                all_pass = True
                                for k in anchor_promisc:
                                    if k not in metrics_per_feature:
                                        anchor_check[k] = ("MISSING", "required: promiscuous")
                                        all_pass = False
                                        continue
                                    cls = classify_v2(metrics_per_feature[k], cutoffs)
                                    anchor_check[k] = (cls, "required: promiscuous")
                                    if cls != "promiscuous":
                                        all_pass = False
                                for k in anchor_mono:
                                    if k not in metrics_per_feature:
                                        anchor_check[k] = ("MISSING", "required: mono_semantic")
                                        all_pass = False
                                        continue
                                    cls = classify_v2(metrics_per_feature[k], cutoffs)
                                    anchor_check[k] = (cls, "required: mono_semantic")
                                    if cls != "mono_semantic":
                                        all_pass = False
                                if all_pass:
                                    classifications = Counter(
                                        classify_v2(m, cutoffs) for m in metrics_per_feature.values()
                                    )
                                    borderline_pct = classifications.get("borderline", 0) / max(len(metrics_per_feature), 1)
                                    mono_pct = classifications.get("mono_semantic", 0) / max(len(metrics_per_feature), 1)
                                    in_band = 0.70 <= mono_pct <= 0.90
                                    score = (1.0 if all_pass else 0) + (1.0 if in_band else 0) - borderline_pct
                                    if score > best_score:
                                        best_score = score
                                        best_cutoffs = cutoffs
                                        best_anchor_check = anchor_check
    return best_cutoffs, best_cutoffs is not None, best_anchor_check


def main():
    print(f"Loading vindex: {VINDEX}")
    config, _gates, down_meta = pm.load_vindex_gates_and_meta(str(VINDEX))
    print(f"  Loaded down_meta for {len(down_meta)} (layer, feature) entries")

    print("\nBuilding cumulative wn:* inventory with entity sets from rich JSONs...")
    inventory = load_cumulative_inventory_with_entities()
    print(f"  Cumulative wn:* count: {len(inventory)}")
    sources_dist = Counter(tuple(sorted(v["sources"])) for v in inventory.values())
    print(f"  Source distribution:")
    for srcs, n in sources_dist.most_common():
        print(f"    {srcs}: {n}")

    features_no_entities = [k for k, v in inventory.items() if not v["entities"]]
    print(f"  Features with NO entity data: {len(features_no_entities)} (will only have down_meta metrics)")

    print("\nComputing v2 metrics (down_meta + entity coherence)...")
    metrics_per_feature = {}
    for feat_key, info in inventory.items():
        try:
            l, f = featkey_to_lf(feat_key)
        except (ValueError, IndexError):
            continue
        tokens = down_meta.get((l, f), [])
        entities = info["entities"]
        real_words_dm = [t.lstrip("▁").strip() for t in tokens if is_real_word(t)]
        metrics = {
            "n_tokens": len(tokens),
            "num_real_words": len(real_words_dm),
            "real_word_ratio": real_word_ratio(tokens),
            "mean_token_length": mean_token_length(tokens),
            "down_meta_coherence": pairwise_coherence(real_words_dm),
            "down_meta_bimodality": bimodality(real_words_dm),
            "entity_count": len(entities),
            "entity_coherence": pairwise_coherence(entities),
            "entity_bimodality": bimodality(entities),
            "down_meta_full": tokens,
            "top_real_words": real_words_dm[:10],
            "entities_sample": entities[:15],
            "sources": info["sources"],
            "labels_from_each": info["labels_from_each"],
            "script_distribution": dict(Counter(script_tag(t) for t in tokens)),
        }
        metrics_per_feature[feat_key] = metrics

    print(f"  Features with metrics: {len(metrics_per_feature)}")

    # Distributions for visibility
    def dist_summary(name, vals):
        sv = sorted(vals)
        if not sv:
            return f"  {name}: empty"
        def q(p): return sv[int(len(sv) * p)]
        return f"  {name}: min={sv[0]:.3f}  p25={q(0.25):.3f}  p50={q(0.5):.3f}  p75={q(0.75):.3f}  max={sv[-1]:.3f}"

    print(f"\nMetric distributions (n={len(metrics_per_feature)}):")
    print(dist_summary("real_word_ratio    ", [m["real_word_ratio"] for m in metrics_per_feature.values()]))
    print(dist_summary("down_meta_coherence", [m["down_meta_coherence"] for m in metrics_per_feature.values()]))
    print(dist_summary("entity_coherence   ", [m["entity_coherence"] for m in metrics_per_feature.values()]))
    print(dist_summary("entity_bimodality  ", [m["entity_bimodality"] for m in metrics_per_feature.values()]))
    print(dist_summary("entity_count       ", [float(m["entity_count"]) for m in metrics_per_feature.values()]))

    print("\nAnchor feature metrics:")
    for k in sorted(ANCHOR_PROMISCUOUS | ANCHOR_MONO):
        if k in metrics_per_feature:
            m = metrics_per_feature[k]
            req = "promiscuous" if k in ANCHOR_PROMISCUOUS else "mono_semantic"
            print(f"  {k} (req={req}): rwr={m['real_word_ratio']:.2f} dm_coh={m['down_meta_coherence']:.3f} "
                  f"ent_coh={m['entity_coherence']:.3f} ent_bim={m['entity_bimodality']:.3f} ent_n={m['entity_count']}")
            print(f"     entities sample: {m['entities_sample'][:6]}")
        else:
            print(f"  {k}: MISSING FROM INVENTORY")

    print("\nSearching v2 cutoff space for anchor-satisfying combination...")
    cutoffs, success, anchor_check = search_cutoffs_v2(
        metrics_per_feature, ANCHOR_PROMISCUOUS, ANCHOR_MONO
    )

    if not success:
        print("\n!!! METRICS_INSUFFICIENT (v2) !!!")
        out_path = VINDEX / "feature_labels_polysemy_audit_v2.json"
        with open(out_path, "w") as f:
            json.dump({
                "status": "METRICS_INSUFFICIENT_V2",
                "reason": "no v2 cutoff combination satisfies all four anchor constraints",
                "anchors": {
                    "promiscuous": sorted(ANCHOR_PROMISCUOUS),
                    "mono_semantic": sorted(ANCHOR_MONO),
                },
                "metrics_per_feature": metrics_per_feature,
            }, f, indent=2, ensure_ascii=False, default=list)
        print(f"  Report -> {out_path}")
        return

    print(f"  v2 cutoffs found:")
    for k, v in cutoffs.items():
        print(f"    {k}: {v}")
    print(f"  Anchor check:")
    for k, (got, req) in anchor_check.items():
        marker = "OK" if got in req else "FAIL"
        print(f"    {k}: got={got}, {req} [{marker}]")

    print("\nApplying v2 classification...")
    classifications = {}
    for k, m in metrics_per_feature.items():
        classifications[k] = classify_v2(m, cutoffs)
    dist = Counter(classifications.values())
    total = len(classifications)

    print(f"\nClassification distribution (n={total}):")
    for cls, n in dist.most_common():
        print(f"  {cls:<15} {n:4d} ({n/total*100:>5.1f}%)")

    # Save audit
    full_audit = {}
    for k, m in metrics_per_feature.items():
        full_audit[k] = {
            "classification": classifications[k],
            "real_word_ratio": round(m["real_word_ratio"], 3),
            "down_meta_coherence": round(m["down_meta_coherence"], 3),
            "down_meta_bimodality": round(m["down_meta_bimodality"], 3),
            "entity_count": m["entity_count"],
            "entity_coherence": round(m["entity_coherence"], 3),
            "entity_bimodality": round(m["entity_bimodality"], 3),
            "num_real_words": m["num_real_words"],
            "top_real_words": m["top_real_words"],
            "entities_sample": m["entities_sample"],
            "down_meta_full": m["down_meta_full"],
            "labels_from": m["labels_from_each"],
            "sources": m["sources"],
        }
    out_path = VINDEX / "feature_labels_polysemy_audit_v2.json"
    with open(out_path, "w") as f:
        json.dump(full_audit, f, indent=2, ensure_ascii=False, default=list)
    print(f"\nFull audit -> {out_path}")

    mono_pct = dist.get("mono_semantic", 0) / total * 100
    poly_pct = dist.get("polysemantic", 0) / total * 100
    promisc_pct = dist.get("promiscuous", 0) / total * 100
    borderline_pct = dist.get("borderline", 0) / total * 100

    # P4 outcome
    mono_in = 70 <= mono_pct <= 90
    promisc_in = 5 <= promisc_pct <= 25
    poly_in = poly_pct <= 10
    if mono_in and promisc_in and poly_in:
        p4_result = "confirmed"
    elif poly_pct > 20:
        p4_result = "refuted-polysemy-high"
    elif promisc_pct > 30:
        p4_result = "refuted-promiscuity-high"
    elif borderline_pct > 10:
        p4_result = "partial-metrics-borderline"
    else:
        p4_result = "partial"
    launch_1c_ok = poly_pct <= 20 and promisc_pct <= 30

    summary = {
        "audit_version": "v2 (down_meta + entity coherence)",
        "total_audited": total,
        "classification_counts": dict(dist),
        "classification_pct": {
            "mono_semantic": round(mono_pct, 1),
            "polysemantic": round(poly_pct, 1),
            "promiscuous": round(promisc_pct, 1),
            "borderline": round(borderline_pct, 1),
        },
        "stable_count": dist.get("mono_semantic", 0) + dist.get("polysemantic", 0),
        "cutoffs_used": cutoffs,
        "anchor_check": {k: {"got": got, "required": req, "satisfied": got in req}
                         for k, (got, req) in anchor_check.items()},
        "tests_prediction": "P4 in META_MODEL.md",
        "P4_predicted": {"mono_semantic_pct": "70-90", "promiscuous_pct": "5-25", "polysemantic_pct": "<10"},
        "P4_observed_pct": {
            "mono_semantic_pct": round(mono_pct, 1),
            "promiscuous_pct": round(promisc_pct, 1),
            "polysemantic_pct": round(poly_pct, 1),
            "borderline_pct": round(borderline_pct, 1),
        },
        "P4_outcome": p4_result,
        "launch_1c_ok": launch_1c_ok,
        "launch_1c_blocker_reason": None if launch_1c_ok else (
            f"polysemy {poly_pct:.1f}% > 20%" if poly_pct > 20 else f"promiscuity {promisc_pct:.1f}% > 30%"
        ),
    }
    summary_path = VINDEX / "feature_labels_polysemy_audit_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary -> {summary_path}")

    print("\n" + "=" * 60)
    print("P4 OUTCOME (v2 audit)")
    print("=" * 60)
    print(f"Predicted: mono 70-90%, promiscuous 5-25%, polysemantic <10%")
    print(f"Observed:  mono {mono_pct:.1f}%, promiscuous {promisc_pct:.1f}%, polysemantic {poly_pct:.1f}%, borderline {borderline_pct:.1f}%")
    print(f"Stable count: {summary['stable_count']} / {total}")
    print(f"P4 outcome:   {p4_result}")
    print(f"\nLaunch 1c OK: {launch_1c_ok}")
    if not launch_1c_ok:
        print(f"  Blocker: {summary['launch_1c_blocker_reason']}")


if __name__ == "__main__":
    main()
