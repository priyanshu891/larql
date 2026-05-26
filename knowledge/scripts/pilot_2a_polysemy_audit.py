#!/usr/bin/env python3
"""Pilot 2a — Polysemy audit. Static down_meta inspection of the cumulative
wn:* labeled inventory at L0-L12. Three-way classification:
mono_semantic / polysemantic / promiscuous (+ borderline escape valve).

Tests P4 in /Users/christopherhay/chris-source/larql/META_MODEL.md.

No model inference. Reads only the vindex down_meta and the four label files.

Anchor constraint (load-bearing):
  L9_F7535  MUST land promiscuous (Dutch nouns + English adjectives, same
                                   wn:synonym, no content coherence)
  L8_F8974  MUST land mono_semantic (cross-lingual EXCEPTION cluster)
  L0_F5560  MUST land mono_semantic (biological-taxa → class)
  L12_F5382 MUST land mono_semantic (poor-synonyms cluster)

Cutoff procedure: compute metrics on all 129 features, search the cutoff
space for combinations that satisfy the anchor constraint. If no combination
satisfies all four anchors: report metrics-insufficient — that is a result.
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

_KNOWLEDGE_DIR = _SCRIPT_DIR.parent

# Anchors per spec
ANCHOR_PROMISCUOUS = {"L9_F7535"}
ANCHOR_MONO = {"L8_F8974", "L0_F5560", "L12_F5382"}

# Loose programming/markup blacklist for the "real word" check
NOISE_TOKENS = {
    "mathrm", "marginleft", "nonatomic", "eqref", "newcommand",
    "textbf", "textit", "begin", "end", "frac", "left", "right",
    "alpha", "beta", "gamma",  # only as standalone latex names
    "src", "href", "html", "div", "span", "img",
    "init", "self", "args", "kwargs", "void", "null", "true", "false",
}


def script_tag(token: str) -> str:
    """Coarse script classification."""
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
    """Token counts as a 'real word' if alpha, >=min_len, not in noise blacklist."""
    s = token.lstrip("▁").strip()
    if len(s) < min_len:
        return False
    if not s.isalpha():
        return False
    if s.lower() in NOISE_TOKENS:
        return False
    return True


def real_word_ratio(tokens):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if is_real_word(t)) / len(tokens)


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


def real_word_coherence(tokens):
    """Mean pairwise Jaccard over character bigrams for the real-word subset.
    High = morphologically/semantically related cluster.
    Low + low real-word ratio = promiscuous noise.
    Low + high real-word ratio = either polysemantic or genuinely diverse mono-semantic.
    """
    real = [t for t in tokens if is_real_word(t)]
    if len(real) < 2:
        return 0.0
    bigrams = [char_bigrams(t) for t in real]
    sims = []
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            sims.append(jaccard(bigrams[i], bigrams[j]))
    return sum(sims) / len(sims) if sims else 0.0


def bimodality(tokens):
    """Try to split real words into 2 clusters; return silhouette-like score.
    >0.3 = bimodal (polysemantic candidate). <0.1 = unimodal.
    Returns 0 if fewer than 4 real words (can't reliably bimodal-test).
    """
    real = [t for t in tokens if is_real_word(t)]
    if len(real) < 4:
        return 0.0
    bigrams = [char_bigrams(t) for t in real]
    n = len(real)
    # Pairwise distances
    dist = [[1.0 - jaccard(bigrams[i], bigrams[j]) for j in range(n)] for i in range(n)]
    # Hierarchical average-linkage 2-cluster: try all balanced splits, pick best avg silhouette
    best_sil = 0.0
    # Brute force across all 2-partitions (n<=30 so manageable; but n is small here, real words rarely >10)
    if n > 12:
        # Cap exhaustive search
        return 0.0
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
    """L<l>_F<f> -> (l, f)."""
    l_str, f_str = k.split("_F")
    return int(l_str[1:]), int(f_str)


def load_cumulative_wn():
    """Load the union of wn:* feature keys across canonical, multilingual, subword pilots.
    Returns: {feature_key: {"sources": [...], "labels_from_each": {...}}}
    """
    vindex = Path("/Users/christopherhay/chris-source/larql/output/gemma3-4b-v2.vindex")
    canonical = json.load(open(vindex / "feature_labels.json"))
    ml = json.load(open(vindex / "feature_labels_multilingual_pilot.json"))
    sw = json.load(open(vindex / "feature_labels_subword_pilot.json"))

    inventory = {}

    for k, v in canonical.items():
        if isinstance(v, str) and v.startswith("wn:"):
            inventory[k] = {"sources": ["canonical"], "labels_from_each": {"canonical": v}}

    for k, v in ml.items():
        if isinstance(v, str) and v.startswith("wn:"):
            if k in inventory:
                inventory[k]["sources"].append("multilingual")
                inventory[k]["labels_from_each"]["multilingual"] = v
            else:
                inventory[k] = {"sources": ["multilingual"], "labels_from_each": {"multilingual": v}}

    for k, v in sw.items():
        if isinstance(v, str) and v.startswith("wn:"):
            if k in inventory:
                inventory[k]["sources"].append("subword")
                inventory[k]["labels_from_each"]["subword"] = v
            else:
                inventory[k] = {"sources": ["subword"], "labels_from_each": {"subword": v}}

    return inventory


def classify(metrics, cutoffs):
    """Apply classification logic per spec pseudocode.
    Returns one of: promiscuous, polysemantic, mono_semantic, borderline.
    """
    rwr = metrics["real_word_ratio"]
    mtl = metrics["mean_token_length"]
    sim = metrics["real_word_coherence"]
    bim = metrics["bimodality_score"]
    nrw = metrics["num_real_words"]

    # Promiscuous: low real-word content AND low coherence
    if rwr < cutoffs["promiscuous_rwr_max"] and sim < cutoffs["promiscuous_sim_max"]:
        return "promiscuous"

    # Polysemantic: high bimodality with enough real words
    if bim > cutoffs["polysemantic_bim_min"] and nrw >= 4:
        return "polysemantic"

    # Mono-semantic: either high real-word ratio OR high coherence
    # The OR is important: short-word clusters (L0_F5560-style) have low rwr
    # but high coherence among the real words; long-word clusters have high rwr
    if rwr >= cutoffs["mono_rwr_min"] or sim >= cutoffs["mono_sim_min"]:
        return "mono_semantic"

    return "borderline"


def search_cutoffs(metrics_per_feature, anchor_promisc, anchor_mono):
    """Search the cutoff space for a combination satisfying anchor constraints.
    Returns (best_cutoffs, success_bool, anchor_check_results).
    """
    # Coarse grid search. Cheap because we only have 129 features.
    rwr_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sim_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    bim_grid = [0.30, 0.40, 0.50, 0.60]

    best_cutoffs = None
    best_score = -1
    best_anchor_check = None

    for rwr_max in rwr_grid:
        for sim_max in sim_grid:
            for mono_rwr in [v for v in rwr_grid if v >= rwr_max]:
                for mono_sim in [v for v in sim_grid if v >= sim_max]:
                    for bim_min in bim_grid:
                        cutoffs = {
                            "promiscuous_rwr_max": rwr_max,
                            "promiscuous_sim_max": sim_max,
                            "mono_rwr_min": mono_rwr,
                            "mono_sim_min": mono_sim,
                            "polysemantic_bim_min": bim_min,
                        }
                        # Check anchors
                        anchor_check = {}
                        all_pass = True
                        for k in anchor_promisc:
                            if k not in metrics_per_feature:
                                anchor_check[k] = ("MISSING", "required: promiscuous")
                                all_pass = False
                                continue
                            cls = classify(metrics_per_feature[k], cutoffs)
                            anchor_check[k] = (cls, "required: promiscuous")
                            if cls != "promiscuous":
                                all_pass = False
                        for k in anchor_mono:
                            if k not in metrics_per_feature:
                                anchor_check[k] = ("MISSING", "required: mono_semantic")
                                all_pass = False
                                continue
                            cls = classify(metrics_per_feature[k], cutoffs)
                            anchor_check[k] = (cls, "required: mono_semantic")
                            if cls != "mono_semantic":
                                all_pass = False
                        if all_pass:
                            # Score: prefer minimal borderline + minimal extreme promiscuous classifications
                            classifications = Counter(
                                classify(m, cutoffs) for m in metrics_per_feature.values()
                            )
                            borderline_pct = classifications.get("borderline", 0) / len(metrics_per_feature)
                            # Prefer fewer borderlines, but also reasonable mono fraction (in P4 70-90%)
                            mono_pct = classifications.get("mono_semantic", 0) / len(metrics_per_feature)
                            in_band = 0.70 <= mono_pct <= 0.90
                            score = (1.0 if all_pass else 0) + (1.0 if in_band else 0) - borderline_pct
                            if score > best_score:
                                best_score = score
                                best_cutoffs = cutoffs
                                best_anchor_check = anchor_check
    return best_cutoffs, best_cutoffs is not None, best_anchor_check


def main():
    vindex = Path("/Users/christopherhay/chris-source/larql/output/gemma3-4b-v2.vindex")
    print(f"Loading vindex: {vindex}")
    config, _gates, down_meta = pm.load_vindex_gates_and_meta(str(vindex))
    print(f"  Loaded down_meta for {len(down_meta)} (layer, feature) entries")

    print("\nBuilding cumulative wn:* inventory...")
    inventory = load_cumulative_wn()
    print(f"  Cumulative wn:* count: {len(inventory)}")
    sources_dist = Counter(tuple(sorted(v["sources"])) for v in inventory.values())
    print(f"  Source distribution: {dict(sources_dist)}")

    print("\nComputing metrics per feature (no model inference)...")
    metrics_per_feature = {}
    missing_tokens = []
    for feat_key, info in inventory.items():
        try:
            l, f = featkey_to_lf(feat_key)
        except (ValueError, IndexError):
            continue
        tokens = down_meta.get((l, f), [])
        if not tokens:
            missing_tokens.append(feat_key)
            continue
        real_words = [t.lstrip("▁").strip() for t in tokens if is_real_word(t)]
        metrics = {
            "n_tokens": len(tokens),
            "num_real_words": len(real_words),
            "real_word_ratio": real_word_ratio(tokens),
            "mean_token_length": mean_token_length(tokens),
            "real_word_coherence": real_word_coherence(tokens),
            "bimodality_score": bimodality(tokens),
            "down_meta_full": tokens,
            "top_real_words": real_words[:10],
            "script_distribution": dict(Counter(script_tag(t) for t in tokens)),
            "sources": info["sources"],
            "labels_from_each": info["labels_from_each"],
        }
        metrics_per_feature[feat_key] = metrics

    print(f"  Features with metrics: {len(metrics_per_feature)}")
    print(f"  Features missing down_meta: {len(missing_tokens)} ({missing_tokens[:5]}...)")

    # Distributions for visibility
    rwrs = sorted(m["real_word_ratio"] for m in metrics_per_feature.values())
    mtls = sorted(m["mean_token_length"] for m in metrics_per_feature.values())
    sims = sorted(m["real_word_coherence"] for m in metrics_per_feature.values())
    bims = sorted(m["bimodality_score"] for m in metrics_per_feature.values())

    def q(arr, p):
        return arr[int(len(arr) * p)] if arr else 0

    print(f"\nMetric distributions (n={len(metrics_per_feature)}):")
    print(f"  real_word_ratio:    min={rwrs[0]:.2f}  p25={q(rwrs,0.25):.2f}  p50={q(rwrs,0.50):.2f}  p75={q(rwrs,0.75):.2f}  max={rwrs[-1]:.2f}")
    print(f"  mean_token_length:  min={mtls[0]:.2f}  p25={q(mtls,0.25):.2f}  p50={q(mtls,0.50):.2f}  p75={q(mtls,0.75):.2f}  max={mtls[-1]:.2f}")
    print(f"  real_word_coherence: min={sims[0]:.3f} p25={q(sims,0.25):.3f} p50={q(sims,0.50):.3f} p75={q(sims,0.75):.3f} max={sims[-1]:.3f}")
    print(f"  bimodality_score:   min={bims[0]:.3f} p25={q(bims,0.25):.3f} p50={q(bims,0.50):.3f} p75={q(bims,0.75):.3f} max={bims[-1]:.3f}")

    print("\nAnchor feature metrics:")
    for k in sorted(ANCHOR_PROMISCUOUS | ANCHOR_MONO):
        if k in metrics_per_feature:
            m = metrics_per_feature[k]
            req = "promiscuous" if k in ANCHOR_PROMISCUOUS else "mono_semantic"
            print(f"  {k} (req={req}): rwr={m['real_word_ratio']:.2f} mtl={m['mean_token_length']:.2f} sim={m['real_word_coherence']:.3f} bim={m['bimodality_score']:.3f} nrw={m['num_real_words']}")
        else:
            print(f"  {k}: MISSING FROM INVENTORY")

    print("\nSearching cutoff space for anchor-satisfying combination...")
    cutoffs, success, anchor_check = search_cutoffs(
        metrics_per_feature, ANCHOR_PROMISCUOUS, ANCHOR_MONO
    )

    if not success:
        print("\n!!! METRICS INSUFFICIENT !!!")
        print("No cutoff combination satisfies all four anchor constraints.")
        print("Audit reports the conflict rather than forcing a classification.")
        # Still emit the metrics for diagnosis
        out_path = vindex / "feature_labels_polysemy_audit.json"
        with open(out_path, "w") as f:
            json.dump({
                "status": "METRICS_INSUFFICIENT",
                "reason": "no cutoff combination satisfies all four anchor constraints",
                "anchors_required": {
                    "promiscuous": sorted(ANCHOR_PROMISCUOUS),
                    "mono_semantic": sorted(ANCHOR_MONO),
                },
                "metrics_per_feature": metrics_per_feature,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Metrics-insufficient report -> {out_path}")
        return

    print(f"  Cutoffs found:")
    for k, v in cutoffs.items():
        print(f"    {k}: {v}")
    print(f"  Anchor check:")
    for k, (got, req) in anchor_check.items():
        marker = "OK" if got in req else ""
        print(f"    {k}: got={got}, {req} [{marker}]")

    # Apply classification to all
    print("\nApplying classification...")
    classifications = {}
    for k, m in metrics_per_feature.items():
        classifications[k] = classify(m, cutoffs)

    dist = Counter(classifications.values())
    total = len(classifications)
    print(f"\nClassification distribution (n={total}):")
    for cls, n in dist.most_common():
        print(f"  {cls:<15} {n:4d} ({n/total*100:>5.1f}%)")

    # Save full audit
    full_audit = {}
    for k, m in metrics_per_feature.items():
        cls = classifications[k]
        full_audit[k] = {
            "classification": cls,
            "real_word_ratio": round(m["real_word_ratio"], 3),
            "mean_token_length": round(m["mean_token_length"], 2),
            "real_word_coherence": round(m["real_word_coherence"], 3),
            "bimodality_score": round(m["bimodality_score"], 3),
            "num_real_words": m["num_real_words"],
            "top_real_words": m["top_real_words"],
            "down_meta_full": m["down_meta_full"],
            "labels_from": m["labels_from_each"],
            "sources": m["sources"],
        }

    out_path = vindex / "feature_labels_polysemy_audit.json"
    with open(out_path, "w") as f:
        json.dump(full_audit, f, indent=2, ensure_ascii=False)
    print(f"\nFull audit -> {out_path}")

    # Summary
    by_relation = defaultdict(lambda: Counter())
    by_layer = defaultdict(lambda: Counter())
    for k, cls in classifications.items():
        m = metrics_per_feature[k]
        # Pick a representative label (canonical preferred, else first available)
        labels = m["labels_from_each"]
        rep_label = labels.get("canonical") or labels.get("multilingual") or labels.get("subword")
        try:
            l, _ = featkey_to_lf(k)
            by_layer[l][cls] += 1
        except (ValueError, IndexError):
            pass
        if rep_label:
            by_relation[rep_label][cls] += 1

    mono_pct = dist.get("mono_semantic", 0) / total * 100
    poly_pct = dist.get("polysemantic", 0) / total * 100
    promisc_pct = dist.get("promiscuous", 0) / total * 100
    borderline_pct = dist.get("borderline", 0) / total * 100

    # P4 outcome
    p4_pred = {"mono": (70, 90), "promiscuous": (5, 25), "polysemantic_max": 10}
    mono_in = p4_pred["mono"][0] <= mono_pct <= p4_pred["mono"][1]
    promisc_in = p4_pred["promiscuous"][0] <= promisc_pct <= p4_pred["promiscuous"][1]
    poly_in = poly_pct <= p4_pred["polysemantic_max"]
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
        "total_audited": total,
        "missing_down_meta": len(missing_tokens),
        "classification_counts": dict(dist),
        "classification_pct": {
            "mono_semantic": round(mono_pct, 1),
            "polysemantic": round(poly_pct, 1),
            "promiscuous": round(promisc_pct, 1),
            "borderline": round(borderline_pct, 1),
        },
        "stable_count": dist.get("mono_semantic", 0) + dist.get("polysemantic", 0),
        "classification_distribution_by_relation": {
            k: dict(v) for k, v in by_relation.items()
        },
        "classification_distribution_by_layer": {
            int(k): dict(v) for k, v in sorted(by_layer.items())
        },
        "cutoffs_used": cutoffs,
        "anchor_check": {k: {"got": got, "required": req, "satisfied": got in req}
                         for k, (got, req) in anchor_check.items()},
        "tests_prediction": "P4 in META_MODEL.md",
        "P4_predicted": {
            "mono_semantic_pct": "70-90",
            "promiscuous_pct": "5-25",
            "polysemantic_pct": "<10",
        },
        "P4_observed_pct": {
            "mono_semantic_pct": round(mono_pct, 1),
            "promiscuous_pct": round(promisc_pct, 1),
            "polysemantic_pct": round(poly_pct, 1),
            "borderline_pct": round(borderline_pct, 1),
        },
        "P4_outcome": p4_result,
        "launch_1c_ok": launch_1c_ok,
        "launch_1c_blocker_reason": None if launch_1c_ok else (
            f"polysemy {poly_pct:.1f}% > 20% threshold" if poly_pct > 20
            else f"promiscuity {promisc_pct:.1f}% > 30% threshold"
        ),
    }

    summary_path = vindex / "feature_labels_polysemy_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary -> {summary_path}")

    print("\n" + "=" * 60)
    print("P4 OUTCOME EVALUATION")
    print("=" * 60)
    print(f"Predicted:   mono 70-90%, promiscuous 5-25%, polysemantic <10%")
    print(f"Observed:    mono {mono_pct:.1f}%, promiscuous {promisc_pct:.1f}%, polysemantic {poly_pct:.1f}%, borderline {borderline_pct:.1f}%")
    print(f"Stable count: {summary['stable_count']} / {total} (mono + polysemantic)")
    print(f"P4 outcome:  {p4_result}")
    print(f"\nLaunch 1c OK: {launch_1c_ok}")
    if not launch_1c_ok:
        print(f"  Blocker: {summary['launch_1c_blocker_reason']}")
        print(f"  Per cold-pickup protocol: DO NOT launch 1c. Working model needs revision.")


if __name__ == "__main__":
    main()
