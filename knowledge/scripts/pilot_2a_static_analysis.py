#!/usr/bin/env python3
"""Pilot 2a: static analysis of vindex features that failed canonical labeling.

Read-only audit of the vindex artifact. No model inference. Asks: of the syntax-layer
features (L0-L12) that have non-trivial down_meta but did NOT receive a wn:* label
in canonical, what do their top-output tokens look like? Are there systematic
patterns (single writing system, mixed scripts, punctuation-heavy, language-specific)
that point at a methodology blind spot?

This is the cheap front-line of Direction 2. It does not run the model; it just
inspects the down_meta token lists and categorizes them. Output is a JSON report
that feeds the decision about whether to launch 2b/2c.
"""

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import probe_mlx as pm  # noqa: E402

_KNOWLEDGE_DIR = _SCRIPT_DIR.parent


def writing_system(token: str) -> str:
    """Classify a token by dominant unicode script. Returns one of:
    latin, cyrillic, greek, arabic, hebrew, cjk, devanagari, thai, other_script,
    digit, punct, mixed, empty.
    """
    s = token.strip()
    if not s:
        return "empty"
    # Strip BPE leading marker
    if s.startswith("▁"):  # ▁ marker for SentencePiece
        s = s[1:]
    if not s:
        return "empty"

    scripts = Counter()
    has_alpha = False
    has_digit = False
    has_punct = False
    for ch in s:
        if ch.isdigit():
            has_digit = True
            continue
        if not ch.isalpha():
            if unicodedata.category(ch).startswith("P"):
                has_punct = True
            continue
        has_alpha = True
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        if "LATIN" in name:
            scripts["latin"] += 1
        elif "CYRILLIC" in name:
            scripts["cyrillic"] += 1
        elif "GREEK" in name:
            scripts["greek"] += 1
        elif "ARABIC" in name:
            scripts["arabic"] += 1
        elif "HEBREW" in name:
            scripts["hebrew"] += 1
        elif any(k in name for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            scripts["cjk"] += 1
        elif "DEVANAGARI" in name:
            scripts["devanagari"] += 1
        elif "THAI" in name:
            scripts["thai"] += 1
        else:
            scripts["other_script"] += 1

    if not has_alpha:
        if has_digit and not has_punct:
            return "digit"
        if has_punct and not has_digit:
            return "punct"
        return "other"

    if len(scripts) == 1:
        return next(iter(scripts))
    if len(scripts) > 1:
        return "mixed"
    return "other"


def categorize_feature(tokens: list, canonical_label) -> dict:
    """Categorize one feature's down_meta tokens."""
    script_counts = Counter(writing_system(t) for t in tokens)
    dominant_script, dom_count = script_counts.most_common(1)[0] if script_counts else ("empty", 0)
    dom_share = dom_count / max(len(tokens), 1)
    is_multi_script = sum(1 for s in script_counts if s in {"latin", "cyrillic", "greek", "arabic", "hebrew", "cjk", "devanagari", "thai", "other_script"}) > 1

    # Length distribution: tokens that look multi-piece (start with ##/▁ or are
    # short fragments) vs full words
    stripped = [(t[1:] if t.startswith("▁") else t).strip() for t in tokens]
    short_frags = sum(1 for s in stripped if 1 <= len(s) <= 2)
    short_frag_share = short_frags / max(len(stripped), 1)

    # Punctuation/digit heavy?
    non_alpha_share = (script_counts.get("punct", 0) + script_counts.get("digit", 0)
                       + script_counts.get("other", 0) + script_counts.get("empty", 0)) / max(len(tokens), 1)

    return {
        "n_tokens": len(tokens),
        "dominant_script": dominant_script,
        "dominant_share": round(dom_share, 3),
        "is_multi_script": is_multi_script,
        "script_distribution": dict(script_counts),
        "short_fragment_share": round(short_frag_share, 3),
        "non_alpha_share": round(non_alpha_share, 3),
        "canonical_label": canonical_label,
        "sample_tokens": tokens[:10],
    }


def main():
    output_root = _KNOWLEDGE_DIR.parent / "output"
    vindex_path = output_root / "gemma3-4b-v2.vindex"
    if not vindex_path.exists():
        print(f"ERROR: vindex not found at {vindex_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading vindex: {vindex_path}")
    config, _gates, down_meta = pm.load_vindex_gates_and_meta(str(vindex_path))
    num_layers = config["num_layers"]
    print(f"  {num_layers} layers, {len(down_meta)} (layer,feat) entries with tokens")

    if "layer_bands" in config and config["layer_bands"]:
        syntax_end = config["layer_bands"].get("knowledge_start", num_layers * 2 // 5)
    else:
        syntax_end = num_layers * 2 // 5
    print(f"  Syntax band: L0-L{syntax_end - 1}")

    canonical_path = vindex_path / "feature_labels.json"
    with open(canonical_path) as f:
        canonical_labels = json.load(f)
    print(f"  Canonical labels: {len(canonical_labels)} entries")

    syntax_features = {(l, f): tokens for (l, f), tokens in down_meta.items()
                       if 0 <= l < syntax_end and tokens}
    print(f"  Syntax-band features with tokens: {len(syntax_features)}")

    wn_labeled = set()
    other_labeled = set()
    for key, label in canonical_labels.items():
        if not (isinstance(key, str) and key.startswith("L") and "_F" in key):
            continue
        try:
            l_str, f_str = key.split("_F")
            layer = int(l_str[1:])
            feat = int(f_str)
        except ValueError:
            continue
        if not (0 <= layer < syntax_end):
            continue
        if isinstance(label, str) and label.startswith("wn:"):
            wn_labeled.add((layer, feat))
        else:
            other_labeled.add((layer, feat))

    unlabeled = set(syntax_features.keys()) - wn_labeled - other_labeled
    print(f"\nSyntax-band feature partition:")
    print(f"  wn:* labeled (canonical):    {len(wn_labeled)}")
    print(f"  other-labeled (canonical):   {len(other_labeled)}")
    print(f"  unlabeled:                   {len(unlabeled)}")

    # Categorize each unlabeled feature
    print(f"\nCategorizing {len(unlabeled)} unlabeled syntax-band features...")
    categories = {}
    for (l, f) in sorted(unlabeled):
        tokens = syntax_features[(l, f)]
        categories[f"L{l}_F{f}"] = categorize_feature(tokens, None)

    # Also categorize wn-labeled and other-labeled for comparison
    wn_categories = {}
    for (l, f) in sorted(wn_labeled):
        tokens = syntax_features.get((l, f), [])
        if tokens:
            wn_categories[f"L{l}_F{f}"] = categorize_feature(
                tokens, canonical_labels.get(f"L{l}_F{f}"))

    other_categories = {}
    for (l, f) in sorted(other_labeled):
        tokens = syntax_features.get((l, f), [])
        if tokens:
            other_categories[f"L{l}_F{f}"] = categorize_feature(
                tokens, canonical_labels.get(f"L{l}_F{f}"))

    # Aggregate stats
    def agg(cats):
        if not cats:
            return {}
        scripts = Counter(c["dominant_script"] for c in cats.values())
        multi_script_count = sum(1 for c in cats.values() if c["is_multi_script"])
        short_frag_heavy = sum(1 for c in cats.values() if c["short_fragment_share"] > 0.3)
        non_alpha_heavy = sum(1 for c in cats.values() if c["non_alpha_share"] > 0.3)
        layer_dist = Counter(int(k.split("_F")[0][1:]) for k in cats.keys())
        return {
            "count": len(cats),
            "dominant_script_distribution": dict(scripts.most_common()),
            "multi_script_count": multi_script_count,
            "multi_script_share": round(multi_script_count / len(cats), 3),
            "short_frag_heavy_count": short_frag_heavy,
            "non_alpha_heavy_count": non_alpha_heavy,
            "layer_distribution": dict(sorted(layer_dist.items())),
        }

    summary = {
        "wn_labeled": agg(wn_categories),
        "other_labeled": agg(other_categories),
        "unlabeled": agg(categories),
    }

    print("\n" + "=" * 60)
    print("UNLABELED FEATURE CHARACTERIZATION")
    print("=" * 60)
    for partition_name, stats in summary.items():
        if not stats:
            continue
        print(f"\n[{partition_name}] {stats['count']} features")
        print(f"  Dominant scripts:    {stats['dominant_script_distribution']}")
        print(f"  Multi-script:        {stats['multi_script_count']} ({stats['multi_script_share']:.1%})")
        print(f"  Short-frag heavy:    {stats['short_frag_heavy_count']}")
        print(f"  Non-alpha heavy:     {stats['non_alpha_heavy_count']}")
        print(f"  Layer distribution:  {stats['layer_distribution']}")

    # Mixed-vocab features = unlabeled + multi-script. The hypothesis we want to
    # test: are the unlabeled features systematically multi-script (suggesting
    # the methodology fails on cross-language features), or systematically
    # short-fragment (suggesting BPE-piece sensitivity), or neither?
    mixed_vocab_unlabeled = {k: v for k, v in categories.items() if v["is_multi_script"]}
    print(f"\nMixed-vocab unlabeled features: {len(mixed_vocab_unlabeled)}")
    if mixed_vocab_unlabeled:
        # Sample 10 for the report
        print("  Sample (10):")
        for k, v in list(mixed_vocab_unlabeled.items())[:10]:
            print(f"    {k}: scripts={v['script_distribution']}, tokens={v['sample_tokens'][:5]}")

    short_frag_unlabeled = {k: v for k, v in categories.items() if v["short_fragment_share"] > 0.3}
    print(f"\nShort-fragment-heavy unlabeled features: {len(short_frag_unlabeled)}")
    if short_frag_unlabeled:
        print("  Sample (10):")
        for k, v in list(short_frag_unlabeled.items())[:10]:
            print(f"    {k}: short_frag={v['short_fragment_share']}, tokens={v['sample_tokens'][:5]}")

    # Save detailed report
    out_path = vindex_path / "feature_labels_pilot_2a_static.json"
    with open(out_path, "w") as f:
        json.dump({
            "pilot_name": "2a_static_analysis",
            "design": "read-only audit of vindex down_meta for syntax-band features; no model inference",
            "summary": summary,
            "unlabeled_categories": categories,
            "mixed_vocab_unlabeled_count": len(mixed_vocab_unlabeled),
            "short_frag_unlabeled_count": len(short_frag_unlabeled),
            "mixed_vocab_unlabeled_features": list(mixed_vocab_unlabeled.keys()),
            "short_frag_unlabeled_features": list(short_frag_unlabeled.keys()),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {out_path}")

    # Decision rule for 2a -> 2b/2c
    n_unlabeled = len(categories)
    multi_share = summary["unlabeled"]["multi_script_share"] if summary["unlabeled"] else 0
    print("\n" + "=" * 60)
    print("2A DECISION RULE")
    print("=" * 60)
    # Compare unlabeled multi-script share to wn-labeled multi-script share
    wn_multi_share = summary["wn_labeled"].get("multi_script_share", 0) if summary["wn_labeled"] else 0
    delta = multi_share - wn_multi_share
    print(f"Multi-script share, unlabeled:  {multi_share:.1%}")
    print(f"Multi-script share, wn-labeled: {wn_multi_share:.1%}")
    print(f"Delta:                          {delta:+.1%}")
    if delta > 0.15:
        decision = "Run 2b (activation context sampling on mixed-vocab subset) — unlabeled features are systematically more multi-script than labeled."
    elif delta < -0.05:
        decision = "Methodology is NOT blind to multi-script — unlabeled features are LESS multi-script than labeled. Don't pursue 2b on multi-script axis; check 2c (semantic-coverage audit) instead."
    else:
        decision = "Multi-script not a distinguishing axis between labeled and unlabeled. 2b mixed-vocab hypothesis weak. Pivot to 2c semantic coverage."
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
