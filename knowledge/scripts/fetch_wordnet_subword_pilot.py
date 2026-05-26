#!/usr/bin/env python3
"""Pilot fetcher: harvest WordNet pairs canonical did NOT see, restricted to
subjects that tokenize to >=2 BPE pieces in the target model's tokenizer.

Parallels scripts/fetch_wordnet_multilingual_pilot.py but along a different axis:
multilingual added new lemmas in new languages; this adds long-tail English
lemmas that canonical never reached (canonical capped at 5000/3000/2000/2000/5000
pairs per relation; long-tail synsets are systematically further down WordNet's
walk order and skew toward rarer, more multi-piece vocabulary).

Decision-rule prerequisite: canonical was profiled to already cover 67-92% of
multi-piece subjects per relation, so a naive "filter canonical to multi-piece"
would not give the probe new data. This fetcher instead walks WordNet from the
beginning, skips every pair canonical already harvested, then keeps only
multi-piece subjects.

Output: data/wordnet_subword_pilot.json   (separate from canonical
        data/wordnet_relations.json; probe consumes standalone).
"""

import json
import os
import sys
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)


# Pull a generous long-tail per relation; the BPE filter + canonical-skip will
# trim. Empirically canonical took ~5k synonym pairs; pulling 15k targets ~10k new.
PAIRS_PER_RELATION = 15000

# Final cap on multi-piece, non-canonical pairs per relation. Keeps probe runtime
# comparable to multilingual pilot (~40min on 14.5k subjects).
MAX_NEW_PAIRS_PER_RELATION = 3000


def ensure_data():
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)


def load_canonical_seen_pairs(canonical_path: Path) -> dict:
    """Return {relation: set of (a, b) tuples} of pairs canonical already harvested.

    Used to skip these in long-tail walk so 1b sees genuinely new data.
    """
    with open(canonical_path) as f:
        canonical = json.load(f)
    seen = {}
    for rel in ["synonym", "hypernym", "antonym", "meronym", "derivation"]:
        pairs = canonical.get(rel, {}).get("pairs", [])
        seen[rel] = {(a, b) for a, b in pairs}
    return seen


def _extract_synonyms_long_tail(limit, seen_canonical):
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        lemmas = [l.name().replace("_", " ").lower() for l in synset.lemmas()
                  if l.name().isalpha() and len(l.name()) >= 3]
        for i in range(len(lemmas)):
            for j in range(i + 1, len(lemmas)):
                a, b = lemmas[i], lemmas[j]
                if a == b or (a, b) in seen or (a, b) in seen_canonical:
                    continue
                pairs.append([a, b])
                seen.add((a, b))
                seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def _extract_hypernyms_long_tail(limit, seen_canonical):
    pairs, seen = [], set()
    for synset in wn.all_synsets("n"):
        lemmas = synset.lemmas()
        if not lemmas:
            continue
        word = lemmas[0].name().replace("_", " ").lower()
        if not word.isalpha() or len(word) < 3:
            continue
        for hyper in synset.hypernyms():
            hl = hyper.lemmas()
            if not hl:
                continue
            parent = hl[0].name().replace("_", " ").lower()
            if not (parent.isalpha() and len(parent) >= 3 and word != parent):
                continue
            if (word, parent) in seen or (word, parent) in seen_canonical:
                continue
            pairs.append([word, parent])
            seen.add((word, parent))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def _extract_antonyms_long_tail(limit, seen_canonical):
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                a = lemma.name().replace("_", " ").lower()
                b = ant.name().replace("_", " ").lower()
                if not (a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3 and a != b):
                    continue
                if (a, b) in seen or (a, b) in seen_canonical:
                    continue
                pairs.append([a, b])
                seen.add((a, b))
                seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def _extract_meronyms_long_tail(limit, seen_canonical):
    pairs, seen = [], set()
    for synset in wn.all_synsets("n"):
        lemmas = synset.lemmas()
        if not lemmas:
            continue
        word = lemmas[0].name().replace("_", " ").lower()
        if not word.isalpha() or len(word) < 3:
            continue
        for mero_fn in [synset.part_meronyms, synset.member_meronyms, synset.substance_meronyms]:
            for mero in mero_fn():
                ml = mero.lemmas()
                if not ml:
                    continue
                part = ml[0].name().replace("_", " ").lower()
                if not (part.isalpha() and len(part) >= 3 and part != word):
                    continue
                if (part, word) in seen or (part, word) in seen_canonical:
                    continue
                pairs.append([part, word])
                seen.add((part, word))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def _extract_derivations_long_tail(limit, seen_canonical):
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                a = lemma.name().replace("_", " ").lower()
                b = related.name().replace("_", " ").lower()
                if not (a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3 and a != b):
                    continue
                if (a, b) in seen or (a, b) in seen_canonical:
                    continue
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


EXTRACTORS = {
    "synonym": _extract_synonyms_long_tail,
    "hypernym": _extract_hypernyms_long_tail,
    "antonym": _extract_antonyms_long_tail,
    "meronym": _extract_meronyms_long_tail,
    "derivation": _extract_derivations_long_tail,
}


def main():
    ensure_data()

    knowledge_dir = Path(__file__).parent.parent
    canonical_path = knowledge_dir / "data" / "wordnet_relations.json"
    if not canonical_path.exists():
        print(f"ERROR: canonical {canonical_path} not found. Run fetch_wordnet_relations.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading canonical seen-pairs index from {canonical_path}...")
    seen_canonical = load_canonical_seen_pairs(canonical_path)
    for rel, pairs in seen_canonical.items():
        print(f"  {rel:<12} canonical has {len(pairs)} pairs")

    print("\nLoading model tokenizer (for piece-count filter)...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm import load as mlx_load
    _, tokenizer = mlx_load("google/gemma-3-4b-it")
    print("  Tokenizer loaded.")

    def n_pieces(s):
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(s, add_special_tokens=False))
        return len(tokenizer(s))

    print(f"\nHarvesting long-tail pairs per relation (target {PAIRS_PER_RELATION}, cap"
          f" {MAX_NEW_PAIRS_PER_RELATION} multi-piece-new)...")
    out_relations = {}
    per_rel_log = {}
    for rel_name, extractor in EXTRACTORS.items():
        print(f"  {rel_name}: walking long-tail...", end=" ", flush=True)
        new_pairs = extractor(PAIRS_PER_RELATION, seen_canonical[rel_name])
        print(f"{len(new_pairs)} new (post-canonical-skip)", end=" ", flush=True)

        # Filter to multi-piece subjects
        filtered = []
        for a, b in new_pairs:
            if 2 <= len(a) <= 30 and n_pieces(a) >= 2:
                filtered.append([a, b])
                if len(filtered) >= MAX_NEW_PAIRS_PER_RELATION:
                    break
        out_relations[rel_name] = {"pairs": filtered}
        per_rel_log[rel_name] = {
            "long_tail_new": len(new_pairs),
            "multi_piece_kept": len(filtered),
        }
        print(f"-> {len(filtered)} multi-piece kept")

    total_pairs = sum(len(d["pairs"]) for d in out_relations.values())
    total_subjects = len({p[0] for d in out_relations.values() for p in d["pairs"]})
    print(f"\nTotal: {total_pairs} pairs, {total_subjects} unique subjects (all multi-piece)")

    output_path = knowledge_dir / "data" / "wordnet_subword_pilot.json"
    with open(output_path, "w") as f:
        json.dump(out_relations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {output_path}")

    # Provenance / pre-registration record
    provenance_path = knowledge_dir / "data" / "wordnet_subword_pilot_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump({
            "pilot": "1b_subword_fragmentation",
            "design": "long-tail WordNet walk skipping canonical pairs, filtered to multi-piece subjects",
            "rationale": "canonical already covers 67-92% multi-piece subjects, so filter-only is not new data; long-tail walk gives genuinely new pairs that skew rare/multi-piece",
            "pairs_per_relation_target": PAIRS_PER_RELATION,
            "max_new_per_relation": MAX_NEW_PAIRS_PER_RELATION,
            "piece_count_min": 2,
            "tokenizer": "google/gemma-3-4b-it",
            "per_relation": per_rel_log,
            "total_pairs": total_pairs,
            "total_unique_subjects": total_subjects,
        }, f, indent=2, ensure_ascii=False)
    print(f"Provenance -> {provenance_path}")

    print("\nSample pairs per relation:")
    for rel, data in out_relations.items():
        examples = data["pairs"][:5]
        sample = ", ".join(f"{a}->{b}" for a, b in examples)
        print(f"  {rel:<12} {len(data['pairs']):5d}  [{sample}]")


if __name__ == "__main__":
    main()
