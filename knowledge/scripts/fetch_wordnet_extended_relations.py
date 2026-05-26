#!/usr/bin/env python3
"""Pilot 1c fetcher: harvest WordNet relations NOT covered by canonical's 5
(synonym, hypernym, antonym, meronym, derivation). Six new candidates split
into two categories:

  Adjective-side (likely L0-L12, hypothesized contributors):
    - pertainym       (adjective → derivation-source noun, e.g., american → America)
    - similar_to      (adjective synset cross-reference, e.g., good ↔ bang-up)
    - attribute       (adjective synset → noun attribute, e.g., good → quality)
    - also_see        (synset cross-reference, mostly adjectives)

  Verb-side (hypothesized to live at L13+ if at all):
    - entailment      (verb V entails verb W, e.g., snore → sleep)
    - cause           (verb V causes verb W, e.g., kill → die)

This split is load-bearing for 1c's decision rule. If adjective-side returns
new labels but verb-side returns ~zero at L0-L12, the depth-stratification
hypothesis is supported (and verb-side would need a deeper-layer probe to test).

Filter: pairs where subject and target are alpha, length >= 3. NO multi-piece
filter (1c tests relation coverage, not surface form). NO canonical-pair-skip
(1c relations don't appear in canonical, so no overlap to dedupe).

Output: data/wordnet_extended_relations.json
"""

import json
import sys
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)


PAIRS_PER_RELATION = 3000


def ensure_data():
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)


def _lemma_name(lemma):
    n = lemma.name().replace("_", " ").lower()
    return n if n.isalpha() and len(n) >= 3 else None


def _first_lemma(synset):
    """Return first valid lemma name from a synset, or None."""
    for lem in synset.lemmas():
        n = _lemma_name(lem)
        if n:
            return n
    return None


def extract_pertainyms(limit):
    """Lemma-level: pertainym(adj) -> noun source.
    E.g., 'american' (adj) → 'America' (noun). Dense for adjectives derived
    from proper/common nouns.
    """
    pairs, seen = [], set()
    for synset in wn.all_synsets("a"):
        for lemma in synset.lemmas():
            a = _lemma_name(lemma)
            if not a:
                continue
            for pert in lemma.pertainyms():
                b = _lemma_name(pert)
                if b and a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_similar_tos(limit):
    """Synset-level: similar_to between adjective synsets. Adjective-only.
    Pair: first lemma of source synset, first lemma of target synset.
    """
    pairs, seen = [], set()
    for synset in wn.all_synsets("a"):
        a = _first_lemma(synset)
        if not a:
            continue
        for sim in synset.similar_tos():
            b = _first_lemma(sim)
            if b and a != b and (a, b) not in seen:
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_attributes(limit):
    """Synset-level: adjective → attribute noun.
    E.g., good.a.01 → quality.n.01. Sparse but semantically central.
    """
    pairs, seen = [], set()
    # Attributes can go in either direction (adj.attribute() returns noun;
    # noun.attribute() returns adj). Harvest both for completeness.
    for synset in wn.all_synsets("a"):
        a = _first_lemma(synset)
        if not a:
            continue
        for attr in synset.attributes():
            b = _first_lemma(attr)
            if b and a != b and (a, b) not in seen:
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    if len(pairs) < limit:
        for synset in wn.all_synsets("n"):
            a = _first_lemma(synset)
            if not a:
                continue
            for attr in synset.attributes():
                b = _first_lemma(attr)
                if b and a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
            if len(pairs) >= limit:
                break
    return pairs[:limit]


def extract_also_sees(limit):
    """Synset-level: cross-reference between related synsets, primarily adj."""
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        a = _first_lemma(synset)
        if not a:
            continue
        for also in synset.also_sees():
            b = _first_lemma(also)
            if b and a != b and (a, b) not in seen:
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_entailments(limit):
    """Synset-level: verb V entails verb W. E.g., snore → sleep.
    Verb-side relation, hypothesized to be sparse at L0-L12.
    """
    pairs, seen = [], set()
    for synset in wn.all_synsets("v"):
        a = _first_lemma(synset)
        if not a:
            continue
        for ent in synset.entailments():
            b = _first_lemma(ent)
            if b and a != b and (a, b) not in seen:
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_causes(limit):
    """Synset-level: verb V causes verb W. E.g., kill → die.
    Verb-side, hypothesized sparse at L0-L12.
    """
    pairs, seen = [], set()
    for synset in wn.all_synsets("v"):
        a = _first_lemma(synset)
        if not a:
            continue
        for cause in synset.causes():
            b = _first_lemma(cause)
            if b and a != b and (a, b) not in seen:
                pairs.append([a, b])
                seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


EXTRACTORS = {
    # Adjective-side (expected L0-L12 contributors)
    "pertainym":   extract_pertainyms,
    "similar_to":  extract_similar_tos,
    "attribute":   extract_attributes,
    "also_see":    extract_also_sees,
    # Verb-side (expected sparse at L0-L12; test of depth stratification)
    "entailment":  extract_entailments,
    "cause":       extract_causes,
}


def main():
    ensure_data()

    knowledge_dir = Path(__file__).parent.parent

    out_relations = {}
    per_rel_log = {}
    for rel_name, extractor in EXTRACTORS.items():
        print(f"  {rel_name}: extracting...", end=" ", flush=True)
        pairs = extractor(PAIRS_PER_RELATION)
        out_relations[rel_name] = {"pairs": pairs}
        per_rel_log[rel_name] = len(pairs)
        print(f"{len(pairs)} pairs")

    total_pairs = sum(len(d["pairs"]) for d in out_relations.values())
    total_subjects = len({p[0] for d in out_relations.values() for p in d["pairs"]})
    print(f"\nTotal: {total_pairs} pairs, {total_subjects} unique subjects")

    output_path = knowledge_dir / "data" / "wordnet_extended_relations.json"
    with open(output_path, "w") as f:
        json.dump(out_relations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {output_path}")

    provenance_path = knowledge_dir / "data" / "wordnet_extended_relations_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump({
            "pilot": "1c_relation_coverage",
            "design": "harvest 6 new WordNet relations not in canonical (pertainym, similar_to, attribute, also_see, entailment, cause)",
            "categories": {
                "adjective_side_likely_l0_l12": ["pertainym", "similar_to", "attribute", "also_see"],
                "verb_side_likely_l13_plus": ["entailment", "cause"],
            },
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
