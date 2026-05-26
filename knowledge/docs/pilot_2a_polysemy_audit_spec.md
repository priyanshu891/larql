# Pilot 2a — Polysemy Audit Spec

**Status:** spec, not yet implemented. To be drafted as `knowledge/scripts/pilot_2a_polysemy_audit.py` and run on the existing labeled inventory before Pilot 1c launches.

**Tests:** [P4 in META_MODEL.md](../../META_MODEL.md) — polysemy and promiscuity rates in the 129-label inventory.

**Substrate role:** the audit produces the **trust filter** for downstream work. Any analysis after this audit that quotes the labeled inventory should use the post-audit stable count, not the raw 129. Pilot 1c references this audit as a dependency, not a co-equal experiment.

---

## What the audit does

Read-only static inspection of `down_meta` for every currently-labeled feature in the wn:* inventory. No model inference. The audit classifies each labeled feature into one of three buckets:

- **Mono-semantic** — `down_meta` real-word tokens cluster around a single semantic group. The label reflects a genuine feature with coherent output structure.
- **Polysemantic** — `down_meta` real-word tokens cluster into 2+ distinct semantic groups, each supported by entity content from the labeling probe. The feature is doing more than one job. Multi-label output recommended.
- **Promiscuous** — `down_meta` is flat-distributed noise with few real words. The ≥2-hits threshold caught the label because sampling happened to land on matching content; the feature does not have coherent semantic structure. **Demote** — should not be counted in the labeled inventory.

The audit also reports the **comparability count** (current ≥2-hits inventory, unchanged) alongside the **stable count** (mono-semantic + polysemantic, with promiscuous removed). Per M3 in META_MODEL.md, downstream analysis uses the stable count as load-bearing.

---

## Input

Existing artifacts at `larql/output/gemma3-4b-v2.vindex/`:
- `feature_labels.json` — canonical 64 wn:* labels (plus 221 other-labeled + 1500 knowledge labels we don't audit here)
- `feature_labels_multilingual_pilot_rich.json` — 30 multilingual labels with entity/output/relation detail
- `feature_labels_subword_pilot_rich.json` — 52 subword labels with detail
- `down_meta.bin` — token outputs per (layer, feature)
- (after 1c) `feature_labels_extended_pilot_rich.json` — adds 1c labels

Cumulative wn:* set after deduplication: 129 features pre-1c. Audit runs on the dedup'd union.

## Output

`larql/output/gemma3-4b-v2.vindex/feature_labels_polysemy_audit.json`:

```json
{
  "L9_F7535": {
    "classification": "promiscuous",
    "real_word_ratio": 0.2,
    "mean_token_length": 4.2,
    "bimodality_score": 0.0,
    "top_real_words": ["grueling", "man"],
    "down_meta_full": ["nonatomic", "grueling", "ღვ", "mathrm", "পড়", "ждены", "말", "man", "저는", "ngu"],
    "labels_from": {"multilingual": "wn:synonym", "subword": "wn:synonym"},
    "reason": "real_word_ratio 0.2 below cutoff; flat token distribution without coherent semantic cluster"
  },
  "L8_F8974": {
    "classification": "mono_semantic",
    "real_word_ratio": 0.8,
    "mean_token_length": 9.1,
    "bimodality_score": 0.0,
    "top_real_words": ["excepcional", "genial", "championnat", "azienda", ...],
    ...
  },
  ...
}
```

Plus a summary JSON `feature_labels_polysemy_audit_summary.json` with:

```json
{
  "total_audited": 129,
  "mono_semantic_count": N,
  "polysemantic_count": M,
  "promiscuous_count": K,
  "stable_count": N + M,
  "classification_distribution_by_relation": {...},
  "classification_distribution_by_layer": {...},
  "cutoffs_used": {
    "real_word_ratio_cutoff": <set post-hoc>,
    "mean_length_cutoff": <set post-hoc>,
    "bimodality_threshold": <set post-hoc>
  },
  "anchor_check": {
    "L9_F7535_classified_as": "promiscuous (REQUIRED)",
    "L8_F8974_classified_as": "mono_semantic (REQUIRED)",
    "L0_F5560_classified_as": "mono_semantic (REQUIRED)",
    "L12_F5382_classified_as": "mono_semantic (REQUIRED)"
  },
  "tests_prediction": "P4 in META_MODEL.md",
  "P4_result": {
    "predicted_mono_semantic_pct": "80-85",
    "predicted_polysemantic_pct": "<5",
    "predicted_promiscuous_pct": "10-15",
    "observed_mono_semantic_pct": ...,
    "observed_polysemantic_pct": ...,
    "observed_promiscuous_pct": ...,
    "outcome": "confirmed | refuted-polysemy-high | refuted-promiscuity-high | partial"
  }
}
```

---

## Metrics computed per feature

For each labeled feature, compute from `down_meta` (top 10-30 tokens depending on vindex storage):

### 1. Real-word ratio

`real_word_count / total_token_count`, where a token counts as "real word" if:
- Length ≥ 3 after stripping leading `▁` (SentencePiece marker)
- All characters alphabetic (no digits, no punctuation, no script-only fragments)
- Not a known programming/markup token (loose blacklist: `mathrm`, `marginLeft`, `nonatomic`, `eqref`, etc. — initial list seeded from observed noise in L9_F7535)

The blacklist is a minor refinement; the alpha+length filter does most of the work. Recorded in output.

Anchor values: L9_F7535 = 0.2, L8_F8974 = 0.8, L0_F5560 = 0.2 (low — see note below), L12_F5382 = 0.7.

**Note on L0_F5560:** the static check showed L0_F5560's down_meta is heavily punctuation (5/9 punct, 2/9 Latin) — only 2 real words including "Class". This is a tension with classifying it as mono-semantic. Two readings:
- The label is correct (the feature fires on biological-taxa entities → "class") but the down_meta is structurally punctuation-heavy, meaning real-word ratio alone won't classify L0 features well.
- L0_F5560 is borderline and should land in polysemantic or be flagged as "low-confidence mono-semantic."

This is why cutoffs are set post-hoc with anchors. The anchor constraint "L0_F5560 = mono_semantic" forces the audit to find a metric combination that captures it. Likely: real-word ratio alone is insufficient; mean-length OR semantic-coherence-among-real-words is needed alongside it.

### 2. Mean token length

`sum(stripped_token_lengths) / total_token_count`.

Anchor values: L9_F7535 = 4.2, L8_F8974 = 9.1, L0_F5560 ≈ 3.9 (low), L12_F5382 = 4.4.

Same tension as above: L0_F5560 and L12_F5382 have low mean length despite being clean labels. This suggests mean length alone discriminates *some* feature types (synonym/exception clusters with long words) but not others (hypernym-of-class clusters with short categorical words).

### 3. Real-word semantic coherence

Among the real-word subset of `down_meta`, compute pairwise similarity. Two approximations available:

- **String-based:** mean pairwise Jaccard over character bigrams. Cheap, no embeddings needed. Captures morphological similarity ("differentiates" / "differentiated") and shared roots.
- **Embedding-based:** if the model is loaded, compute mean pairwise cosine similarity of token embeddings. More accurate for cross-lingual coherence (исключением / excepcional / genial).

Recommendation: ship with string-based first; add embedding-based as a refinement if string-based produces too many borderline cases.

Mean similarity high → mono-semantic. Mean similarity low → either promiscuous (no structure) or polysemantic (two coherent groups, low mean across).

### 4. Bimodality score

To distinguish promiscuous from polysemantic when real-word ratio is moderate: cluster the real-word subset by similarity, check whether 2 clusters explain the variance better than 1.

Cheap version: hierarchical clustering with average-linkage on the Jaccard distances, cut at k=2, compute silhouette. Silhouette > 0.3 with both clusters having ≥2 members → bimodal. Silhouette < 0.1 → unimodal (promiscuous if low real-word ratio, mono-semantic if high).

Polysemantic anchors aren't available in the spot-check (we found 1 confirmed polysemantic candidate in 1500+ features, and that candidate is debatable). The audit may find zero polysemantic features in the labeled inventory — which would itself be a result (consistent with P4's <5% prediction).

---

## Cutoff procedure (the procedural commitment)

Per the n=2 problem flagged in spec discussion: do NOT hardcode cutoffs from the L9 vs L8 result. The procedure is:

1. Compute the three metrics on every feature in the audit set (129 features).
2. Plot the empirical distribution of each metric. Look for natural inflection points.
3. Set cutoffs to maximize agreement with the **anchor constraint**:
   - L9_F7535 MUST land in promiscuous
   - L8_F8974, L0_F5560, L12_F5382 MUST land in mono-semantic
   - The L0_F5560 case forces the cutoffs to allow short, punctuation-heavy down_meta to be mono-semantic — i.e. real-word ratio cannot be the sole discriminator
4. Record the chosen cutoffs in the output summary along with the distribution and the anchor check.
5. If no cutoff combination satisfies all four anchors, the metrics are insufficient; audit produces a "metrics-insufficient" classification and reports the conflict rather than forcing a choice.

This is the falsifiable version: either the metrics support the anchor-validated classification, or they don't, and the audit reports honestly.

---

## Classification logic

Pseudocode:

```
for feat in audit_set:
    rwr = real_word_ratio(feat.down_meta)
    mtl = mean_token_length(feat.down_meta)
    sim = mean_pairwise_jaccard(real_words(feat.down_meta))
    bim = bimodality_score(real_words(feat.down_meta))

    if rwr < CUTOFF_RWR and mtl < CUTOFF_MTL and sim < CUTOFF_SIM:
        classification = "promiscuous"
    elif bim > CUTOFF_BIMODALITY and num_real_words >= 4:
        classification = "polysemantic"
    elif rwr >= CUTOFF_RWR_LOW or sim >= CUTOFF_SIM_HIGH:
        # captures short-word mono-semantic clusters like L0_F5560
        classification = "mono_semantic"
    else:
        classification = "borderline"
```

`borderline` is a separate bucket — features where the metrics don't decisively place them. Report count and sample. If >10% borderline, the metrics need refinement before the audit can be load-bearing.

---

## Connection to 1c

The 1c probe runs the same polysemy classification inline on its new candidate features (the relation-coverage probe doesn't have to wait for a separate audit pass). 1c uses the cutoffs set by this audit on the existing 129 — so the audit must run before 1c launches. If 1c discovers a candidate that fails the polysemy check, it is reported but not counted toward the new-vs-cumulative number.

---

## Runtime estimate

~minutes, not hours. Read-only on a 348K-feature vindex; only 129 features need full inspection. Token similarity is O(n²) per feature but n ≤ 30 tokens.

---

## What the result tells us (per P4)

- **Confirmed (mono ~80%, polysemy <5%, promiscuous 10-15%):** working model holds. Stable count ~115-120. Multi-label features ~5. Promiscuous demotion ~13-19 features.
- **Refuted-polysemy-high (polysemy >20%):** the labeled inventory is much messier than pilot quality metrics suggested. Many of the "129 labels" represent features doing multiple jobs. Major re-read of the multilingual + subword findings; cross-lingual rate (M1) may be inflated by polysemy artifacts.
- **Refuted-promiscuity-high (promiscuous >30%):** the ≥2-hits threshold is broadly too permissive. The cumulative trajectory should be reconsidered using tighter thresholds; possibly the per-axis ceiling estimate (25-45 new) was inflated. Re-derive the meta-model.

The audit *cannot* fix the methodology — it's a diagnostic. But the result determines whether the working model survives in its current form or needs revision before 1c is interpretable.
