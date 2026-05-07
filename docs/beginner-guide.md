# LARQL for beginners

A plain-English walkthrough of what LARQL does and how. Zero ML background assumed.

---

## The one-sentence version

A trained language model has knowledge baked into its weights. **LARQL reads that knowledge out as a searchable graph** — without ever running the model.

---

## The flashcard picture

Forget matrices for a moment. Imagine the trained model is a deck of **350,000 flashcards** (for a 4B-parameter model like Gemma-3). Each flashcard has two sides:

```
  ┌─────────────────────────┐
  │  FRONT:                 │
  │  "Looks like France"    │  ← when to fire
  │                         │
  │  BACK:                  │
  │  "Say Paris"            │  ← what to do
  └─────────────────────────┘
```

Each card is called a **feature**. The deck is split into 34 piles, called **layers**.

When the model runs normally, it:
1. Reads your input.
2. For every card, checks "does the front match?"
3. If yes, adds what's on the back to its running answer.
4. After going through all 350,000 cards, reads out the final answer.

---

## The graph picture

Three kinds of nodes:

```
     ◯ Words you type         "France", "Einstein", "pizza"
                              (on the left)

     ◉ Features (neurons)     350,000 of them
                              (in the middle)

     ● Words in vocabulary    "Paris", "the", "Tokyo"
                              (on the right)
```

And this is the graph:

```
     WORDS YOU TYPE              FEATURES                    VOCABULARY
     (left side)                 (middle)                    (right side)


                                    ◉ (L0,F0)
                                    ◉ (L0,F1)
                                      :

        ◯ "France"  ─────────────►  ◉ (L27,F8821) ───────►  ● "Paris"
             │      \                    │
             │       \                   │
             │        ────────────────►  ◉ (L24,F3102) ──►  ● "French"
             │
             └─────────────────────►  ◉ (L25,F7711) ──────►  ● "Europe"


        ◯ "Einstein" ────────────►  ◉ (L26,F5512) ───────►  ● "physics"
                 \
                  ───────────────►  ◉ (L27,F9001) ───────►  ● "relativity"
                                      :
```

### Two kinds of edges

| | Left edges (word → feature) | Right edges (feature → word) |
|---|---|---|
| **Connects** | your query → feature | feature → vocab word |
| **Example** | "France" → (L27, F8821) | (L27, F8821) → "Paris" |
| **When drawn** | every time you ask | once, at extraction |
| **How many per query** | top 5 per layer = up to ~170 | each feature has exactly 1 |
| **Has a score?** | yes (how strongly it matches) | no (it's a label) |
| **Stored on disk?** | no, recomputed live | yes, precomputed |

---

## Where are the cards actually stored?

Inside two big spreadsheets (matrices) in the model:

- **Matrix A (`W_gate`)** — every **row** is the *front* of one card
- **Matrix B (`W_down`)** — every **column** is the *back* of one card

```
   Matrix A (fronts)              Matrix B (backs)
   ─────────────────              ────────────────
   row 0:    "Looks like cat"     col 0:    "Say meow"
   row 1:    "Looks like dog"     col 1:    "Say woof"
   row 2:    "Looks like France"  col 2:    "Say Paris"
   row 3:    "Looks like Japan"   col 3:    "Say Tokyo"
   ...                            ...
```

Row 2 of Matrix A and column 2 of Matrix B belong to the **same card** (card #2). That's the feature.

Each row and column is just a list of 2560 numbers (for Gemma 4B) — a direction in the model's internal "concept space".

---

## What LARQL actually does — extraction

Two steps.

### Step 1 — Slice up the matrices

For every layer `L` and every feature `i`:

```
  gate_vec    =  row i of Matrix A       (the "front")
  down_vec    =  column i of Matrix B    (the "back")
```

Save them to disk laid out feature-first so each card's two vectors are contiguous chunks. That's `gate_vectors.bin` and `down_features.bin` in the vindex folder.

No transformation, no training — **just slicing**.

### Step 2 — Translate each card's back into a word

The back of a card (`down_vec`) is a list of 2560 numbers — a direction, not a word. But the model ships with a **dictionary** (`W_E`, the unembedding matrix) that converts any direction into a word.

For each feature, we run one small matrix-vector multiply:

```
  down_vec  ──(multiply by W_E)──►  list of scores for every vocab word
                                    ↓
                                    argmax → top word ("Paris")
                                    keep top-k → ("Paris", "Parisian", ...)
```

Save the result as `FeatureMeta { top_token: "Paris", c_score: 89.3, ... }` for every feature.

That's it. Extraction takes ~5-10 minutes on a laptop. No GPU. No forward pass.

---

## What happens when you ask `DESCRIBE "France"`

```
   Step 1: turn "France" into a direction
   ────────────────────────────────────────
   "France"  ──tokenize──►  token IDs  ──look up embeddings──►  q (2560 numbers)


   Step 2: for every layer, find cards whose front matches q
   ──────────────────────────────────────────────────────────
   For each layer L:
       scores = gate_vectors_L · q       (one dot product per feature)
       take top 5 scores


   Step 3: look up each matching card's back
   ──────────────────────────────────────────
   (L27, F8821)  →  read FeatureMeta from disk  →  "Paris"
   (L24, F3102)  →  read FeatureMeta from disk  →  "French"
   (L25, F7711)  →  read FeatureMeta from disk  →  "Europe"


   Step 4: chain and sort
   ──────────────────────
   "France" ──► "Paris"    (via L27, score 1436)
   "France" ──► "French"   (via L24, score 35)
   "France" ──► "Europe"   (via L25, score 14)
```

That's the output. A little graph walk: **word → feature → word**.

---

## The mental model in one line

> **When you search for a word, the system goes through every layer of the model and finds the features whose "front" matches that word. For each match, it reads the "back" and reports what the feature outputs.**

Or, reframed as graph:

> **The trained model is secretly a bipartite graph: your queries on the left, 350,000 features in the middle, vocabulary words on the right. LARQL precomputes the feature → word edges once, and at query time it draws the query → feature edges on the fly. A DESCRIBE is just a two-hop walk through that graph.**

---

## Why it's called "decompilation"

Software decompilation takes a compiled `.exe` and recovers source code.

Training a model is like compiling: lots of text gets squashed into weights.

LARQL does the reverse — it recovers facts (source) from the weights (compiled output) — by **noticing that the matrices were already organized as flashcards and just slicing them up**.

The model **is** the database; LARQL just makes it readable.

---

## Three extract levels

Same graph either way — what differs is what you can *do* with it.

| Level | Size (4B model) | What you can do |
|---|---|---|
| `browse` | ~3 GB | `DESCRIBE`, `WALK`, `SELECT` — pure graph queries |
| `inference` | ~8 GB | + `INFER` (real forward pass through the model) |
| `all` | larger | + `COMPILE` (stamp new edges back into the weights) |

---

## Relation labels (optional)

Sometimes edges have a label like `capital → Paris` instead of just `→ Paris`. Where does "capital" come from?

It's optional extra step called **probing**. We run the model on test sentences like "The capital of X is Y" and see which features consistently fire. Those features get tagged as "capital" features in `feature_labels.json`.

Without probing, the graph still works — you just don't get the relation labels.

---

## File map (where to find things in this repo)

| Concept | File |
|---|---|
| `FeatureMeta` type (the per-card data) | [crates/larql-vindex/src/index/types.rs](../crates/larql-vindex/src/index/types.rs) |
| `gate_knn` (find cards whose front matches) | [crates/larql-vindex/src/index/compute/gate_knn/dispatch.rs](../crates/larql-vindex/src/index/compute/gate_knn/dispatch.rs) |
| `DESCRIBE` handler | [crates/larql-server/src/routes/lql.rs](../crates/larql-server/src/routes/lql.rs) |
| Extraction pipeline | [crates/larql-vindex/src/extract/streaming.rs](../crates/larql-vindex/src/extract/streaming.rs) |
| Circuit types (identity / transform / projector / …) | [docs/circuit-types.md](circuit-types.md) |
| Advanced hooks for real forward passes | [docs/mech-interp.md](mech-interp.md) |

---

## Try it

```bash
# start the server
cargo run --release -p larql-server -- models/gemma3-1b.vindex

# ask it about France
curl 'http://localhost:8080/v1/describe?entity=France'

# or run an LQL statement
curl -X POST http://localhost:8080/v1/lql \
  -H 'content-type: application/json' \
  -d '{"statement":"DESCRIBE \"France\""}'
```

You'll see the two-hop graph walk happen in real time.
