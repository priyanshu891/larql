# ADR-0009 — Config Validation Does Not Require `head_dim | hidden_size`

**Status:** Implemented
**Area:** `larql-models::validation`

---

## Problem

`ModelArchitecture::validate` used to enforce:

> `head_dim` must divide `hidden_size`.

That invariant was applied both globally (`validate_hidden_head_dim`) and
per-layer (`validate_one_layer`). It caused every extraction level
(`browse` / `attention` / `inference` / `all`) of `google/gemma-3-1b-it`
to fail at the validator gate with:

```
config validation failed: [
  ConfigValidationError { field: "head_dim",
    message: "head_dim 256 must divide hidden_size 1152" },
  ConfigValidationError { field: "head_dim_for_layer",
    message: "layer 0 head_dim 256 must divide hidden_size 1152" }
]
```

before any tensors were touched.

---

## Why the invariant is wrong

In the attention block, the Q projection weight has shape
`[hidden_size, num_q_heads * head_dim]`. `hidden_size` is the model's
residual-stream width; `num_q_heads * head_dim` is the total Q output
width. They are **independent dimensions**. There is no math requirement
that `head_dim` divides `hidden_size`.

The rule only *happens* to hold for Llama / Mistral / Qwen because those
models pick `num_q_heads * head_dim == hidden_size` (square Q/K/V
projections), but that's a design convention, not a constraint imposed
by the transformer.

### Models that legitimately violate it

| Model                | hidden_size | head_dim | num_q_heads | Q-dim |
|----------------------|-------------|----------|-------------|-------|
| `gemma-3-1b-it`      | 1152        | 256      | 4           | 1024  |
| `gemma-3-270m`       | 1152        | 256      | 4           | 1024  |
| `gemma-4-*` sliding  | 1536        | 256      | 8           | 2048  |
| `gemma-4-*` global   | 1536        | 512      | 8           | 4096  |

All of these are valid configurations. `num_q_heads * head_dim` ≠
`hidden_size` by design.

DeepSeek MLA variants and some per-layer-heterogeneous architectures
have the same property.

---

## Decision

Drop `head_dim | hidden_size` from `validate_architecture`.

Removed:

- `validate_hidden_head_dim` (global)
- the matching branch inside `validate_one_layer` (per-layer)
- the now-unused `cfg` parameter threaded through `validate_one_layer`

Kept (all the real geometry constraints):

- `head_dim > 0`, `num_q_heads > 0`, `num_kv_heads > 0`
- `num_q_heads % num_kv_heads == 0` — GQA requirement
- Per-layer positivity of `head_dim` / `num_q_heads` / `num_kv_heads`
- Per-layer GQA divisibility
- `rotary_fraction ∈ (0, 1]`, `rope_base > 0`
- All rope-scaling, MoE, layer-types, shared-KV rules

---

## Where the real Q-projection shape check belongs

Config-only validation cannot catch a Q/K/V projection mismatch — you
need the safetensors tensor shapes to compare against. The correct
place is the safetensors-loading step
(`larql-models::loading::safetensors`), where `arch.num_q_heads_for_layer(l)
* arch.head_dim_for_layer(l)` can be verified against the actual
`q_proj.weight` tensor's trailing dim.

We did **not** add that check here because:

1. Tensor-shape mismatches already produce clear load-time errors
   downstream (safetensors deserialization, followed by kernel dim
   checks in `larql-inference`).
2. Adding a speculative shape check in the loader would duplicate
   existing error paths without improving diagnostics for the models
   that were actually failing.

If we later want a friendlier error at load time, the hook is
`safetensors.rs:~146` where `detect_architecture_validated` is called.

---

## Tests updated

`larql-models/tests/test_architectures.rs::validation_rejects_invalid_attention_geometry`
used a config with `hidden_size=4100, head_dim=128, num_q_heads=10,
num_kv_heads=3` and asserted both `FIELD_HEAD_DIM` (from the removed
rule) and `FIELD_NUM_Q_HEADS` (from GQA). After the fix only
`FIELD_NUM_Q_HEADS` is expected — and that's the error that actually
matters: `10 % 3 ≠ 0`.

---

## Verification

- `cargo build -p larql-models` — clean
- `cargo test -p larql-models --test test_architectures validation` —
  all 9 validation tests pass
- `google/gemma-3-1b-it` extraction now reaches tensor streaming at
  every level

---

## Don't re-add this rule

If you find yourself tempted to re-introduce a "head_dim must divide
hidden_size" check:

- Gemma 3 and Gemma 4 break it.
- The Q projection's output dim is `num_heads * head_dim`, which is
  independent of `hidden_size`.
- If the real concern is "the Q projection tensor is the right shape,"
  write that as a tensor-shape check in the safetensors loader against
  `num_q_heads_for_layer(l) * head_dim_for_layer(l)`, not a config
  invariant.
