# NoCacheEngine — Specification

**Status:** ✅ Shipped. Debug / correctness fallback only.
**Audience:** LARQL contributors.

---

## 1. Purpose

`NoCacheEngine` is what an engine looks like when you remove every
optimization. Each decode step re-runs the entire prefill over the
growing token sequence — `[prompt + generated_so_far + new_token]`
— and discards the K/V cache at the end of the step. O(N²) total
wall-time over a full generation.

It exists for two reasons:

1. **Correctness baseline.** If a new engine claims "exact under
   contract" against `StandardEngine`, you can also compare it
   against `NoCacheEngine` for additional confidence — the two
   paths share no state-management code, so a parity match against
   both is a strong signal.
2. **Debugging.** When K/V cache corruption is suspected, switching
   to `--engine no-cache` removes the cache from the equation.

`NoCacheEngine` is not an optimization target. The roadmap explicitly
lists it as out of scope for performance work.

---

## 2. Contract

### 2.1 Correctness contract

> Bit-identical to `StandardEngine` on the same `(prompt,
> sampling_config)`. Both engines call the same `kv_prefill_run`
> helper under the hood — `NoCacheEngine` just calls it every step
> over `[tokens ++ [new]]` instead of incrementally.

### 2.2 Memory contract

> Persistent state: `O(generated_so_far)` for the token list. No
> K/V tensors. `engine.memory_bytes() = tokens.len() *
> sizeof(u32)`.

### 2.3 Compute contract (NOT a complexity guarantee)

> Per decode step at position N: full forward pass over N+1 tokens.
> Total generation cost: O(prompt_len × decode_steps +
> decode_steps²).

The quadratic term is the whole point — this is what `StandardEngine`
amortises away with its incremental K/V update.

---

## 3. Implementation

| Concern | Location |
|---|---|
| Engine | `crates/larql-kv/src/engines/no_cache.rs` |
| `prefill` / `decode_step` | calls `kv_prefill_run` from `larql_kv::generation` |
| `prefill_quant` / `decode_step_quant` | calls the same after dequantising attn tensors + building a `WalkFfn` |
| W1-GPU `*_via_executor` overrides | inherit the executor's FFN dispatcher; no separate state-policy |

The engine has no state struct of its own beyond `tokens: Vec<u32>`
+ the backend handle.

---

## 4. Performance

Per-step time = full forward over the growing context. On Gemma 3 4B
this is ~28 ms/step at the empty-context start, scaling linearly with
generated length. Don't benchmark this against the other engines — the
shape is fundamentally different (O(N) per step vs O(1)).

---

## 5. Non-goals

- **Performance.** Not on the optimization roadmap. The engine's
  identity is "no cache." Any optimization that adds one
  contradicts the spec.
- **Sliding window.** `--engine no-cache` always re-runs the full
  context. For bounded re-forward windows, use
  `Standard { window_size: Some(N) }`.
