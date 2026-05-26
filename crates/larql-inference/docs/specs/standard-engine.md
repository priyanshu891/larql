# StandardEngine — Specification

**Status:** ✅ Shipped. The default `--engine standard`; the reference
against which every other engine's "exact" claim is measured.
**Audience:** LARQL contributors.
**Scope:** Contract for the production K/V cache engine in `larql-kv`.

---

## 1. Purpose

`StandardEngine` is the engine your code is using if you don't think
about engines. It wraps the production K/V cache that `larql bench`
and `larql run` have been using since the project started — same
forward pass, same accuracy, same speed.

It exists as an engine (rather than just "the way decode works") so
that the alternate engines (`markov_residual`, `turbo_quant`, etc.)
can be slotted in via the same `KvEngine` trait, and so that A/B
parity tests have a precise reference target. Its state policy is
"the backend owns the K/V cache; we hold an opaque `KvHandle`."

`StandardEngine` also implements the **fused fast path** through the
backend's `coarse_prefill` / `coarse_decode_step` intent surface:
on Metal this routes to the production multi-layer kernel that
submits one command buffer per token. As of the 2026-05-17 cut, it is
the **only** engine that takes this path — the per-layer engines used
to silently piggyback on it via hidden `fused_prefill` short-circuits
(see ROADMAP §"Closed (recent)" for the bypass-strip).

---

## 2. Contract

### 2.1 Correctness contract

> For any prompt `P` and any decode step `t`, the next-token
> distribution produced by `StandardEngine` is the **reference
> distribution** for the model — bit-identical to running the model
> through `predict_with_ffn` on the same `(prompt, sampling_config)`.

Every other engine's "exact under contract" claim is measured
against `StandardEngine`. There is no upstream reference for
`StandardEngine` itself; it IS the reference.

### 2.2 State-sufficiency contract

> Persistent state = the per-layer K/V tensors covering all tokens
> in the current attention window. State is owned by the
> `ComputeBackend` (`MetalBackend` keeps it in `MetalKvCache` inside
> a mutex; `CpuBackend` keeps it in `CpuKvCache`). The engine holds
> an opaque `KvHandle` returned from `coarse_prefill` that engines
> must treat as a token, not as a state container.

### 2.3 Memory contract

> Hot state: `O(num_layers × seq_len × kv_dim × sizeof(f32))` when
> `window = None`; `O(num_layers × window × kv_dim × sizeof(f32))`
> when `window = Some(N)`. The engine itself adds zero accounting
> overhead — `engine.memory_bytes()` reports the backend's K/V
> bytes via `KvHandle::cached_len()`, not a duplicate count.

For Gemma 3 4B at 1k tokens unwindowed: ≈ 0 MB engine-side, full
K/V volume backend-side.

### 2.4 Sliding-window variant

`Standard { window_size: Some(N) }` clips the K/V cache to the most
recent N tokens after each decode step. This is the production
"bounded-memory" path; CLI flag `--kv-cache markov-bounded
--context-window N` maps to this, **not** to `MarkovResidual` —
naming history that pre-dates the residual-stream engine. The
sliding window keeps the exact-reference contract for tokens
within the window; tokens beyond the window are dropped (not
preserved at a lower fidelity).

---

## 3. Implementation

| Concern | Location |
|---|---|
| Engine type + trait impl | `crates/larql-kv/src/engines/standard.rs` |
| Trait surface (`KvDispatch::coarse_prefill` etc.) | `crates/larql-inference/src/kv_dispatch/mod.rs` |
| Metal fused kernel | `crates/larql-compute-metal/src/decode/mod.rs::decode_token_with_moe_split_fn` |
| Sliding-window clip | `larql_kv::engines::standard::StandardEngine::do_prefill` + `do_decode_step` |

The engine intentionally has minimal code — most of the work is on
the backend side. `do_prefill` / `do_decode_step` route through
either `kv_prefill_via_dispatch` (CPU) or `MetalBackend::
coarse_prefill` (Metal); the engine maintains only the `KvHandle`
across steps.

---

## 4. Performance (Gemma 3 4B Q4K, M3 Max, 2026-05-17)

| Backend | Decode (tok/s) | Per-step latency |
|---|---:|---:|
| Metal (fused fast path) | **105.9** | 9.4 ms |
| CPU (BLAS + C Q4 kernel) | 28.2 | 35 ms |

These are the **ceiling numbers** for the model on this machine.
Every other engine compares against these.

Engine memory accounting: `hot=0.0MB cold=0.0MB` always — the engine
doesn't store anything; the backend's internal cache does. Use
`larql_kv::cache::KvCache` directly if you need to inspect cache
state.

---

## 5. Non-goals

- **K/V eviction beyond sliding-window.** If you need
  per-chunk-frame eviction with replay, use `unlimited_context`. If
  you need codec-encoded eviction, use `markov_residual_codec`.
  `StandardEngine` either keeps everything or keeps the last N
  tokens — no per-token policy.
- **Cross-session resume.** That's `boundary_kv` (which composes
  with `StandardEngine` + emits `larql-boundary` chunk frames).
- **Compression.** That's `turbo_quant` (in-place K/V compression)
  or `markov_residual_codec` (compressed cold tier).
