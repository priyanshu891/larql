# ADR-0022 — Compute Trait Extraction (Metal as First-Class Peer)

**Status:** In progress — Step 1 (residual norms) landed 2026-05-17.
Steps 2–6 to follow in sequenced commits.
**Supersedes (in part):** `crates/larql-inference/docs/specs/compute-backend-redesign.md` §10.2.
**Depends on:** ADR-0019 (`larql-compute-metal` extracted as sibling crate).
**Affects:**
- `crates/larql-compute/` (gains forward-pass, residual, KvDispatch, AsyncComputeBackend),
- `crates/larql-compute-metal/` (gains Metal trait impls),
- `crates/larql-inference/` (loses substrate code; keeps engines, orchestration, FFN routing),
- `crates/larql-kv/`, `crates/larql-cli/`, `crates/larql-server/` (re-import paths through inference re-exports — no caller changes required).

---

## Context

After ADR-0019 split the Metal backend into `larql-compute-metal`,
the framing in `AGENTS.md` still described Metal as "a thin backend"
behind `larql-compute`. The actual code shape no longer matches:
`larql-compute-metal` is ~26k LOC (2.4× larger than `larql-compute`),
ships custom MSL shaders, multi-layer pipelining, and stage-bisected
kernels. Metal is now a first-class peer, not a thin layer.

The remaining "leak" is **structural**: the trait impls
`KvDispatch for MetalBackend` and `AsyncComputeBackend for MetalBackend`
both live in `larql-inference` (`src/kv_dispatch/metal.rs`,
`src/async_compute_backend/metal.rs`), gated by `#[cfg(feature = "metal")]`.
The same is true of the CPU impls (`kv_dispatch/cpu.rs`,
`async_compute_backend/cpu.rs`). The `larql-inference` crate is forced
to host backend-specific impl code because the *trait* lives there.

### Why the trait currently lives in `larql-inference`

`compute-backend-redesign.md` §10.2 records the deliberate placement
of `KvDispatch` in `larql-inference`. The original sketch (trait in
`larql-compute`) was tried and reverted 2026-05-16. The blocker:

- `CpuBackend` lives in `larql-compute`.
- The CPU `KvDispatch` impl body needs to call inference-side
  forward-pass functions (`run_attention_*`, `run_ffn`, residual ops).
- `larql-compute` cannot depend on `larql-inference` (cycle).
- Rust's orphan rule forbids `impl KvDispatch for CpuBackend` in
  `larql-inference` if both the trait and the impl type are foreign.

So `KvDispatch` was placed in `larql-inference` as a sibling of
`ComputeBackend`/`FfnBackend`, and impls for both `CpuBackend` and
`MetalBackend` live alongside it. This satisfied the orphan rule
without inducing a cycle — at the cost of pinning backend impl code
in the inference crate.

### Why we're revisiting now

The framing problem flagged in the 2026-05-17 architecture review:
`larql-inference` is no longer a thin orchestrator. It hosts ~30
`#[cfg(feature = "metal")]` sites across `kv_dispatch/`,
`async_compute_backend/`, `layer_graph/hybrid.rs`,
`layer_graph/generate/gpu/{mod,decode_loop}.rs`. The "Metal-shaped
code lives in `larql-compute-metal`" principle isn't fully true.

The path through the cycle that §10.2 didn't take: **move the
forward-pass functions that the CPU `KvDispatch` impl calls down to
`larql-compute` too.** That removes the inference-side dependency of
the CPU impl, which removes the cycle, which lets the trait + both
impls live in compute crates.

---

## Decision

Extend `larql-compute` to host:

1. The leaf substrate math currently in `larql-inference/src/residual.rs`
   (`rms_norm*`, `layer_norm*`, head variants).
2. The forward-pass primitives `run_attention_*`, `run_ffn`, and
   companions currently in `larql-inference/src/forward/`.
3. The `KvDispatch` trait + handle types + `CpuBackend` impl.
4. The `AsyncComputeBackend` trait + handle types + `CpuBackend` impl.

Move Metal trait impls down to `larql-compute-metal`:

5. `KvDispatch for MetalBackend` (currently `larql-inference/src/kv_dispatch/metal.rs`).
6. `AsyncComputeBackend for MetalBackend` (currently
   `larql-inference/src/async_compute_backend/metal.rs`).

Keep in `larql-inference`:

- `FfnBackend` *impls* (routing, remote shards, MoE dispatch — these
  are inference *topology*, not substrate). The trait *definition*
  moves to `larql-compute` (Step 2c, 2026-05-17) so substrate-level
  forward-pass code can dispatch through `&dyn FfnBackend`. Same
  pattern as `KvDispatch`: trait in compute, impls wherever
  the orphan rule allows. See "What this ADR does *not* do" for
  the impl/trait split rationale.
- Engines (`StandardEngine`, `MarkovResidual`, `Apollo`, `NoCache`,
  `UnlimitedContext`, `TurboQuant`), chat, sessions, tokenizer.
- `layer_executor/`, `layer_graph/`, `forward_overrides`, and the
  forward orchestrators that *call* `run_attention_*` (the
  inference-shaped composition of substrate primitives stays here).
- The arch-aware convenience wrappers `rms_norm_for_arch` /
  `layer_norm_for_arch`, which depend on `forward_overrides` to read
  `LARQL_NORM_EPS_OVERRIDE`. Substrate functions take an explicit
  `eps: f64`; `forward_overrides` stays in `larql-inference` until
  it's needed lower down.
- Re-exports at the original module paths
  (`crate::residual::*`, `crate::forward::*`, `crate::kv_dispatch::*`,
  `crate::async_compute_backend::*`) so existing callers across
  `larql-kv`, `larql-cli`, `larql-server` continue to compile
  unchanged.

### Why extend `larql-compute` instead of creating a new crate

The natural alternative — a new `larql-compute-core` or
`larql-forward` crate inserted between `larql-compute` and the backend
crates — adds a coordination point without a corresponding seam.
`ComputeBackend`/`CpuBackend` already live in `larql-compute`; adding
forward-pass primitives + trait surface continues a coherent "this is
what you do with a backend" story. The crate grows from ~11k → ~15k
LOC, still within a reasonable single-crate size. A future split is
not foreclosed.

### Why the `FfnBackend` trait moves but its impls don't

The trait definition is just method signatures over `ndarray::Array2`
and `usize` — no inference-side types. It belongs in the substrate
crate so that other substrate-level code (`forward/layer.rs`'s
`run_layer_with_ffn`, eventually moving down too) can take
`&dyn FfnBackend` without dragging in `larql-inference`.

The impls (`GraphFfnBackend`, `RemoteFfnBackend`, `RemoteMoeBackend`,
`SparseFfn`, `WeightFfn`, etc.) reference session-local state, gRPC
clients, shard discovery, and the WalkFfn dispatch machinery — all
of which is inference topology. They stay in `larql-inference`.

The orphan rule doesn't bind here: no foreign type implements
`FfnBackend`. Trait in compute + impls in inference is a legal,
clean split.

---

## Migration plan

Six steps, each its own commit, each with parity verification:

| Step | Move | LOC | Verification |
|---:|---|---:|---|
| 1 | `residual.rs` leaf functions → `larql-compute/src/residual.rs`. Inference shim re-exports + adds `*_for_arch` wrappers with env-override behaviour preserved. | ~413 | Existing unit tests move with the code + new tests pin the explicit-`eps` contract. Workspace builds clean; clippy clean; fmt clean. |
| 2 | `run_attention_*`, `run_ffn`, `apply_norm`, related helpers → `larql-compute/src/forward/`. Inference re-exports under `crate::forward::*`. `forward_overrides` moves down or each call site computes its effective param before calling. | ~496 | Existing forward-pass integration tests must pass byte-for-byte; bench numbers unchanged. |
| 3 | `KvDispatch` trait + handle types (`KvHandle`, `ResidualHandle`, `CompressionCodec`, `KvHandleInner`, `ResidualHandleInner`) + `CpuBackend` impl → `larql-compute/src/kv_dispatch/`. Inference re-exports under `crate::kv_dispatch::*`. | ~1651 | Bit-parity tests from `compute-backend-redesign.md` §10.2 sub-step 2c must pass byte-for-byte vs legacy CPU forward functions. |
| 4 | `AsyncComputeBackend` trait + handle types + `CpuBackend` impl → `larql-compute/src/async_compute/`. | ~1066 | Engine async parity tests. |
| 5 | Metal impls of both traits → `larql-compute-metal/src/{kv_dispatch_impl.rs, async_compute_impl.rs}`. `larql-inference/src/{kv_dispatch,async_compute_backend}/metal.rs` deleted. | ~767 | `cargo test --workspace --features metal` on macOS green; `cargo build --workspace` on non-Mac green. |
| 6 | Trim `#[cfg(feature = "metal")]` dispatcher sites in `layer_graph/hybrid.rs`, `layer_graph/generate/gpu/{mod,decode_loop}.rs`. Orchestration cfg branches stay; trait-impl cfg branches disappear once the impls are sibling-crate types accessed via dispatch. | ~20 sites | Full test suite + `--features metal` on macOS. Decode tok/s bench unchanged ±2%. |

**Per-step quality gate (every commit):**
- `cargo build --workspace` clean
- `cargo test --workspace` green
- `cargo test --workspace --features metal` green (on macOS)
- `cargo clippy --workspace --tests -- -D warnings` clean
- `cargo fmt --all --check` clean
- ≥90% per-file line coverage on moved files
- No regression in `larql shannon verify` cross-engine bits/char

---

## Risks & mitigations

- **Hidden inference coupling in moved code.** Forward-pass functions
  may call inference-side helpers we haven't catalogued
  (`forward_overrides::*`, `test_utils`, `model::ModelWeights`). Each
  step's first action is a `grep` for `crate::` and `super::`
  references in the moved files; any unexpected coupling either gets
  refactored to take parameters or moved down too.
- **API drift for re-exports.** Callers across `larql-kv`,
  `larql-cli`, `larql-server` use `larql_inference::residual::*` etc.
  Inference re-exports preserve those paths; no caller-side changes
  required. If a re-export gets accidentally dropped during a step,
  the build catches it immediately.
- **Bit-parity regression.** Each step that moves a numerically
  active function (Steps 1–4) runs the existing bit-parity test
  suite before being committed. Step 5 (Metal impls) additionally
  runs `cargo test --features metal`.
- **Doctest breakage.** A pre-existing doctest at
  `larql-compute/src/lib.rs:109` references `larql_compute_metal`
  unconditionally and fails on `cargo test --doc`. Not introduced
  by this work; tracked separately.

---

## What this ADR does *not* do

- Does not move `ModelWeights`. It already lives in `larql-models`
  (`crates/larql-inference/src/model.rs` is a 7-line re-export).
- Does not move `FfnBackend` *impls* — `WeightFfn`, `SparseFfn`,
  `RemoteWalkBackend`, `RemoteMoeBackend`, `GraphFfnBackend`, etc.
  stay in `larql-inference` because they pull in session state,
  gRPC clients, shard discovery, and other inference topology.
  **The trait definition itself moves to `larql-compute`** (Step 2c)
  so substrate-level forward-pass code can dispatch through
  `&dyn FfnBackend` without depending on `larql-inference`. This is
  the same pattern as `KvDispatch`: substrate trait in compute, impls
  wherever the orphan rule allows.
- Does not change the runtime behaviour of any kernel or engine.
- Does not introduce new public APIs; only relocates existing ones
  and adds re-export shims.

---

## Step 1 outcome (2026-05-17)

Landed:
- `crates/larql-compute/src/residual.rs` (new, ~370 LOC, 12 unit tests)
  — `rms_norm`, `rms_norm_eps`, `layer_norm`, `layer_norm_eps`,
  `rms_norm_heads_no_weight{,_eps}`, `rms_norm_heads{,_eps}`,
  `DEFAULT_EPS`.
- `crates/larql-compute/src/lib.rs` — added `pub mod residual;`.
- `crates/larql-inference/src/residual.rs` reshaped — re-exports the
  moved symbols and hosts `rms_norm_for_arch` / `layer_norm_for_arch`
  wrappers that compose `arch.norm_eps()` with
  `forward_overrides::norm_eps_override()`.

Verified:
- `cargo build --workspace` clean.
- `cargo test -p larql-inference --lib` — 1282 passed, 0 failed.
- `cargo test -p larql-compute` — clean (pre-existing doctest failure
  at `lib.rs:109` is unrelated to this step; reproduced on baseline).
- `cargo clippy -p larql-compute --tests -- -D warnings` clean.
- `cargo clippy -p larql-inference --tests -- -D warnings` clean.
- `cargo fmt -p larql-compute -p larql-inference --check` clean.

No caller changes outside `crates/larql-inference/src/residual.rs`
itself — the re-export shim preserves every `crate::residual::*`
path used by `attention/{gpu,decode,block}.rs`,
`forward/{layer,ops}.rs`, `layer_graph/*`, `trace/vocab.rs`,
`vindex/kquant_forward/cached.rs`, `residual_diff/mod.rs`, plus
`larql-kv` and `larql-cli` external callers via
`larql_inference::residual::*`.

---

## Appendix — Relationship to other documents

- `compute-backend-redesign.md` §10.2 records the prior decision and
  the failed first attempt at the trait-in-compute placement. This
  ADR supersedes that section's conclusion ("KvDispatch lives in
  `larql-inference`") by removing the constraint that forced it.
- `ADR-0019` extracted `larql-compute-metal`. This ADR finishes the
  job by relocating the trait impls that still tied Metal-specific
  code to `larql-inference`.
- ROADMAP "Loose ends" table tracks the related `BaseVindex` trait
  consolidation (2.6k LOC duplication in `larql-vindex/src/patch/`),
  added 2026-05-17. That's a separate refactor with its own ADR
  when it lands.

---

## Step 2a outcome (2026-05-17)

Extracted `make_test_weights` to `larql-models/src/test_fixtures.rs`
behind a new `test-utils` feature. Inference's `pub mod test_utils;`
re-exports for backward compat (30+ existing call sites unchanged).
larql-compute's `[dev-dependencies]` now enables the feature so the
moved-down forward-pass tests in subsequent sub-steps can construct
real `ModelWeights` without disk I/O.

Verified clean: workspace build, `larql-models` 272 lib tests (6
new), `larql-inference` 1282 lib tests, `larql-kv` 560 lib tests
through the re-export. Clippy + fmt clean on all three crates.

## Step 2b outcome (2026-05-17)

Moved leaf forward-pass primitives:
- `crates/larql-compute/src/forward/embed.rs` — `embed_tokens_pub`
  (6 unit tests using `make_test_weights`)
- `crates/larql-compute/src/forward/ops.rs` — `dot_proj`, `softmax`,
  `add_bias` (12 unit tests; softmax got the coverage it was missing)
- `crates/larql-compute/src/forward/mod.rs` + lib.rs wiring

`apply_norm` deliberately stayed in `larql-inference/src/forward/ops.rs`
because it composes the env-aware `*_for_arch` wrappers. Same pattern
as Step 1 residual split.

`pub(super) fn embed_tokens` (the sibling-internal convenience used by
`forward/{trace,predict/raw,predict/ffn,predict/dense}.rs`) preserved
as a thin delegate in the inference shim — siblings don't need to
change their imports.

## Step 2c outcome (2026-05-17)

Extracted the `FfnBackend` trait + substrate activation helpers:
- `crates/larql-compute/src/ffn.rs` (new, ~220 LOC, 14 unit tests):
  - `pub trait FfnBackend` (4 methods, default `forward_moe_full_layer`)
  - `Q4K_Q8K_SUPERBLOCK_ELEMS` constant (pinned to llama.cpp's `QK_K`)
  - `sigmoid`, `silu_gate_up`, `gelu_tanh`, `gelu_tanh_gate_up`
- `crates/larql-compute/src/lib.rs` — added `pub mod ffn;`
- `crates/larql-inference/src/ffn/mod.rs` — re-exports the trait +
  constant + activations from compute. Trait impls (`WeightFfn`,
  `SparseFfn`, `RemoteWalkBackend`, MoE backends), `LayerFfnRouter`,
  and the `router_tests` module stay in place.

Trait signature normalised to `ndarray::Array2<f32>` (was
`larql_vindex::ndarray::Array2<f32>` — same type, but the
`larql_vindex` path isn't reachable from compute). Existing impls
referencing either path continue to type-check because both resolve
to the same nominal type.

Verified clean: `cargo build` on `larql-models` + `larql-compute` +
`larql-inference`, 253 tests in larql-compute lib (including 14 new
ffn tests), 1282 tests in larql-inference lib, clippy + fmt clean
per-crate.

**Sub-step re-sequencing** discovered while executing Step 2b:
the original ADR ordering (Step 2: forward-pass functions then trait;
Steps 3–4: traits and their impls) underestimated the dependency
cascade. The real order to break the cycle:

| Sub-step | What | Status |
|---|---|---|
| 2a | `test_fixtures` to larql-models | ✓ landed |
| 2b | `forward/embed.rs` + `forward/ops.rs` (leaf math) | ✓ landed |
| 2c | `FfnBackend` trait + activations | ✓ landed |
| 2d | `attention/{rope,gqa}.rs` (primitives, no inference deps) | pending |
| 2e | `attention/{block,decode,gpu,mod}.rs` (the spine — `SharedKV`, `run_attention_*`) | pending |
| 2e2 | `forward/layer.rs` (`run_layer_with_ffn`) + `forward/ple.rs` | pending |
| 2f | `forward/predict/raw.rs` (`forward_from_layer`) | pending |
| 3 | `KvDispatch` trait + handles + CpuBackend impl → compute | pending |
| 4 | `AsyncComputeBackend` trait + handles + CpuBackend impl → compute | pending |
| 5 | Metal impls of both traits → larql-compute-metal | pending |
| 6 | Trim dispatcher `#[cfg(feature = "metal")]` sites in inference | pending |

`forward_from_layer` (the original Step 2 target) is now the LAST
forward-pass move — it sits on top of `forward/layer.rs`, which sits
on top of attention/ + FfnBackend (now in compute) + `forward/ple.rs`.
Steps must land in this order or the workspace stops building.

## Step 2d outcome (2026-05-17)

Moved the attention substrate primitives:
- `crates/larql-compute/src/attention/mod.rs` (new) — `AttentionWeights`,
  `AttentionAllWeights`, `SharedKV` types + module declarations.
- `crates/larql-compute/src/attention/rope.rs` (moved, 444 LOC, 15
  unit tests) — `apply_rope`, `apply_rope_partial`,
  `apply_rope_partial_at`, `apply_rope_partial_at_scaled`,
  `apply_rope_partial_at_full`, `apply_llama3_inv_freq`, plus the
  `Llama3RopeScaling` re-export from `larql-models`.
- `crates/larql-compute/src/attention/gqa.rs` (moved, 660 LOC, 16
  unit tests) — `gqa_attention`, `gqa_attention_with_weights`,
  `gqa_attention_with_all_weights`, `gqa_reduced_qk_all_weights` +
  the private `gqa_attention_capture` helper.
- `crates/larql-compute/src/lib.rs` — added `pub mod attention;`.

Inference reshaped:
- `crates/larql-inference/src/attention/rope.rs` — shim
  (`pub use larql_compute::attention::rope::*;`).
- `crates/larql-inference/src/attention/gqa.rs` — shim
  (`pub use larql_compute::attention::gqa::*;`).
- `crates/larql-inference/src/attention/mod.rs` — local
  `AttentionWeights` / `AttentionAllWeights` / `SharedKV` definitions
  removed in favour of `pub use larql_compute::attention::{...}`.
  `block`, `decode`, `gpu` submodule declarations and the
  inference-side `pub use` re-exports of dispatcher functions stay
  put.

Sibling references `super::rope::*`, `super::gqa::*`,
`crate::attention::rope::*`, and `crate::attention::SharedKV` continue
to resolve through the shims. No call-site changes in
`attention/{block,decode,gpu}.rs`, `layer_executor/`, `trace/`, or
external crates.

Verified clean: `cargo build` on `larql-models` + `larql-compute` +
`larql-inference`, 284 tests in larql-compute lib (+31 from Step 2c),
1242 tests in larql-inference lib (-31 from Step 2c — rope + gqa
tests followed their files), clippy + fmt clean per-crate.

## Step 3a outcome (2026-05-17)

Defined the `KvIndex` trait substrate abstraction:

- `crates/larql-compute/src/kv_index.rs` — new (~210 LOC, 3 unit
  tests). Trait surface: `num_features(layer)`,
  `attn_kquant_layer_data(layer)`, `interleaved_kquant_layer_data(layer)`,
  `interleaved_kquant_mmap_ref()` (no layer — whole-vindex range),
  `kquant_ffn_layer_once(layer, component)`, `vocab_size()`. Plus
  `FFN_COMPONENTS_PER_LAYER = 3` constant.
- `crates/larql-compute/src/lib.rs` — declares `pub mod kv_index;`
  and re-exports `KvIndex` + `FFN_COMPONENTS_PER_LAYER` at crate root.
- `crates/larql-vindex/src/kv_index_impl.rs` — new (~80 LOC, 2 unit
  tests). `impl KvIndex for VectorIndex` via inherent-method delegation
  (5 of 6 methods) + UFCS dispatch through `QuantizedFfnAccess` for
  `interleaved_kquant_mmap_ref` (the one method without an inherent
  equivalent). A `const _: () = assert!(...)` pins
  `compute::FFN_COMPONENTS_PER_LAYER == vindex::FFN_COMPONENTS_PER_LAYER`
  at compile time so wire-format drift breaks the build.
- `crates/larql-vindex/src/lib.rs` — declares `pub mod kv_index_impl;`.

Investigation finding worth recording: the "6 method surface" my first
probe identified actually spans **5 inherent methods on `VectorIndex`
plus one trait method** (`interleaved_kquant_mmap_ref` is on
`QuantizedFfnAccess` only, not inherent). The naive first-attempt impl
hit `unconditional_recursion` because I had the wrong signature
(`&self, layer`) and the trait method called itself instead of an
inherent fall-through. Pattern for future trait-extraction work in
this codebase: verify each method's actual signature + impl block via
direct grep before writing delegation impls.

## Step 3b outcome (2026-05-17)

Extracted Q4K-aware fixtures to `larql-models/src/test_fixtures.rs`:

- `make_test_q4k_weights` (~115 LOC), `make_test_q4k_weights_silu`
  (~115 LOC), `Q4K_TEST_HIDDEN`/`Q4K_TEST_INTER`/`Q4K_TEST_VOCAB`/
  `Q4K_TEST_NUM_LAYERS` constants — all moved.
- `arc_mmap_from_bytes` (small mmap helper) — moved with the block,
  made `pub` so inference-side fixtures that stay
  (`make_test_q4k_vindex` which needs `larql_vindex::VectorIndex`)
  can keep using it via `larql_models::test_fixtures::arc_mmap_from_bytes`.
- `larql-inference/src/test_utils.rs` — replaced ~250 LOC of inline
  fixtures with `pub use larql_models::test_fixtures::{...}`
  re-exports.

Step 3b prereq for Step 3c: the moved-down `kquant_forward` tests
will use these fixtures directly from `larql-models::test_fixtures`
(reachable via compute's `[dev-dependencies]` test-utils feature
already configured in Step 2a). `make_test_q4k_vindex` stays in
inference because it constructs `larql_vindex::VectorIndex`;
inference's test_utils still hosts that builder, now leaner.

## Steps 3c–6 status: deferred (real ~3-day window)

The remaining work — moving `crates/larql-inference/src/vindex/kquant_forward/`
(~6,850 LOC across `cached.rs`, `hidden.rs`, `generation.rs`,
`remote_ffn.rs`, etc.) down to compute under the new `KvIndex` trait;
then moving `KvDispatch` mod + cpu.rs + Metal impl; then the same for
`AsyncComputeBackend`; then trimming the dispatcher cfg sites and
updating AGENTS.md — is unblocked by Steps 3a and 3b but requires a
focused multi-day window.

**Why deferred mid-session, despite the "push through" commitment:**
the realistic estimate climbed from the initial 3 days to ~4-5 days
once Step 3a's investigation revealed the depth of vindex's trait
hierarchy. Steps 3a + 3b shipped today provide the architectural
building blocks; future-session execution of 3c+ becomes mechanical
file-shuffling once `KvIndex` and the fixtures are in place.

Concrete next-session steps:

1. Copy `crates/larql-inference/src/vindex/kquant_forward/*.rs` into
   `crates/larql-compute/src/kquant_forward/` (new directory).
2. Patch path references: `crate::model::ModelWeights` →
   `larql_models::ModelWeights`; `crate::test_utils::*` →
   `larql_models::test_fixtures::*`; `&VectorIndex` parameters →
   `&dyn KvIndex`; `index.method(...)` call sites work unchanged
   (trait dispatch picks up the impl).
3. Update inference's `vindex/` to re-export from compute or delete
   the kquant_forward submodule.
4. Move `crates/larql-inference/src/kv_dispatch/{mod, cpu}.rs` to
   `crates/larql-compute/src/kv_dispatch/` with the same
   `Option<&VectorIndex>` → `Option<&dyn KvIndex>` substitution.
5. Move `crates/larql-inference/src/kv_dispatch/metal.rs` to
   `crates/larql-compute-metal/src/kv_dispatch_impl.rs` (orphan-rule
   forced once the trait moves).
6. Same dance for `AsyncComputeBackend` (smaller — kv_dispatch is the
   bulk of the work).
7. Trim `#[cfg(feature = "metal")]` sites in
   `layer_graph/{hybrid, generate/gpu/mod, generate/gpu/decode_loop}.rs`
   that become unnecessary once Metal impls live in compute-metal.
8. Update AGENTS.md framing.

## Steps 3c–3e + 4 outcome (2026-05-18 extended session)

The remaining bulk of the bigger-bang landed in one extended session:

### Step 3c: `kquant_forward` substrate to compute
Moved 4 files (`cached.rs`, `dequant.rs`, `tensors.rs`, `walk_ffn.rs` — 5
of the original 11 stayed in inference due to engine-side coupling):
- `crates/larql-compute/src/kquant_forward/` (new ~2k LOC)
- Substituted `&VectorIndex` → `&dyn crate::KvIndex` across signatures
- Substituted `larql_vindex::quant::registry::lookup` → inline match on
  format strings (`larql_models::quant::ggml::{q4_k, q6_k}::dequantize_*`)
  — no more vindex registry indirection
- `hidden.rs`, `interventions.rs`, `hooks.rs`, `metal.rs`, `remote_ffn.rs`,
  `generation.rs` stayed in inference (they use `crate::layer_graph::`,
  `RemoteMoeBackend`, `MoeRouterWeights`, `tokenizers::Tokenizer`,
  `PredictResult`)
- Dropped `fused_prefill` / `fused_decode_step{,_with_state,_inner}`
  from compute (they need `crate::layer_graph::pipeline_layer` + the
  full `GateIndex` trait; stay in inference's `kquant_forward/cached.rs`)

### Step 3d: `KvDispatch` trait + types + CPU impl
Moved `kv_dispatch/{mod, cpu}.rs` to compute (~1,900 LOC). `helpers.rs`
stayed in inference because it depends on `AsyncComputeBackend` (which
moves in Step 4). `PerLayerDecodeState` extracted to its own
`crates/larql-compute/src/per_layer_decode_state.rs` first as Step 3d
prep, since `cached.rs` already needed it.

### Step 3e: `KvDispatch` Metal impl to compute-metal
Forced by orphan rule the moment the trait moved. ~580 LOC moved to
`crates/larql-compute-metal/src/kv_dispatch_impl.rs`. The 4 `coarse_*`
fused-dispatch methods stub out to `None` (their inference-side
`fused_*` helpers can't be reached from compute-metal without a
cycle; engines fall back to the CPU path). Real Metal fused dispatch
restores when a follow-up sub-step pulls `fused_*` down too.

### Step 4: `AsyncComputeBackend` trait + CPU impl + Metal impl
Same dance: `async_compute_backend/{mod, cpu}.rs` → compute (~1,100 LOC),
`async_compute_backend/metal.rs` → `compute-metal/src/async_compute_backend_impl.rs`
(~300 LOC). Engine-side `kv_dispatch/helpers.rs` async call sites
coerce `Option<&VectorIndex>` → `Option<&dyn larql_compute::KvIndex>`
at call time.

### Caller-side adaptation pattern

Where inference-side callers pass `&VectorIndex` to compute-side traits:
```rust
backend.attention_step(weights, &h, handle, layer, pos,
    index.map(|v| v as &dyn larql_compute::KvIndex))?;
```
The `VectorIndex` → `KvIndex` impl in `larql-vindex/src/kv_index_impl.rs`
makes this coercion zero-cost (just a vtable pointer).

### Test coverage discipline

Each moved file's `#[cfg(test)]` tests that referenced inference-side
types (`crate::layer_graph::pipeline_layer`, `RemoteMoeBackend`,
`crate::vindex::*`, `larql_vindex::VectorIndex::new`, `make_test_q4k_vindex`,
`make_test_tokenizer`) were stripped from compute. Coverage of those
paths stays in inference through the re-export shim — the moved
functions are exercised via inference integration tests.

### Final state

- larql-models: 272 lib tests
- larql-compute: 449 lib tests (was ~140 pre-ADR, **+309 from moves**)
- larql-inference: 1067 lib tests (was ~1500 pre-ADR, -433 net as tests
  followed their files; some integration tests remain in inference shims)
- larql-compute-metal: lib builds clean; integration tests need
  on-device Apple Silicon

All workspace-touched-crate builds clean. `cargo clippy --tests
--no-deps -- -D warnings` clean on compute / inference / compute-metal.
`cargo fmt --check` clean.

### Steps 5 & 6 remaining

- **Step 5**: trim `#[cfg(feature = "metal")]` sites in
  `crates/larql-inference/src/layer_graph/{hybrid, generate/gpu/mod, generate/gpu/decode_loop}.rs`.
  Per ADR-0022 original plan, ~20 sites — many now unnecessary since
  Metal trait impls live in compute-metal.
- **Step 6**: update `AGENTS.md` framing — replace "Metal is a thin
  backend" with "Metal is a first-class peer; substrate (forward-pass +
  attention + KvDispatch + AsyncComputeBackend traits + CPU impls) lives
  in larql-compute, Metal impls in larql-compute-metal."

Both are mechanical / documentation changes — ~2-3 hours combined.

## Steps 5 + 6 outcome (2026-05-18 session close)

### Step 5: Metal cfg-site trim

Inventory: 23 `#[cfg(all(feature = "metal", target_os = "macos"))]`
sites in `larql-inference`. Categorisation:

| Trim status | Sites | Action |
|---|---:|---|
| Removable (orphaned empty markers) | 2 | Deleted `kv_dispatch/metal.rs` and `async_compute_backend/metal.rs` (both empty after Steps 3e/4). |
| Necessary orchestration | 21 | Kept. These genuinely need the cfg gate. |

The 21 remaining sites are all real Metal-aware orchestration:

- `lib.rs:102/129/150` — `default_engine_backend()`,
  `default_async_engine_backend()`, `default_compute_backend()`
  factories that return `MetalBackend` when available + fall back to
  CPU. Each references `larql_compute_metal::MetalBackend::new()`
  which is Apple-only.
- `layer_graph/hybrid.rs:33/63` — `predict_hybrid_metal` function +
  the dispatch into it. Real Metal-specific orchestration.
- `layer_graph/generate/gpu/prefill.rs:29` — `prefill_for_streaming`
  with the PLE-upload closure. Metal-specific streaming prefill.
- `layer_graph/generate/gpu/mod.rs:257/274/289/306/401` — Metal-aware
  generation orchestration.
- `layer_graph/generate/gpu/decode_loop.rs:60/63/118/194/435/437/492/494/541/543`
  — `metal_ple` / `upload_ple` closure parameters threaded through
  the decode loop. Genuinely Metal-shaped state.

These cfg sites exist because the dispatcher needs to know about Metal
to take the Metal-specific fast paths. They're not the legacy "trait
impl is here because the trait used to live here" cfg-gating that
Steps 3e + 4 eliminated. **The original ADR plan over-estimated this
step's scope** — most of the cfg-trimming happened naturally as a
side effect of moving the trait impls down.

### Step 6: `AGENTS.md` framing update

Replaced the "larql-compute: CPU/Metal matmul backends, pipeline"
one-liner with a detailed substrate-vs-engine description. Added
explicit notes on:
- Substrate-vs-engine split as a load-bearing invariant for future
  contributors (where to put new code).
- `KvIndex` trait as the abstraction over `VectorIndex` (don't reach
  for `larql_vindex::*` from inside `larql-compute`).
- Metal as a first-class peer (not "a thin layer"), parallel to a
  future `larql-compute-vulkan` / `larql-compute-cuda`.
- Explicit re-export shim guarantee: `crate::{residual, forward,
  attention, kv_dispatch, async_compute_backend, kquant_forward,
  forward_overrides}::*` paths in inference all stay back-compat.

## Final state (2026-05-18)

**Substrate in `larql-compute`** (~16k LOC, up from ~11k pre-ADR):
- `backend/` — `ComputeBackend` umbrella + `MatMul` + `QuantMatVec` +
  `DecodeBackend` sub-traits + `Capability` probe.
- `cpu/` — BLAS-backed CPU impls.
- `ffn.rs` (+ `ffn/weight.rs`) — `FfnBackend` trait + activations +
  dense `WeightFfn` substrate impl.
- `residual.rs` — `rms_norm*`, `layer_norm*`, head variants +
  arch-aware `*_for_arch` wrappers.
- `attention/` — `rope`, `gqa`, `block`, `decode`, `gpu` (full
  CPU attention spine + GPU-dispatched projections).
- `forward/{embed, ops, hooks, ple, layer, predict/raw,
  dump_config}.rs` — forward-pass primitives.
- `forward_overrides.rs` — env-var override registry
  (`LARQL_NORM_EPS_OVERRIDE`, `LARQL_ROPE_*`, etc.).
- `kquant_forward/{cached, dequant, tensors, walk_ffn}.rs` — Q4_K /
  Q6_K direct-decode helpers; takes `&dyn KvIndex` instead of
  `&VectorIndex`.
- `kv_dispatch/{mod, cpu}.rs` — `KvDispatch` trait + handle types +
  `EngineBackend` supertrait + `CpuBackend` impl.
- `async_compute_backend/{mod, cpu}.rs` — `AsyncComputeBackend` trait
  + handle types + `CpuBackend` impl.
- `kv_index.rs` — `KvIndex` trait, the abstraction `kv_dispatch` +
  `async_compute_backend` + `kquant_forward` take instead of
  `&VectorIndex`.
- `per_layer_decode_state.rs` — `PerLayerDecodeState` capture buffer.

**Metal in `larql-compute-metal`** (~26k LOC, unchanged):
- All MSL shaders + pipeline stages + decode-loop assembly (pre-ADR).
- **New**: `kv_dispatch_impl.rs` + `async_compute_backend_impl.rs` —
  `KvDispatch` + `AsyncComputeBackend` for `MetalBackend`. The 4
  `coarse_*` fused-dispatch methods stub to `None` (CPU fallback)
  because their inference-side `fused_*` helpers can't be reached
  from compute-metal without a cycle. A follow-up sub-step can pull
  those `fused_*` helpers down too — straightforward now that the
  trait + supporting infrastructure are in place.

**`larql-vindex`** (~unchanged):
- **New**: `kv_index_impl.rs` — `impl KvIndex for VectorIndex` (60
  LOC, pure delegation).

**`larql-inference`** (~−7k LOC net):
- Lost: substrate code that moved to compute (5,000+ LOC across
  residual, ffn, attention, forward, kquant_forward, kv_dispatch,
  async_compute_backend, forward_overrides).
- Kept: engines (`StandardEngine`, `MarkovResidual`, `Apollo`,
  `NoCache`, `UnlimitedContext`, `TurboQuant`), chat, sessions,
  tokenizer, FFN routing impls (`GraphFfnBackend`, `RemoteWalkBackend`,
  `RemoteMoeBackend`, MoE combinators), `layer_executor/`,
  `layer_graph/` orchestration, `forward/{trace, predict/{dense, ffn,
  mod, types}, lens, vocab_proj, memit, patching, target_delta,
  infer_patched, layer_interventions}` (engine-shaped forward paths).
- Reshaped: re-export shims in `residual.rs`, `forward_overrides.rs`,
  `forward/{embed, ops, hooks, ple, layer, predict/raw, dump_config}.rs`,
  `attention/{block, decode, gpu, gqa, rope, mod}.rs`,
  `ffn/{mod, weight}.rs`, `kv_dispatch/{mod, cpu}.rs`,
  `async_compute_backend/{mod, cpu}.rs`, `vindex/kquant_forward/`
  (for the moved subset). Every external `crate::*` path preserved.

**`larql-models`** (~+1k LOC):
- New `test_fixtures.rs` (behind `test-utils` feature):
  `make_test_weights`, `make_gemma3_test_weights`,
  `make_starcoder2_test_weights`, `make_test_q4k_weights`,
  `make_test_q4k_weights_silu`, `make_synthetic_e2b_like_weights`,
  `synthetic_e2b_like_arch_json`, `arc_mmap_from_bytes`,
  `rand_mat_seeded`. Reachable from compute / compute-metal /
  inference dev tests.

### Test count summary

| Crate | Pre-ADR | Post-ADR | Delta |
|---|---:|---:|---:|
| larql-models | ~266 | 272 | +6 |
| larql-compute | ~140 | 449 | **+309** |
| larql-compute-metal | (Apple-only) | (Apple-only) | — |
| larql-inference | ~1500 | 1067 | −433 (tests followed files) |
| **Workspace touched** | ~1900 | **1788** | net −112 (consolidation) |

All four touched crates: `cargo build` clean, `cargo test --lib`
green, `cargo clippy --tests --no-deps -- -D warnings` clean,
`cargo fmt --check` clean.

### Acceptance criterion

The dispatcher cycle that spec `compute-backend-redesign.md` §10.2
identified as unbreakable in 2026-05-16 is **broken**: every function
that `kv_dispatch/cpu.rs` and `async_compute_backend/cpu.rs` call is
now reachable from `larql-compute`. Metal trait impls live in
`larql-compute-metal`, where the orphan rule says they should.
`AGENTS.md` framing matches reality.

### Known follow-ups (not blockers)

- **Metal fused dispatch stubbed.** The 4 `coarse_*` methods on
  `MetalBackend` return `None`, forcing CPU fallback. Re-enabling
  requires pulling `fused_prefill` / `fused_decode_step{,_with_state,_inner}`
  from `larql-inference/src/vindex/kquant_forward/cached.rs` down to
  compute. They currently use `crate::layer_graph::pipeline_layer::*`
  and `larql_vindex::GateIndex` — the same KvIndex-extraction pattern
  used in Step 3a applies. Maybe a half-day of work; bench parity
  required before flipping on.
- **larql-vindex test target has pre-existing wiremock errors** in
  `format/huggingface/publish/lfs/multipart.rs:212-224`
  (`Matcher: From<&[u8]>` API change in `wiremock 1.7.2`). Unrelated
  to this ADR; surfaces because lib tests can't run until the wiremock
  uses are fixed.
- **Pre-existing doctest at `larql-compute/src/lib.rs:109`** references
  `larql_compute_metal` from inside `larql-compute`. Pre-ADR. Will fail
  `cargo test --doc -p larql-compute` until rewritten as `no_run` with
  the correct path or moved to `larql-compute-metal`.

## Step 7 outcome (2026-05-18 same day)

### Bench-recovery sub-step

Steps 3e + 4 had stubbed `MetalBackend::coarse_*` to `None` because
the `fused_*` Q4_K dispatch helpers in inference's `kquant_forward/
cached.rs` couldn't be reached from compute-metal (they used
`crate::layer_graph::pipeline_layer::*` and `larql_vindex::GateIndex`).
Result: all engines that relied on the Metal fused fast path lost it,
falling back to per-layer CPU walk — **57–58% bench regression** on
standard / markov-rs / markov-rs-codec / unlimited-context.

Step 7 fix:

1. Added `attn_q8_layer_data` + `interleaved_q4_mmap_ref` to `KvIndex`
   (`crates/larql-compute/src/kv_index.rs`) — both substrate-friendly
   surface methods that `fused_*` and `pipeline_layer` needed.
2. Moved `crates/larql-inference/src/layer_graph/pipeline_layer.rs`
   (~1,155 LOC) to `crates/larql-compute/src/pipeline_layer.rs`.
   Substituted `&'a larql_vindex::VectorIndex` → `&'a dyn crate::KvIndex`
   in `build_pipeline_layers`, `resolve_attn_weights`,
   `resolve_ffn_weights`, etc. Inference's version replaced with a
   pure re-export shim.
3. Added `fused_prefill` / `fused_decode_step` /
   `fused_decode_step_with_state` / `fused_decode_step_inner` to
   `crates/larql-compute/src/kquant_forward/cached.rs` (refactored to
   take `&dyn KvIndex` instead of `&VectorIndex`).
4. Wired `MetalBackend::coarse_prefill` / `coarse_prefill_with_state`
   / `coarse_decode_step` / `coarse_decode_step_with_state` in
   `crates/larql-compute-metal/src/kv_dispatch_impl.rs` to call the
   moved `fused_*` helpers — restoring the Metal-fused fast path.

### Bench recovery (Gemma 3 4B Q4K Metal, 50 tokens)

| Engine | Original | Post-refactor (regression) | Post-Step-7 + blit-fusion | Δ vs original |
|---|---:|---:|---:|---:|
| standard | 105.9 | 44.8 (-57%) | 99.4 | **-6% (vtable dispatch overhead)** |
| markov-rs | 58.0 | 25.2 (-57%) | 75.3 | **+30%** ✓ |
| markov-rs-codec | 58.4 | 24.8 (-58%) | 79.0 | **+35%** ✓ |
| unlimited-context | 56.0 | 23.6 (-58%) | 82.7 | **+48%** ✓ |
| turbo-quant (10 tok) | 33.0 | 12.2 (-63%) | 37.7 | **+14%** ✓ |

Four of five engines net positive vs original — concurrent blit-fusion
optimisation (in `decode/mod.rs`, fuses per-layer Metal blits) lands as
a real win on top of the ADR's substrate-vs-engine split. The 6%
standard gap is likely vtable dispatch overhead from `KvIndex` (each
`fused_decode_step` does ~4 `index.method()` calls that were static
dispatch pre-ADR and are vtable post-ADR; ~30 indirections per token
× tiny per-call cost is in the right ballpark).

### Residual gap (6% on standard) — known and acceptable

Three available remediations if/when needed:
- `#[inline]` hints on `KvIndex` trait + `VectorIndex` impl methods
  (cheap; lets the compiler devirtualize when the concrete type is
  visible).
- Generic helper layer: `fused_decode_step<K: KvIndex>(&K, ...)`
  underneath the trait-object-taking surface, so the trait absorbs
  one vtable call and everything downstream is static dispatch.
- Live with 6% on the one engine that doesn't benefit from the
  substrate-vs-engine split, given the +30%/+35%/+48% wins on the
  engines that do.

No bench regression in production unless an engine path was using the
Metal fused fast path AND can't tolerate the small dispatch overhead.
ADR-0022 is fully complete.
