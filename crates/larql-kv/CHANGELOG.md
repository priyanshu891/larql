# Changelog — larql-kv

All notable changes to `larql-kv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-20] — boundary_per_layer: bugfixes + W1-GPU dispatch + modular split

**Engine bottleneck audit** (`PERFORMANCE.md` §"2026-05-20"). Findings
across all engines:

- `apollo` — O(N²) **by design** (`forward_from_layer` rebuilds KV each
  step over the growing context; no cross-step persistence). Not a
  bug; documented as a contract caveat for short-query workloads.
- `boundary_per_layer` — two real O(N²) bugs, both fixed:
  - **Bug A** (hot-tier rebuild): every `decode_step` rebuilt every
    layer's `stored[layer]` via `Array2::zeros((s_old+1, h)) + assign`.
    O(N · num_layers · hidden) per step → O(N²) total in unbounded
    mode. Replaced with `ndarray::Array2::push_row` (amortised O(m)).
  - **Bug B** (cold_kv nuke): every overflow set `cold_kv = None`,
    forcing the next decode to recompute K/V over the entire cold
    tier — O(N²) windowed mode. Replaced with
    `cold_tier::extend_cold_kv_with_overflow` which appends K/V at
    each overflow at the pre-`cold_encoded.append` absolute position.

**W1-GPU dispatch wired** for `boundary_per_layer`. New
`try_prefill_via_dispatch` + `decode_step_via_dispatch` route through
the Metal-fused per-layer state-dump kernel when the backend/vindex
support it. Closes the perf gap to its sister engine
`markov_residual_codec`: **91.8 tok/s** vs codec's 92.6 (−0.9%) on
Gemma 3 4B Q4K, M3 Max — with **44% less hot memory** (19.6 MB vs
35.3 MB). Falls back to dense walk on backends/vindexes lacking
direct-matvec.

**FFN routing fix** — `boundary_per_layer`'s dense `run_prefill` /
`run_decode` previously constructed `BackendFfn` internally, ignoring
the caller-supplied `ffn`. This panicked on `--compact` vindexes
where dense FFN weights aren't present. Now routes the caller's FFN
through (e.g. `WalkFfn` from the bench CLI).

**`EngineKind` variant + parser**. `BoundaryPerLayer { window_size,
num_layers }` with three aliases (`boundary-per-layer`,
`boundary_per_layer`, `boundary-pl`); default `num_layers=34` (Gemma
3 4B), override via `layers=N`. Build dispatch seeds a uniform-bf16
`InMemoryCalibrationStore` automatically. Added to
`examples/engine_ladder.rs`.

**Parity gate** — `examples/boundary_per_layer_parity_gate.rs` runs
`boundary-per-layer` vs `markov-rs-codec` end-to-end on a real Gemma
3 4B Q4K vindex. Token-level agreement check (not bit-identity,
because incremental cold_kv vs recompute-each-step differ in BLAS
accumulation order). Pass criterion: first divergence ≥ step 5.
Result on Gemma 3 4B: **100% token agreement** across 50 tokens in
both unbounded and windowed (window=512) — RoPE positioning in
`extend_cold_kv_with_overflow` and codec round-trip are exactly
right.

**Modular split** of `boundary_per_layer/engine.rs` (1250 → 716 LOC),
mirroring `markov_residual_codec`'s module layout. New sibling files
in `engines/boundary_per_layer/`:

- `walk.rs` (204 LOC) — CPU dense walk path
  (`run_prefill` / `run_decode` as free functions).
- `dispatch.rs` (162 LOC) — W1-GPU dispatch path.
- `executor.rs` (186 LOC) — `LayerExecutor`-driven path.
- `cold_tier.rs` (130 LOC) — `extend_cold_kv_with_overflow` +
  `roundtrip` / `last_row` helpers + their unit tests.

Struct fields moved to `pub(super)` so sibling modules can read them
via free-function inputs.

**Test count**: 591 → 598 lib tests (3 parser variants + 1 cold_kv
invariant + 3 from cold_tier extraction). All passing.

The same split pattern is queued for the other 6 engines
(`markov_residual_codec`, `turbo_quant`, `unlimited_context`,
`apollo`, `boundary_kv`, and `markov_residual` last) — deferred to
follow-up turns since each requires its own care and at least one is
gated on in-flight WIP in `markov_residual/compute.rs`.

## [2026-05-16] — KV engine unification (steps 1-5 of 7)

Unifies the parallel "live decode cache" and "research KV engine" code
paths so `larql run` / `larql walk` dispatch through the same `KvEngine`
trait that `larql bench --engine` uses. Spec at
[`crates/larql-inference/docs/specs/kv-engine-unification.md`](../larql-inference/docs/specs/kv-engine-unification.md).

**Trait surface relocated.** `KvEngine` + `EngineInfo` +
`DecodeStageSummary` now live in `larql-inference::kv_engine`; this
crate re-exports them so `larql_kv::KvEngine` keeps the same public
shape. Engine impls in `larql-kv/src/engines/*` continue to write
`impl KvEngine for ...` against the same trait — just resolved through
the re-export. The trait moved upstream so the dispatch entry point
(`larql_inference::forward::generate_with_engine`) can reference it
without inducing a circular dep on `larql-kv`. See spec §10.4.

**Trait widened for FFN dispatch.** `KvEngine::{prefill, decode_step,
prefill_q4k, decode_step_q4k}` now take `ffn: &dyn FfnBackend` after
`weights`. Existing four engines ignore the parameter (FFN is recomputed
from weights as before); new param is plumbing for future engines that
route FFN remotely (`RemoteWalkBackend`, `RemoteMoeBackend`).
`larql_inference::ffn::NullFfn` added as a trait-satisfying stub that
holds no references — used by Q4K callers where `&mut weights` rules
out a `WeightFfn`.

**Two new engines** in `larql-kv/src/engines/`:

- `StandardEngine` — wraps the production K/V tensor cache. `window=None`
  matches today's `--kv-cache standard`; `Some(N)` matches
  `--kv-cache markov-bounded --context-window N`. Bit-identical output.
- `NoCacheEngine` — wraps the O(N²) re-forward fallback. Matches today's
  `--kv-cache none` on non-PLE architectures.

`EngineKind` gains `Standard { window_size: Option<usize> }` and
`NoCache` variants. `from_name` recognises `standard[:window=N]`,
`markov-bounded[:window=N]` (legacy alias → `Standard`), `no-cache`,
`none` (legacy → `NoCache`), plus existing aliases.

**Default flipped** to engine dispatch. `walk_cmd::generate_stream` no
longer carries the legacy `match` over `KvCacheKind`; it builds an
`EngineKind` from the flag and drives `generate_with_engine`.

**Bit-parity gate** lives in `larql-kv/src/engines/{standard,no_cache}.rs`:
- `Standard { window=None }` vs `generate_cached_backend(window=None)` ✓
- `Standard { window=Some(3) }` vs `generate_cached_backend(window=Some(3))` ✓
- `Standard { window=Some(64) }` short-prompt edge case ✓
- `NoCacheEngine` vs legacy `predict_with_ffn` loop ✓ (non-PLE)

**Engine-trait dispatch overhead** measured at ~1.6 % (within noise) on
the synthetic test substrate. See [`PERFORMANCE.md`](PERFORMANCE.md).

**Coverage:** new files land at 99.1 % (`standard.rs`) and 96.1 %
(`no_cache.rs`); upstream `kv_engine.rs` at 94.3 %. Per-file 90 % floor
met for everything new.

Steps 6 (CLI `--engine` flag, `LARQL_KV_ENGINE` env var, server wiring,
ROADMAP update) and 7 (cleanup) pending.

## [2026-05-10] — Coverage push

Total line coverage **67.44 % → 85.13 %** (+17.69 pp, 217 tests, +66 vs
extraction-day). 15 of 21 source files now at ≥ 90 %; the remaining 6
all carry tightened debt baselines.

| File | Before | After |
|---|---:|---:|
| `profiler.rs` | 0.00 % | 100.00 % |
| `engines/apollo/npy.rs` | 58.20 % | 93.61 % |
| `engines/apollo/engine.rs` | 71.98 % | 96.31 % |
| `engines/apollo/store.rs` | 17.81 % | 89.78 % |
| `engines/markov_residual/engine.rs` | 72.02 % | 93.23 % |
| `engines/markov_residual/q4k.rs` | 0.00 % | 57.14 % |
| `lib.rs` | 84.79 % | 90.03 % |

Notable additions:

- 8 `profiler` tests covering `StageAccumulator`, `EngineProfiler`, and
  `DecodeStageSummary` (including the `print()` smoke test for both the
  recompute-tier-present and total-zero branches).
- 4 `compliance_tests` lifting the default `KvEngine::prefill_q4k` /
  `decode_step_q4k` trait-method fallbacks via a synthetic
  `DefaultMethodsEngine` fixture.
- 5 `markov_residual::engine` tests covering profiling on/off split, the
  `with_profiling` setter, and the Q4K CPU fallback (Metal returns
  `None` → `rs_prefill_walk` / `rs_decode_step_walk`).
- 22 `apollo::npy` tests covering all `NpyError` variants, structured
  vs simple dtype dispatch, header field-parser branches.
- 13 `apollo::store` tests including end-to-end `ApolloStore::load`
  against a synthetic on-disk store built with `tempfile` + handwritten
  `.npy`/`.npz` fixtures.
- 11 `apollo::engine` tests including KvEngine `prefill` / `decode_step`
  for both compressed (boundary residual) and uncompressed paths,
  `query_greedy` smoke test, and `store()` getter.

### Warnings cleanup

Same day: removed 3 unused-import warnings in
`kv-cache-benchmark/src/real_model/{decode_comparison,runner}.rs`,
reverted a `kv_dim.is_multiple_of(hd)` clippy-fix in
`turbo_quant/engine.rs` (1.87.0 stable, MSRV 1.80), and reordered
`apollo/engine.rs` so the `KvEngine` impl precedes the test module
(satisfies clippy's `items-after-test-module`). `cargo clippy -p
larql-kv --all-targets --no-deps` is now clean.

### Cross-platform CI

Added `.github/workflows/larql-kv.yml` modelled on
`.github/workflows/larql-vindex.yml`. Test matrix runs on
`ubuntu-latest`, `windows-latest`, and `macos-14` covering fmt check,
`cargo check --all-targets`, examples, clippy, unit tests, and
bench-compile/test. Coverage job runs on Ubuntu only and gates on
`make larql-kv-coverage-policy` (the per-file 90 % floor + the 6
inherited debt baselines). OpenBLAS gets installed via apt on Linux
and via vcpkg on Windows; macOS uses the Accelerate framework — same
matrix the larql-vindex workflow already exercises.

`cargo fmt -p larql-kv` was run to bring three files (benches,
examples, an apollo test fixture) into conformance with the rest of
the workspace.

The Makefile's `larql-kv-lint` uses `--no-deps` so it doesn't trip on
pre-existing clippy debt in the larql-inference dependency. Other
crates' lint targets don't need this because they don't depend on
larql-inference.

## [2026-05-09] — Initial extraction from larql-inference

Genesis commit. The crate was carved out of
`larql-inference/src/engines/` (~5,540 LOC) where the four KV engines and
the supporting trait/dispatch had grown into a self-contained subsystem
with a real second consumer (`kv-cache-benchmark`) already importing it
through compatibility shims.

### Moved into larql-kv

| Component | Origin | Notes |
|---|---|---|
| `KvEngine` trait, `EngineKind`, `EngineInfo` | `engines/mod.rs` | Now the crate root. |
| `accuracy` module | `engines/accuracy.rs` | `softmax` re-exported from `larql_inference::forward::softmax` instead of being internal. |
| `profiler` module | `engines/profiler.rs` | Verbatim. |
| `engines::apollo` | `engines/kv_engines/apollo/` | Drop the redundant `kv_engines/` middle path. |
| `engines::markov_residual` | `engines/kv_engines/markov_residual/` | |
| `engines::turbo_quant` | `engines/kv_engines/turbo_quant/` | |
| `engines::unlimited_context` | `engines/kv_engines/unlimited_context/` | |

All `crate::{attention,ffn,forward,layer_graph,model,residual,vindex}::*`
paths inside the moved code rewritten to `larql_inference::*`.

### Stayed in larql-inference

- `engines::test_utils` — relocated to `larql_inference::test_utils`. ~20
  internal tests across `attention/`, `forward/`, `ffn/`, `layer_graph/`,
  `trace/`, `vindex/walk_ffn/` use these synthetic-weight fixtures and
  cannot follow into a downstream crate without a circular dep.

### Public-API surface widened in larql-inference

- `DEFAULT_GPU_KV_CACHE_MAX_SEQ` lifted from `pub(crate)` to `pub` in
  `layer_graph::pipeline_layer` so engines can read it from the new home.

### Removed re-exports from `larql_inference::*`

The following used to be at the `larql_inference` crate root or in
`research::*` and now live in `larql-kv`:

- `EngineInfo`, `EngineKind`, `KvEngine`
- `MarkovResidualEngine`, `UnlimitedContextEngine`
- `compare_hidden`, `cosine_similarity`, `js_divergence`, `kl_divergence`,
  `mse`, `softmax`, `HiddenAccuracy`

Downstream consumers should add `larql-kv` to their Cargo.toml and import
from there.

### Consumer updates

- `larql-cli` — `bench_cmd.rs` now imports `EngineKind` and
  `kv_memory_bytes_for_seq` from `larql_kv`. Workspace metal feature gained
  `larql-kv/metal`.
- `kv-cache-benchmark` — compat shims (`apollo/`, `turboquant/`,
  `unlimited_context/`, `real_model/markov_layer.rs`) now re-export from
  `larql_kv` directly. README updated.
- `larql-inference` examples — `apollo_rd_backend.rs` imports from
  `larql_kv::apollo`; `mech_interp_demo.rs` uses
  `larql_inference::test_utils`.

### kv-cache-benchmark cleanup

After the extraction landed, `crates/kv-cache-benchmark/src/apollo/` still
contained five orphan `.rs` files (`engine.rs`, `store.rs`, `routing.rs`,
`entry.rs`, `npy.rs`) — pre-extraction copies that the `mod.rs` re-export
shim didn't reference but had been kept around. Two `#[ignore]`'d
`real-model`-feature demo tests (`tests/test_apollo_query.rs`,
`tests/test_apollo_accuracy.rs`) called four demo helpers that lived only
in the orphan `engine.rs` (`query_greedy_with_tokenizer`,
`query_greedy_compressed`, `query_generate_compressed`,
`query_generate_uncompressed`); the test build was failing on
`--features real-model` as a result.

All seven files were deleted as part of this cleanup. The
`apollo-demo/apollo11_store` end-to-end harness can be reconstructed from
git history if needed; the underlying functionality (routing, entry
retrieval, boundary-residual injection) is exercised by the surviving
larql-kv apollo unit tests plus the `kv-cache-benchmark` criterion bench.

### Coverage at extraction

After running `cargo llvm-cov --package larql-kv` plus `profiler.rs` test
top-up, total line coverage was **69.82 %** (2 838 / 4 065 lines, 143 unit
tests + 8 new profiler tests). 10 inherited files sat below the 90 %
per-file floor and carried baselines in `coverage-policy.json` that may
only ratchet upward. See [`ROADMAP.md`](ROADMAP.md) for the remediation
list. `make larql-kv-coverage-policy` enforces the baselines.

### Rationale

The four engines collectively share a trait and dispatch but diverge on
state management. Keeping them inside `larql-inference` meant every change
to a single engine recompiled the whole inference crate (transformer
forward pass, mech-interp surface, layer graphs). They are also the
target of independent benchmarking — the `kv-cache-benchmark` crate already
treated them as separable. Splitting tightens the API contract between
"transformer forward" (larql-inference) and "KV state strategy" (larql-kv).

The cut was clean: every primitive engines depend on (`ModelWeights`,
`BackendFfn`, `WalkFfn`, `KvCache`, `forward_*`, `rms_norm_heads`, …) was
already public in larql-inference, so this extraction did not require
designing new API.
