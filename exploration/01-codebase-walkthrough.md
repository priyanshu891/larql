# LARQL Codebase Walkthrough

## Project Overview

**LARQL** (Lazarus Query Language) is an innovative system that treats transformer model weights as a queryable knowledge graph database.

**Core Premise:** The model *is* the database. Gate vectors become KNN indices. Embeddings become token lookups. Down projections become edge labels. Extract once, query/edit many times via patches.

### Key Philosophy
- No fine-tuning required
- No GPU needed for querying
- Surgical edits via patches overlay immutable base files
- `COMPILE` hardens edits into a new standalone vindex

### Important Documents
- [README.md](../../README.md) — Complete feature guide, quick start, benchmarks
- [AGENTS.md](../../AGENTS.md) — Architectural invariants, workspace layout, contributor guidelines
- [ROADMAP.md](../../ROADMAP.md) — Phase-by-phase development plan (currently on Act 2: MoE distributed inference)

### Three Extraction Levels
These control what operations are enabled on a vindex:
- `browse` (~3 GB) — DESCRIBE, WALK, SELECT only
- `inference` (~6 GB, default) — + INFER operations
- `all` (~10 GB) — + COMPILE and full recompilation

---

## Rust Workspace Layout (14 Crates)

Strict dependency order. Each crate builds on the layers below.

### Core Pipeline Crates

#### 1. `crates/larql-models`
**Role:** Model architecture traits, config parsing, tensor key mappings

Supported models: Gemma/Llama/Mistral/Mixtral/Qwen/Phi/DeepSeek/GPT-2

**Key responsibilities:**
- Define tensor names and layer structure for each model family
- Parse model configs from HuggingFace
- Map logical layer names to physical weight names

#### 2. `crates/larql-compute`
**Role:** Compute backends (CPU/Metal GPU), quantization, MoE dispatch

**Features:**
- Pluggable `ComputeBackend` trait
- Q4K quantization kernels
- BLAS-backed matmul (Apple Accelerate on Mac)
- Metal GPU support for Apple Silicon
- MoE expert dispatch logic

#### 3. `crates/larql-vindex` ⭐
**Role:** Vector index lifecycle — the heart of storage

**Pipeline:**
1. **Extract** — decompose model weights into gate/down/embedding components (streaming)
2. **Load** — mmap-first zero-copy access
3. **Query** — KNN via BLAS
4. **Mutate** — overlay patches without touching base files
5. **Patch** — `.vlp` JSON overlays (INSERT/DELETE/UPDATE operations)
6. **Save** — hardlink base + rewrite columns to bake patches into new vindex

**Storage Format:**
- `gate_vectors.bin` — K-NN index for each layer
- `embeddings.bin` — token lookup table
- `down_meta.bin` — edge label metadata
- `index.json` — config and metadata
- `tokenizer.json` — BPE/Unigram tokenizer
- `weight_manifest.json` — weight file locations
- Optional: `*.vlp` patch overlays (JSON)

#### 4. `crates/larql-inference` ⭐
**Role:** Forward pass, residual stream tracing, KV cache, walk FFN

**Highlights:**
- Fused attention computation
- Q4K GPU decode (Metal on Apple Silicon)
- KV cache management and strategies (markov-bounded, turboquant, graph-walk)
- **Walk FFN:** Sparse KNN (K≈10) over 10K features per layer — **faster than dense** (517ms vs 535ms Gemma 4B)
- Autoregressive generation with beam search
- MoE support: Mixtral, DeepSeek, Gemma 4 hybrid-MoE with CPU/GPU interleave

**Residual Stream Trace:**
- Records every layer's attention vs FFN contribution
- Token trajectory decomposition
- Output formats: binary (`.bin`), boundary-compressed (`.bndx`), context-tiered (`.ctxt`)
- Use case: circuit discovery, trajectory analysis

#### 5. `crates/larql-core`
**Role:** Portable graph algorithms — independent of vindex

**Algorithms:**
- BFS traversal
- PageRank
- Shortest-path
- Graph merge/diff
- Clustering

Can be extracted to a sibling repo (no `larql-*` dependencies).

#### 6. `crates/larql-lql` ⭐
**Role:** LQL surface language — lexer, parser, executor

**20+ statement types:**
- **Browse:** DESCRIBE, WALK, SELECT
- **Inference:** INFER, GENERATE
- **Mutation:** INSERT, DELETE, UPDATE
- **Lifecycle:** EXTRACT, USE, COMPILE
- **Introspection:** SHOW, LIST, STATS
- **Patches:** PATCH, SAVE

**Features:**
- Full REPL with history
- Remote client mode (`RemoteExecutor` via HTTP)
- Parser + Executor separation (can be reused)
- Patch system: `INSERT/DELETE/UPDATE` auto-start `.vlp` JSON overlays
- `COMPILE`: bake patches into new standalone vindex (hardlink + column rewrite)
- **Compose mode** for inserts with cross-fact regression checks

#### 7. `crates/larql-models`
**Role:** Model configs, tensor mappings

Maps model architecture names → layer structure → tensor keys.

#### 8. `crates/larql-cli` ⭐
**Role:** Main entry point — command dispatcher

**Commands:**
```
larql run <model> "prompt"
larql chat <model>
larql extract-index <model> -o <path> --level [browse|inference|all] --quant q4k
larql extract <model> [into] <path> [with level] [with quant]
larql serve <vindex> --port 8080
larql repl
larql lql 'USE ...; DESCRIBE ...'
larql pull hf://org/model
larql link <path>
larql list / show
larql dev walk|compile|...  # research commands
```

### Network & Distributed Crates

#### 9. `crates/larql-server`
**Role:** HTTP/gRPC serving layer

**Endpoints:**
- `POST /v1/describe` — describe entities
- `POST /v1/walk` — walk FFN graph
- `POST /v1/infer` — run forward pass
- `POST /v1/walk-ffn` — remote FFN backend (Phase 0 complete)
- `POST /v1/insert` — **⚠️ known issue: different from LQL**
- WebSocket support for streaming
- Stats, auth hooks, rate limiting

**Modes:**
- `--ffn-only` — remote FFN backend (client runs attention, server runs FFN)

#### 10. `crates/larql-router`
**Role:** Router protocol for distributed MoE dispatch

Part of Phase 1–2 (in progress).

#### 11. `crates/larql-router-protocol`
**Role:** Protocol definitions (gRPC/protobuf)

#### 12. `crates/larql-experts`
**Role:** Placeholder for expert sharding

Phase 2 in roadmap.

### Bindings & Research

#### 13. `crates/larql-python`
**Role:** PyO3 bindings (maturin-built)

Module name: `larql._native`

API:
```python
from larql import WalkModel
model = WalkModel(vindex_path)
model.trace(prompt)       # residual stream trace
model.infer(prompt)       # forward pass
model.describe(entity)    # lookup fact
```

Test: [tests/test_vindex_bindings.py](../../tests/test_vindex_bindings.py)

#### 14. `crates/kv-cache-benchmark`
**Role:** Standalone benchmark for KV cache strategies

Markov-bounded, turboquant, graph-walk comparison.

### Portable/Extractable

#### `crates/model-compute`
**Role:** Bounded compute — native Rust arithmetic/datetime + optional wasmtime WASM host

**Important:** Never imports `larql-*` — designed to be extracted to a sibling repo.

---

## Pre-built Vector Indices

Located at repo root:
- `gemma3-4b.vindex/` — ~6–7 GB, f16, inference-ready
- `gemma3-4b-it.vindex/` — ~6–7 GB, f16, inference-ready (instruction-tuned)

### VIndex File Structure
```
<vindex>/
├── index.json                 # config + metadata
├── tokenizer.json            # BPE tokenizer
├── weight_manifest.json      # pointer to weights
├── gate_vectors.bin          # KNN index
├── embeddings.bin            # token lookup (f16)
├── down_meta.bin             # edge label metadata
├── [optional *.vlp]          # JSON patch overlays (PATCH mode)
└── [optional weight files]   # actual model weights (if all level)
```

---

## Key Subsystems Explained

### A. LLM Inference (Gemma/Llama optimized)

**Path:** [crates/larql-inference/src/](../../crates/larql-inference/src/)

**Highlights:**
- **Fused attention:** single kernel for QKV matmul + softmax
- **KV cache:** three strategies (markov-bounded, turboquant, graph-walk)
- **Walk FFN:** sparse KNN instead of dense forward — K≈10, 10K features/layer
  - Benchmark: 517ms (Walk) vs 535ms (dense) for Gemma 4B — **Walk is faster**
  - Why: MLP is 13K→43K→13K, but only ~10 features fire per token. KNN finds them
- **Metal GPU:** Apple Silicon support via Accelerate
- **MoE:** Mixtral, DeepSeek, Gemma 4 hybrid-MoE support with CPU/GPU interleave
- **Autoregressive:** beam search, temperature, top-k sampling

### B. Vector Indices & Storage

**Path:** [crates/larql-vindex/src/](../../crates/larql-vindex/src/)

**Extraction (streaming):**
1. Download model weights from HF
2. Load into memory in chunks
3. For each layer:
   - Extract gate vector → KNN index
   - Extract embedding matrix → token lookup
   - Cluster down vector → edge label table
4. Write mmap-safe binary files

**Querying:**
- BLAS KNN: find top-K features by cosine similarity to a query vector
- Zero-copy: mmap the binary files, cast to float arrays
- No deserialization overhead

**Mutation & Patching:**
- `INSERT/DELETE/UPDATE` write to `.vlp` JSON overlays (vindex landing page)
- Base files never modified
- On-disk merging: KNN queries union base + overlay results

**COMPILE:**
- Load all patches into memory
- Hardlink base gate/embedding vectors to new vindex
- Rewrite down vectors in-place (new columns) or write delta
- Optionally re-quantize to Q4K
- Single new file: ready to serve

### C. Knowledge Graph & Graph Queries

**Path:** [crates/larql-core/src/](../../crates/larql-core/src/)

**Concepts:**
- Entities: tokens or composite concepts
- Relations: discovered via clustering of down vectors
- Edges: entity → relation → target

**Queries:**
- `DESCRIBE entity` — list all outgoing edges
- `WALK entity` — trace FFN path through knowledge band
- `SHORTEST-PATH entity1 entity2` — BFS through edges
- `PAGERANK` — find central concepts

**Discovery:**
- Relations are inferred from down-vector clustering
- Can be queried with `DESCRIBE entity` to inspect discovered relations
- Named in probes (research experiments)

### D. LQL Query Language

**Path:** [crates/larql-lql/src/](../../crates/larql-lql/src/)

**Parser:** [crates/larql-lql/src/parser/](../../crates/larql-lql/src/parser/)
- Tokenizer + recursive descent parser
- 20+ statement types
- Error recovery and helpful diagnostics

**Executor:** [crates/larql-lql/src/executor/](../../crates/larql-lql/src/executor/)
- Statement dispatch
- Patch composition (INSERT/DELETE/UPDATE)
- Graph queries (DESCRIBE/WALK)
- Inference execution
- COMPILE orchestration

**Composition Mode (Inserts):**
- `INSERT INTO EDGES ... MODE COMPOSE;`
- Runs `balance_installed()` to match layer norms
- Runs `cross_fact_regression_check()` to verify prior inserts don't break
- Backs off `alpha` if regression detected
- Guarantees multi-insert stability

### E. Distributed Inference (Act 2 in progress)

**Current state:**
- Phase 0 (complete): Remote FFN backend via `POST /v1/walk-ffn`
  - Client: run attention (layers 0–N/2), trace residuals, compute gate scores
  - Server: run FFN (sparse KNN), return delta
- Phase 1 (in progress): Per-expert endpoints, Metal GPU on client
- Phase 2 (planned): Expert sharding across servers

**Endpoints:**
- `POST /v1/walk-ffn` — { "residuals": [...], "gate_scores": [...] } → { "delta": [...] }

---

## Testing, Examples, Experiments

### Tests

**Coverage:** 490+ tests across 14 suites

```
cargo test -p larql-lql        # 272 parser/executor tests
cargo test -p larql-vindex     # 104 storage tests
cargo test -p larql-inference  # 109 inference tests (+6 Metal)
cargo test --workspace         # all
```

**Architecture regression suite:**
- Gemma 3/4, Llama 2, Mistral 7B
- CPU + Metal GPU backends

### Examples & Demos

**Path:** [examples/](../../examples/)

- `ffn/` — FFN gate synthesis experiments
- `gemma_4b_knowledge.json` — sample knowledge graph
- `mock_knowledge.json` — test fixtures

### Experiments

**Path:** [experiments/](../../experiments/)

01–07: Self-contained research projects:
- Gate synthesis
- Manifold analysis
- Constellation insertion
- Syntax routing
- Backprop insert
- WASM compute
- (more to come)

Each has detailed findings documented.

### Python Tests

**Path:** [tests/test_vindex_bindings.py](../../tests/test_vindex_bindings.py)

PyO3 binding coverage for the Python API.

---

## Build & Run Commands

### Main Builds

```bash
# Release binary
cargo build --release

# With Metal GPU (Apple Silicon)
cargo build --release --features metal

# Entire workspace
cargo test --workspace
cargo test -p larql-lql --all-features
cargo test -p larql-inference --features metal
```

### Quality Assurance

```bash
make ci        # fmt-check + clippy + test
make fmt       # cargo fmt --all
make lint      # cargo clippy -D warnings
```

### Python

```bash
make python-setup    # create .venv
make python-build    # maturin develop --release
make python-test     # pytest tests/
```

### CLI Operations

```bash
# Extract (default: inference level, f16)
./target/release/larql extract-index google/gemma-3-4b-it \
  -o model.vindex --level inference

# Extract with full level and quantization
./target/release/larql extract google/gemma-3-4b-it \
  into model.vindex with all with q4k

# Inference
./target/release/larql run model.vindex "The capital of France is"

# Chat
./target/release/larql chat model.vindex

# Serve
./target/release/larql serve model.vindex --port 8080

# REPL
./target/release/larql repl

# LQL one-liner
./target/release/larql lql 'USE "model.vindex"; DESCRIBE "France";'

# Download from HF
./target/release/larql pull hf://google/gemma-3-4b-it

# List cached models
./target/release/larql list
```

### Benchmarks (Criterion)

```bash
cargo bench -p larql-lql --bench parser      # parse speed
cargo bench -p larql-vindex --bench vindex_ops  # KNN latency
cargo bench -p larql-inference --bench fwd   # forward pass
cargo bench -p larql-lql --bench compile     # COMPILE speed
```

---

## Core vs. Peripheral Classification

### 🔥 Core (system would not work without)
- **larql-vindex** — lifecycle, extraction, KNN, patch overlay
- **larql-inference** — forward pass, attention, FFN walk, generation
- **larql-lql** — LQL parser/executor, the query interface
- **larql-cli** — primary user entry point (binary)

### 🔧 Peripheral (extensions/utilities)
- **larql-server** — network serving (HTTP/gRPC)
- **larql-python** — language bindings
- **larql-router, larql-experts** — MoE infrastructure (Phase 1–2)
- **model-compute** — portable compute (extractable)
- **kv-cache-benchmark** — research sandbox
- **larql-core** — reusable graph algos (independent of vindex)

---

## Current Development Status

**Recent commits:**
```
ae93375 Merge pull request #45 - virtual-experts
4064bf4 fixed bench
6cb7c33 working on demo script
24cd90f performance improvements for script
c224008 cleanup for script and remote ffn
```

**Current branch:** `main`

**Act 2 Progress:**
- Phase 0 (✅ done): Remote FFN backend
- Phase 1 (🔄 in progress): Per-expert endpoints, Metal GPU on client
- Phase 2 (📋 planned): Expert sharding across servers

---

## Key Architectural Insight

The system achieves its speed by treating transformer weights *structurally*:

1. **Gate vectors** become a searchable KNN index
2. **Down projections** become edge labels in a knowledge graph
3. **Inference becomes a sparse graph walk** instead of dense computation
4. **Patches overlay immutable base files** — no recomputation needed until COMPILE
5. **COMPILE hardens edits** into a new vindex via hardlinking + selective rewrite

This is fundamentally different from fine-tuning: **no GPU, no gradients, just surgical edits to weight structure**.
