# Exploration Directory Index

This directory contains a comprehensive analysis of the LARQL codebase and known issues discovered during exploration.

## Files

### 1. [01-codebase-walkthrough.md](01-codebase-walkthrough.md)
**Complete codebase overview**

- Project philosophy and core concepts
- 14-crate workspace layout with dependencies
- Role of each crate (core vs. peripheral)
- Pre-built vector indices
- Key subsystems (inference, storage, query language, distributed)
- Testing, examples, experiments
- Build and run commands
- Development status and roadmap

**Use this when:** You need to understand the overall architecture, navigate the codebase, or onboard new team members.

### 2. [02-extract-command-differences.md](02-extract-command-differences.md)
**Why CLI extract (~8 GB) vs. LQL extract (~17 GB) produce different sizes**

- Root cause: Different extraction pipelines
- Gate vector pruning differences (CLI: top-100, LQL: all)
- File format differences and storage overhead
- Relation discovery differences (heuristic vs. hierarchical clustering)
- Detailed size breakdown per component
- Design philosophy behind each approach
- Recommendations for which to use

**Use this when:** You're confused about why two extraction commands produce different output sizes, or deciding which extraction method to use for your workflow.

### 3. [03-api-vs-lql-insert-differences.md](03-api-vs-lql-insert-differences.md)
**Critical: Why HTTP API inserts cause poor results and leakage, while LQL inserts are stable** ⚠️

- Problem: API produces 2.5× stronger installations
- Root causes:
  - Alpha parameter too high (0.25 vs. 0.1)
  - Missing norm balancing
  - Missing cross-fact regression checks
- Why leakage occurs (detailed mechanism)
- Multi-insert degradation proof
- Recommended fixes ranked by impact
- Verification checklist

**Use this when:** You're experiencing poor results from HTTP `/v1/insert` API, facts leaking to other entities, or degradation after multiple inserts. **This is a known bug with three recommended fixes.**

---

## Quick Facts

### Codebase
- **Language:** Rust
- **Size:** 14 interdependent crates
- **Test coverage:** 490+ tests
- **Core:** `larql-vindex`, `larql-inference`, `larql-lql`, `larql-cli`
- **Roadmap:** Act 2 — distributed MoE inference (Phase 1 in progress)

### Known Issues
1. **HTTP API inserts** produce poor results due to 2.5× excessive alpha + missing guardrails
   - Fix: Route through LQL executor or inline guardrails
   - Severity: High (production impact)
   - Status: Documented, fixes proposed

### Key Insights
1. Extract commands use different pipelines — CLI for serving, LQL for research
2. LQL insert ("MODE COMPOSE") is the safe path for multi-insert workloads
3. The system treats weight structure as a queryable graph — not traditional fine-tuning
4. Norm balancing and regression checks are critical for multi-fact stability

---

## For Next Steps

1. **If investigating insert issues:** Read `03-api-vs-lql-insert-differences.md` and implement Option 1 fix
2. **If evaluating extract sizes:** Read `02-extract-command-differences.md`
3. **If onboarding to the project:** Start with `01-codebase-walkthrough.md`

---

## Navigation

- **Codebase root:** `/Users/priyanshu.rai/Documents/Research-Lab/Projects_2026/graph-ai/larql-api/`
- **Crates:** `./crates/`
- **Tests:** `./tests/`, `cargo test --workspace`
- **Binary:** `./target/release/larql` (after `cargo build --release`)

---

## Last Updated

Generated: 2026-05-06
Codebase state: main branch, recent commit `ae93375` (virtual-experts merge)
