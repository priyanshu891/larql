//! Grouped-Query Attention — re-exported from `larql_compute::attention::gqa`.
//!
//! The math moved to `larql-compute` (ADR-0022 Step 2d). This shim
//! preserves `crate::attention::gqa::*` paths used by sibling modules
//! (`attention/block.rs`, `attention/gpu.rs`).

pub use larql_compute::attention::gqa::*;
