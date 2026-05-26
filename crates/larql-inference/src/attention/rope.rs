//! Rotary Position Embeddings — re-exported from `larql_compute::attention::rope`.
//!
//! The math moved to `larql-compute` (ADR-0022 Step 2d). This shim
//! preserves `crate::attention::rope::*` paths used by sibling modules
//! (`attention/block.rs`, `attention/decode.rs`, `attention/gpu.rs`).

pub use larql_compute::attention::rope::*;
