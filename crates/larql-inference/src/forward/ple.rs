//! Per-Layer Embeddings — moved to `larql_compute::forward::ple`
//! (ADR-0022 Step 2e2). This shim preserves `crate::forward::ple::*`
//! paths used by `forward/layer.rs`, `forward/predict/*`, and tests.

pub use larql_compute::forward::ple::*;
