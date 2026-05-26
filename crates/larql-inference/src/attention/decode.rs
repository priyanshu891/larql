//! Per-step KV-cached attention decode — moved to
//! `larql_compute::attention::decode` (ADR-0022 Step 2e). This shim
//! preserves `crate::attention::decode::*` paths used by
//! `kv_dispatch/cpu.rs`, `layer_executor/local_walk.rs`, and others.

pub use larql_compute::attention::decode::*;
