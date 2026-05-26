//! GPU-accelerated attention dispatch — moved to
//! `larql_compute::attention::gpu` (ADR-0022 Step 2e). This shim
//! preserves `crate::attention::gpu::*` paths.

pub use larql_compute::attention::gpu::*;
