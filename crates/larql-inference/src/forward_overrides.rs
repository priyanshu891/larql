//! Forward-path override surface — moved to `larql_compute::forward_overrides`
//! (ADR-0022 Step 2e). This shim preserves `crate::forward_overrides::*`
//! paths used across the inference crate (residual norm, attention RoPE,
//! layer_graph dispatch).

pub use larql_compute::forward_overrides::*;
