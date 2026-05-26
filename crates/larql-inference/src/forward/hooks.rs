//! Mid-forward hook system — moved to `larql_compute::forward::hooks`
//! (ADR-0022 Step 2e2). This shim preserves `crate::forward::hooks::*`
//! paths used by `trace/`, `predict/`, and external test/example
//! crates.

pub use larql_compute::forward::hooks::*;
