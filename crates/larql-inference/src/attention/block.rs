//! CPU attention block — moved to `larql_compute::attention::block`
//! (ADR-0022 Step 2e). This shim preserves `crate::attention::block::*`
//! paths used by `layer_executor/`, `vindex/kquant_forward/`, and
//! tests + examples.

pub use larql_compute::attention::block::*;
