//! kquant-aware forward hook helper — moved to
//! `larql_compute::kquant_forward::hooks` (ADR-0022 follow-up).
//! This shim preserves `crate::vindex::kquant_forward::hooks::*`
//! paths used by inference tracers/CLI.

pub use larql_compute::kquant_forward::predict_kquant_hidden_hooked;
