//! Prediction-related types — moved to
//! `larql_compute::forward::predict::types` (ADR-0022 follow-up).
//! This shim preserves `crate::forward::predict::types::*` paths used
//! by `forward/trace.rs`, `forward/predict/{dense,ffn}.rs`, and
//! external consumers.

pub use larql_compute::forward::predict::types::*;
