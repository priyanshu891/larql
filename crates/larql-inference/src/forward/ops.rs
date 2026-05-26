//! Small math utilities — re-exported from `larql_compute::forward`.
//!
//! Step 2e moved `apply_norm` down once `forward_overrides` followed.
//! This shim preserves `crate::forward::{apply_norm, add_bias,
//! dot_proj, softmax}` paths used across `forward/`, `attention/`,
//! `vindex/`, `layer_graph/`, and external crates.

pub use larql_compute::forward::{add_bias, apply_norm, dot_proj, softmax};
