//! Raw-logits forward passes — moved to
//! `larql_compute::forward::predict::raw` (ADR-0022 Step 2f). This
//! shim preserves `crate::forward::predict::raw::*` and
//! `crate::forward::{forward_raw_logits, forward_from_layer, RawForward}`
//! paths used by target-delta optimisation, Apollo, layer_graph, and
//! engines (`kv_dispatch::cpu`, `markov_residual_codec`).

pub use larql_compute::forward::predict::raw::*;
