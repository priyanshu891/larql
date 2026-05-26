//! Dense FFN backend — moved to `larql_compute::ffn::weight`
//! (ADR-0022 Step 2f). The substrate-level dense impl belongs in
//! compute alongside `forward/predict/raw.rs`'s `forward_from_layer`,
//! which constructs `WeightFfn` directly. This shim preserves
//! `crate::ffn::weight::*` paths used by `forward/layer.rs` tests,
//! `examples/`, and external test/example crates.

pub use larql_compute::ffn::weight::*;
