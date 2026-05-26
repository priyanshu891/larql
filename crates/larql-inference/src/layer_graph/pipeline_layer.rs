//! `FullPipelineLayer` construction — moved to
//! `larql_compute::pipeline_layer` (ADR-0022 Step 7). This shim
//! preserves `crate::layer_graph::pipeline_layer::*` paths used by
//! `layer_graph/{generate/gpu_setup, grid/setup}.rs` and tests.

pub use larql_compute::pipeline_layer::*;
