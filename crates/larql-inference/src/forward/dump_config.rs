//! Diagnostic per-layer/per-stage dump env-var accessor — moved to
//! `larql_compute::forward::dump_config` (ADR-0022 Step 2e) so the
//! attention spine can reach it from compute. This shim preserves
//! `crate::forward::dump_config::*` paths.

pub use larql_compute::forward::dump_config::*;
