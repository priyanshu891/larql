//! Test fixtures for compute-side tests that need a real `KvIndex`
//! implementation backed by Q4_K-quantized bytes.
//!
//! Gated behind the `test-utils` feature so production builds never
//! compile the fixture code. Enabled from a consumer's
//! `[dev-dependencies]` entry, e.g.
//! `larql-compute = { path = "../larql-compute", features = ["test-utils"] }`.
//!
//! ## Why this lives in `larql-compute`
//!
//! The `KvIndex` trait is defined here, so the fixture's `impl KvIndex`
//! is an in-crate impl (no orphan-rule issues). Downstream test code
//! in `larql-compute` itself and `larql-compute-metal` can both use it
//! without a `larql-vindex` dev-dep (vindex itself impls `KvIndex` on
//! `VectorIndex`, but it depends on `larql-compute` — pulling vindex in
//! as a dev-dep would create a back-edge that's better avoided when a
//! ~150-LOC standalone fixture works just as well).

use std::collections::HashMap;
use std::sync::Arc;

use larql_models::ModelWeights;

use crate::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};
use crate::kv_index::{KvIndex, FFN_COMPONENTS_PER_LAYER};

/// Per-(layer, component) dequantised FFN block — lazily populated on first
/// request through `kquant_ffn_layer_once`. Aliased here only to keep the
/// containing struct under clippy's `type_complexity` threshold.
type FfnDequantCache = std::sync::Mutex<HashMap<(usize, usize), Arc<Vec<f32>>>>;

/// `KvIndex` backed by Q4_K-quantized weight tensors held in
/// in-process memory. Drives `kquant_forward::fused_*` and the
/// `coarse_*` paths on `KvDispatch` impls end-to-end without
/// constructing a full `VectorIndex`.
///
/// Construct via [`make_q4k_fixture_index`].
pub struct Q4kFixtureIndex {
    /// Concatenated Q4_K bytes for FFN gate/up/down across all layers,
    /// laid out as `[layer 0: gate, up, down; layer 1: gate, up, down; ...]`.
    /// `interleaved_kquant_mmap_ref` returns this whole slice;
    /// `interleaved_kquant_layer_data` slices into it at the per-layer
    /// offset.
    ffn_mmap: Vec<u8>,
    /// Per-component byte count: `Q4_K::packed_matrix_bytes(intermediate, hidden)`.
    /// Same value for every (layer, component) at this fixture scale.
    ffn_per_matrix: usize,
    /// Concatenated Q4_K bytes for attention Q/K/V/O across all layers,
    /// laid out as `[layer 0: Q, K, V, O; layer 1: Q, K, V, O; ...]`.
    attn_mmap: Vec<u8>,
    /// Per-layer (offset, length) pairs for Q/K/V/O in `attn_mmap`. Q/K/V/O
    /// have different shapes (q_dim vs kv_dim) so the offsets aren't a
    /// fixed stride.
    attn_offsets: Vec<[(usize, usize); 4]>,
    /// Per-(layer, component) dequantised FFN cache populated lazily
    /// on first request through `kquant_ffn_layer_once`.
    ffn_cache: FfnDequantCache,
    /// Intermediate dimension — `num_features` returns this.
    intermediate: usize,
    /// Vocabulary size — `vocab_size` returns this.
    vocab_size: usize,
    /// When `true`, `kquant_ffn_layer_once` returns `None` unconditionally
    /// so callers take the dequant-from-bytes fallback path. Default
    /// `false` (lazy cache enabled). Flipped on by
    /// [`Q4kFixtureIndex::without_ffn_cache`] for tests that need to
    /// drive both branches.
    disable_ffn_cache: bool,
    /// When `true`, the trait method `interleaved_kquant_mmap_ref`
    /// returns None and `interleaved_q4_mmap_ref` returns the bytes
    /// — drives the Q4_0 fallback branch in `fused_prefill`.
    use_legacy_q4_mmap: bool,
}

impl Q4kFixtureIndex {
    /// Disable the lazy dequant cache. Subsequent
    /// `kquant_ffn_layer_once` calls always return `None`, forcing
    /// callers down the `dequantize_matrix` path. Used to test the
    /// fallback branch of `kquant_ffn_forward_layer`.
    pub fn without_ffn_cache(mut self) -> Self {
        self.disable_ffn_cache = true;
        self
    }

    /// Swap the FFN mmap accessor to return `interleaved_q4_mmap_ref`
    /// (Q4_0 legacy format) instead of `interleaved_kquant_mmap_ref`.
    /// Drives the Q4_0 fallback branch in `fused_prefill` /
    /// `fused_decode_step_inner` that picks between the two mmap
    /// accessors. The underlying bytes stay Q4_K-quantized — the
    /// branch just records `ffn_is_q4k = false` and tags the format
    /// downstream.
    pub fn as_legacy_q4_mmap(mut self) -> Self {
        self.use_legacy_q4_mmap = true;
        self
    }
}

impl KvIndex for Q4kFixtureIndex {
    fn num_features(&self, _layer: usize) -> usize {
        self.intermediate
    }

    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let offsets = self.attn_offsets.get(layer)?;
        let attn = &self.attn_mmap;
        Some([
            (&attn[offsets[0].0..offsets[0].0 + offsets[0].1], "Q4_K"),
            (&attn[offsets[1].0..offsets[1].0 + offsets[1].1], "Q4_K"),
            (&attn[offsets[2].0..offsets[2].0 + offsets[2].1], "Q4_K"),
            (&attn[offsets[3].0..offsets[3].0 + offsets[3].1], "Q4_K"),
        ])
    }

    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
        let per_matrix = self.ffn_per_matrix;
        let layer_start = layer * per_matrix * FFN_COMPONENTS_PER_LAYER;
        let mmap = &self.ffn_mmap;
        if layer_start + FFN_COMPONENTS_PER_LAYER * per_matrix > mmap.len() {
            return None;
        }
        Some([
            (&mmap[layer_start..layer_start + per_matrix], "Q4_K"),
            (
                &mmap[layer_start + per_matrix..layer_start + 2 * per_matrix],
                "Q4_K",
            ),
            (
                &mmap[layer_start + 2 * per_matrix..layer_start + 3 * per_matrix],
                "Q4_K",
            ),
        ])
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        if self.use_legacy_q4_mmap {
            return None;
        }
        Some(&self.ffn_mmap)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        if self.use_legacy_q4_mmap {
            Some(&self.ffn_mmap)
        } else {
            None
        }
    }

    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        if component >= FFN_COMPONENTS_PER_LAYER {
            return None;
        }
        if self.disable_ffn_cache {
            return None;
        }
        let mut cache = self.ffn_cache.lock().ok()?;
        if let Some(cached) = cache.get(&(layer, component)) {
            return Some(Arc::clone(cached));
        }
        let per_matrix = self.ffn_per_matrix;
        let layer_start = layer * per_matrix * FFN_COMPONENTS_PER_LAYER;
        let comp_start = layer_start + component * per_matrix;
        let comp_end = comp_start + per_matrix;
        if comp_end > self.ffn_mmap.len() {
            return None;
        }
        let bytes = &self.ffn_mmap[comp_start..comp_end];
        // Component-major: gate/up are [intermediate × hidden]; down is
        // [hidden × intermediate]. Element count is the same (`intermediate × hidden`).
        let n_elements = per_matrix / 144 * 256; // Q4_K block: 144 bytes / 256 elements
        let arc = Arc::new(dequantize_q4_k(bytes, n_elements));
        cache.insert((layer, component), Arc::clone(&arc));
        Some(arc)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Build a [`Q4kFixtureIndex`] from `weights`, quantizing every
/// per-layer Q/K/V/O and gate/up/down tensor to Q4_K bytes. Pair with
/// [`larql_models::test_fixtures::make_test_q4k_weights`] (or its
/// SiLU sibling) to satisfy the Q4_K-shape constraint that every
/// dimension be a multiple of `K_QUANT_BLOCK_ELEMS` (256).
///
/// Panics if any expected tensor key is missing or non-contiguous —
/// both are bugs in the calling weight fixture, not user-visible.
pub fn make_q4k_fixture_index(weights: &ModelWeights) -> Q4kFixtureIndex {
    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let intermediate = weights.intermediate_size;
    let vocab_size = weights.vocab_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_mmap: Vec<u8> = Vec::new();
    let mut attn_offsets: Vec<[(usize, usize); 4]> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let mut layer_offsets: [(usize, usize); 4] = [(0, 0); 4];
        for (i, key) in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ]
        .iter()
        .enumerate()
        {
            let bytes = q4k_for(key);
            let offset = attn_mmap.len();
            let length = bytes.len();
            attn_mmap.extend_from_slice(&bytes);
            layer_offsets[i] = (offset, length);
        }
        attn_offsets.push(layer_offsets);
    }

    let mut ffn_mmap: Vec<u8> = Vec::new();
    let mut ffn_per_matrix = 0;
    for layer in 0..num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let bytes = q4k_for(&key);
            // Pin a single per-matrix size — every component in every
            // layer must produce the same byte length for the
            // contiguous-mmap layout to make sense.
            if ffn_per_matrix == 0 {
                ffn_per_matrix = bytes.len();
            } else {
                assert_eq!(
                    bytes.len(),
                    ffn_per_matrix,
                    "Q4_K per-matrix size drifted across (layer, component)"
                );
            }
            ffn_mmap.extend_from_slice(&bytes);
        }
    }

    Q4kFixtureIndex {
        ffn_mmap,
        ffn_per_matrix,
        attn_mmap,
        attn_offsets,
        ffn_cache: std::sync::Mutex::new(HashMap::new()),
        intermediate,
        vocab_size,
        disable_ffn_cache: false,
        use_legacy_q4_mmap: false,
    }
}

/// Minimal `ComputeBackend` that overrides the `DecodeBackend` methods
/// `kquant_forward::cached::fused_*` reaches: `supports_quant(Q4_K)`,
/// `prefill_kquant`, `decode_token{,_with_state_dump}`. Each override
/// returns a synthetic zero vector of the right shape so the wrappers
/// can run their post-call shape-and-slice logic without
/// short-circuiting. End-to-end *correctness* of those kernels lives
/// in `MetalBackend` integration tests; this mock exists only to drive
/// coverage of the `kquant_forward` glue code.
pub struct MockKquantBackend;

impl crate::MatMul for MockKquantBackend {
    fn matmul(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("mock MatMul never invoked")
    }
    fn matmul_transb(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("mock MatMul never invoked")
    }
}

impl crate::QuantMatVec for MockKquantBackend {
    fn supports_quant(&self, format: crate::QuantFormat) -> bool {
        matches!(format, crate::QuantFormat::Q4_K)
    }
}

impl crate::DecodeBackend for MockKquantBackend {
    fn prefill_kquant(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        Some(vec![0.0; seq_len * hidden])
    }

    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        Some(vec![0.0; hidden])
    }

    fn decode_token_with_state_dump_masked(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut crate::DecodeStateDump>,
        mask: crate::StateDumpMask,
    ) -> Option<Vec<f32>> {
        if let Some(dump) = state {
            let want_kv = matches!(mask, crate::StateDumpMask::Full);
            let want_h = !matches!(mask, crate::StateDumpMask::None);
            for layer in layers {
                if want_h {
                    dump.h_in_per_layer.push(vec![0.0; hidden]);
                }
                if want_kv {
                    let kv_dim = layer.num_kv_heads * layer.head_dim;
                    dump.k_new_per_layer.push(vec![0.0; kv_dim]);
                    dump.v_new_per_layer.push(vec![0.0; kv_dim]);
                }
            }
        }
        self.decode_token(layers, x, hidden, inter)
    }
}

impl crate::ComputeBackend for MockKquantBackend {
    fn name(&self) -> &str {
        "mock-kquant"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn supports(&self, cap: crate::Capability) -> bool {
        matches!(cap, crate::Capability::QuantMatVec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_q4k_weights;

    /// Smoke test: build the fixture index from Q4K-friendly weights
    /// and verify every accessor returns sensible data.
    #[test]
    fn fixture_index_returns_per_layer_q4k_slices() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);

        // Trait accessors.
        assert_eq!(idx.num_features(0), weights.intermediate_size);
        assert_eq!(idx.vocab_size(), weights.vocab_size);

        // Per-layer Q4K attention bytes.
        let attn = idx.attn_kquant_layer_data(0).expect("layer 0 attn");
        assert_eq!(attn.len(), 4);
        for (bytes, fmt) in &attn {
            assert_eq!(*fmt, "Q4_K");
            assert!(!bytes.is_empty(), "empty Q4K bytes");
        }

        // Per-layer FFN data slices into the mmap.
        let ffn = idx.interleaved_kquant_layer_data(0).expect("layer 0 ffn");
        assert_eq!(ffn.len(), FFN_COMPONENTS_PER_LAYER);
        let mmap = idx.interleaved_kquant_mmap_ref().expect("mmap");
        assert!(!mmap.is_empty());

        // Out-of-range layer returns None.
        assert!(idx.attn_kquant_layer_data(weights.num_layers).is_none());
        assert!(idx
            .interleaved_kquant_layer_data(weights.num_layers)
            .is_none());

        // Dequantised cache populates on demand.
        let cached0 = idx.kquant_ffn_layer_once(0, 0).expect("layer 0 gate cache");
        assert!(!cached0.is_empty());
        // Second call returns the same Arc.
        let cached0_again = idx.kquant_ffn_layer_once(0, 0).expect("hit cache");
        assert!(Arc::ptr_eq(&cached0, &cached0_again));

        // Out-of-range component returns None.
        assert!(idx.kquant_ffn_layer_once(0, 99).is_none());

        // Legacy Q4_0 mmap not provided — default `None`.
        assert!(idx.interleaved_q4_mmap_ref().is_none());
    }

    #[test]
    fn fixture_drives_fused_prefill_to_some_on_mock_backend() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let result = crate::kquant_forward::fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        let h = result.expect("MockKquantBackend.prefill_kquant returns Some");
        // `fused_prefill` slices to the last position → shape `[1 × hidden]`.
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn fixture_drives_fused_decode_step_to_some_on_mock_backend() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let result = crate::kquant_forward::fused_decode_step(&weights, &idx, 0u32, &backend);
        let h = result.expect("MockKquantBackend.decode_token returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn fixture_drives_fused_decode_step_with_state_to_some() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
        let result = crate::kquant_forward::fused_decode_step_with_state(
            &weights, &idx, 0u32, &backend, &mut dump,
        );
        let h = result.expect("decode_step_with_state returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // The mock populates per-layer dump entries.
        assert_eq!(dump.h_in_per_layer.len(), weights.num_layers);
    }

    #[test]
    fn fixture_satisfies_fused_prefill_input_gates() {
        use crate::QuantMatVec;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        assert!(QuantMatVec::supports_quant(
            &backend,
            crate::QuantFormat::Q4_K
        ));
        assert!(idx.interleaved_kquant_mmap_ref().is_some());
        assert!(idx.attn_kquant_layer_data(0).is_some());
        assert!(idx.num_features(0) > 0);
        let result = crate::kquant_forward::fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        // CpuBackend's `prefill_kquant` default returns None, so the
        // chain bottoms out there — but every gate above passes.
        assert!(result.is_none());
    }
}
