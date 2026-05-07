//! `ModelSlot` — a lazy-loadable handle around a vindex.
//!
//! Each vindex known to the server is represented as a `ModelSlot`.  A slot
//! always carries the cheap metadata (path + parsed `index.json`); the
//! expensive resident weights only show up once the slot is in the `Loaded`
//! state.
//!
//! Slots exist in four states:
//!
//! - `Discovered` — metadata is known, nothing is mapped.  New slots found by
//!   scanning a workspace directory land here.
//! - `Loading`    — a load is in flight.  Concurrent callers piggy-back on
//!   the in-flight load via a broadcast channel rather than each kicking off
//!   their own `spawn_blocking`.
//! - `Loaded`     — weights resident; requests served immediately.
//! - `Failed`     — the last load errored; the message is kept so the UI can
//!   show it.  A subsequent caller retries automatically.
//!
//! The eager-boot path (single-path CLI invocation) uses
//! [`ModelSlot::from_loaded`] to construct a slot that is already `Loaded`
//! — skipping the state machine entirely.
//!
//! Ported from the sibling `larql` repo, adapted to this repo's
//! `LoadedModel` shape and `bootstrap::load_single_vindex` entry point.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_vindex::{load_vindex_config, VindexConfig};
use tokio::sync::{broadcast, RwLock};

use crate::state::{model_id_from_name, LoadedModel};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Result type a slot-load publishes to concurrent waiters. `Arc<LoadedModel>`
/// is cheap to clone on the happy path; `String` is cheap on the failure path.
type LoadResult = Result<Arc<LoadedModel>, String>;

/// Per-slot load options, captured at startup from CLI flags so the same
/// options apply on re-load after unload.
///
/// This is a minimal superset of what `bootstrap::LoadVindexOptions` needs —
/// lazy-loaded slots reuse the shared `load_single_vindex` helper.
#[derive(Clone, Default)]
pub struct SlotLoadOpts {
    pub no_infer: bool,
    pub ffn_only: bool,
    pub embed_only: bool,
    pub layer_range: Option<(usize, usize)>,
    pub max_gate_cache_layers: usize,
    pub max_q4k_cache_layers: usize,
    pub hnsw: Option<usize>,
    pub warmup_hnsw: bool,
    pub release_mmap_after_request: bool,
    pub expert_filter: Option<(usize, usize)>,
    pub unit_filter:
        Option<Arc<std::collections::HashSet<(usize, usize)>>>,
    pub moe_remote: Option<Arc<larql_inference::ffn::RemoteMoeBackend>>,
}

/// Snapshot of a slot's current state — serialized into `/v1/vindexes`.
#[derive(Clone, Debug, serde::Serialize, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SlotState {
    Discovered,
    Loading,
    Loaded,
    Failed,
}

/// Internal state.  `Loading` carries a broadcast sender so late arrivals
/// can subscribe and await the owner's result instead of each spawning
/// their own blocking load.
enum SlotInner {
    Discovered,
    Loading(broadcast::Sender<LoadResult>),
    Loaded(Arc<LoadedModel>),
    Failed(String),
}

impl SlotInner {
    fn state(&self) -> SlotState {
        match self {
            SlotInner::Discovered => SlotState::Discovered,
            SlotInner::Loading(_) => SlotState::Loading,
            SlotInner::Loaded(_) => SlotState::Loaded,
            SlotInner::Failed(_) => SlotState::Failed,
        }
    }
}

/// A single vindex known to the server.  Metadata is cheap to hold; weights
/// load on demand via [`Self::get_or_load`].
pub struct ModelSlot {
    /// Short id — `model_id_from_name(config.model)`.
    pub id: String,
    /// Vindex directory on disk.
    pub path: PathBuf,
    /// Parsed `index.json`.  Present even when unloaded.
    pub config: VindexConfig,
    /// Total directory size in bytes, computed at discovery.
    pub size_bytes: u64,
    /// Load options snapshot, used when lazy-loading.
    pub opts: SlotLoadOpts,
    inner: RwLock<SlotInner>,
    /// Current load stage ("loading tokenizer", ...), populated while in
    /// the `Loading` state by the worker via [`Self::set_progress`].
    load_progress: std::sync::Mutex<Option<String>>,
}

impl ModelSlot {
    /// Construct a slot by reading `index.json`.  No weights touched.
    pub fn discover(path: PathBuf, opts: SlotLoadOpts) -> Result<Self, BoxError> {
        let config = load_vindex_config(&path)?;
        let id = model_id_from_name(&config.model);
        let size_bytes = dir_size(&path);
        Ok(ModelSlot {
            id,
            path,
            config,
            size_bytes,
            opts,
            inner: RwLock::new(SlotInner::Discovered),
            load_progress: std::sync::Mutex::new(None),
        })
    }

    /// Construct a slot already in the `Loaded` state — used for the eager
    /// boot path where the CLI loaded the vindex up front.
    pub fn from_loaded(loaded: Arc<LoadedModel>) -> Self {
        let id = loaded.id.clone();
        let path = loaded.path.clone();
        let config = loaded.config.clone();
        let size_bytes = dir_size(&path);
        let opts = SlotLoadOpts {
            no_infer: loaded.infer_disabled && !loaded.ffn_only && !loaded.embed_only,
            ffn_only: loaded.ffn_only,
            embed_only: loaded.embed_only,
            release_mmap_after_request: loaded.release_mmap_after_request,
            expert_filter: loaded.expert_filter,
            unit_filter: loaded.unit_filter.clone(),
            moe_remote: loaded.moe_remote.clone(),
            ..SlotLoadOpts::default()
        };
        ModelSlot {
            id,
            path,
            config,
            size_bytes,
            opts,
            inner: RwLock::new(SlotInner::Loaded(loaded)),
            load_progress: std::sync::Mutex::new(None),
        }
    }

    pub async fn state(&self) -> SlotState {
        self.inner.read().await.state()
    }

    pub async fn last_error(&self) -> Option<String> {
        match &*self.inner.read().await {
            SlotInner::Failed(e) => Some(e.clone()),
            _ => None,
        }
    }

    pub fn progress(&self) -> Option<String> {
        self.load_progress.lock().ok().and_then(|g| g.clone())
    }

    fn set_progress(&self, stage: Option<String>) {
        if let Ok(mut g) = self.load_progress.lock() {
            *g = stage;
        }
    }

    /// Return the loaded model if weights are resident; never triggers a
    /// load.  Used by accessors that must not block.
    pub async fn loaded(&self) -> Option<Arc<LoadedModel>> {
        if let SlotInner::Loaded(ref m) = *self.inner.read().await {
            Some(Arc::clone(m))
        } else {
            None
        }
    }

    /// Sync variant of [`Self::loaded`].  Returns the loaded model if the
    /// inner `tokio::sync::RwLock` can be acquired without waiting, which
    /// is the steady-state case for already-loaded slots — the only
    /// contenders are the millisecond-scale state transitions during
    /// load/unload, and the backwards-compatible accessors that call this
    /// (`AppState::model`, `AppState::model_or_err`) are happy to report
    /// `None` during those windows.
    pub fn loaded_unchecked(&self) -> Option<Arc<LoadedModel>> {
        let guard = self.inner.try_read().ok()?;
        match &*guard {
            SlotInner::Loaded(m) => Some(Arc::clone(m)),
            _ => None,
        }
    }

    /// Load weights on first access; concurrent callers wait on the same
    /// result instead of each spawning their own blocking load.
    pub async fn get_or_load(self: &Arc<Self>) -> Result<Arc<LoadedModel>, String> {
        // Fast path — already loaded, or a load is in flight.
        {
            let guard = self.inner.read().await;
            match &*guard {
                SlotInner::Loaded(m) => return Ok(Arc::clone(m)),
                SlotInner::Loading(tx) => {
                    let mut rx = tx.subscribe();
                    drop(guard);
                    return rx
                        .recv()
                        .await
                        .unwrap_or_else(|_| Err("load task dropped without publishing result".into()));
                }
                _ => {}
            }
        }

        // Slow path — claim ownership under the write lock.
        let tx = {
            let mut guard = self.inner.write().await;
            match &*guard {
                SlotInner::Loaded(m) => return Ok(Arc::clone(m)),
                SlotInner::Loading(tx) => {
                    let mut rx = tx.subscribe();
                    drop(guard);
                    return rx
                        .recv()
                        .await
                        .unwrap_or_else(|_| Err("load task dropped without publishing result".into()));
                }
                _ => {
                    let (tx, _rx) = broadcast::channel::<LoadResult>(1);
                    *guard = SlotInner::Loading(tx.clone());
                    tx
                }
            }
        };

        self.run_owned_load(tx).await
    }

    async fn run_owned_load(
        self: &Arc<Self>,
        tx: broadcast::Sender<LoadResult>,
    ) -> Result<Arc<LoadedModel>, String> {
        let path = self.path.clone();
        let opts = self.opts.clone();
        let slot_for_progress = Arc::clone(self);
        let join = tokio::task::spawn_blocking(move || {
            slot_for_progress.set_progress(Some("loading".into()));
            let bopts = crate::bootstrap::LoadVindexOptions {
                no_infer: opts.no_infer,
                ffn_only: opts.ffn_only,
                embed_only: opts.embed_only,
                layer_range: opts.layer_range,
                max_gate_cache_layers: opts.max_gate_cache_layers,
                max_q4k_cache_layers: opts.max_q4k_cache_layers,
                hnsw: opts.hnsw,
                warmup_hnsw: opts.warmup_hnsw,
                release_mmap_after_request: opts.release_mmap_after_request,
                expert_filter: opts.expert_filter,
                unit_filter: opts.unit_filter,
                moe_remote: opts.moe_remote,
            };
            crate::bootstrap::load_single_vindex(path.to_string_lossy().as_ref(), bopts)
        })
        .await;

        self.set_progress(None);

        let result: LoadResult = match join {
            Ok(Ok(loaded)) => Ok(Arc::new(loaded)),
            Ok(Err(e)) => Err(e.to_string()),
            Err(e) => Err(format!("load task panicked: {e}")),
        };

        {
            let mut guard = self.inner.write().await;
            match &result {
                Ok(arc) => *guard = SlotInner::Loaded(Arc::clone(arc)),
                Err(e) => *guard = SlotInner::Failed(e.clone()),
            }
        }

        let _ = tx.send(result.clone());
        result
    }

    /// Drop resident weights.  A subsequent `get_or_load` will re-load.
    pub async fn unload(&self) {
        *self.inner.write().await = SlotInner::Discovered;
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}
