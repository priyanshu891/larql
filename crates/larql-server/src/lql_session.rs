//! Per-HTTP-session LQL executor management.
//!
//! Each `X-Session-Id` (plus a single shared `"__global__"` bucket for
//! header-less requests) owns a persistent `larql_lql::Session`. The
//! session carries cross-fact state used by `INSERT`'s Compose pipeline
//! — `installed_edges`, `raw_install_residuals`, `decoy_residual_cache`
//! — so multi-insert workloads get the `balance_installed` +
//! `cross_fact_regression_check` guarantees that a fresh session can't
//! provide.
//!
//! The server's `PatchedVindex` is swapped into the LQL session's
//! backend for the duration of each insert (see
//! `larql_lql::Session::swap_patched_vindex`), then swapped back out.
//! Mutations land on the real server state; the LQL session's
//! accumulators live on the `Session` struct itself and are untouched
//! by the swap.
//!
//! Sessions are created lazily: `get_or_create` calls `USE "<path>"` on
//! first access so the session knows the vindex path (needed by the
//! Compose pipeline's tokenizer / weights loaders).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use larql_lql::Session;
use larql_lql::ast::{Statement, UseTarget};

/// A single slot in the session manager — wraps a `Session` behind a
/// `Mutex` so individual HTTP sessions serialise their own inserts
/// without blocking each other.
pub type SharedLqlSession = Arc<Mutex<Session>>;

/// Manages per-HTTP-session `larql_lql::Session` instances.
pub struct LqlSessionManager {
    sessions: RwLock<HashMap<String, SharedLqlSession>>,
}

impl LqlSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Get the existing LQL session for `key` or create a fresh one
    /// bootstrapped against `vindex_path` via `USE "<path>"`.
    ///
    /// `key` is the `X-Session-Id` header value, or `"__global__"` for
    /// header-less requests.
    pub fn get_or_create(
        &self,
        key: &str,
        vindex_path: &Path,
    ) -> Result<SharedLqlSession, String> {
        if let Some(existing) = self
            .sessions
            .read()
            .map_err(|e| format!("lql session map poisoned: {e}"))?
            .get(key)
            .cloned()
        {
            return Ok(existing);
        }

        let mut write = self
            .sessions
            .write()
            .map_err(|e| format!("lql session map poisoned: {e}"))?;

        // Another caller may have raced us to creation; re-check.
        if let Some(existing) = write.get(key).cloned() {
            return Ok(existing);
        }

        let mut session = Session::new();
        let path_str = vindex_path.to_string_lossy().to_string();
        session
            .execute(&Statement::Use {
                target: UseTarget::Vindex(path_str),
            })
            .map_err(|e| format!("LQL USE failed: {e}"))?;

        let arc = Arc::new(Mutex::new(session));
        write.insert(key.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Drop a session's cross-fact state. Called by administrative
    /// endpoints when a session is terminated.
    #[allow(dead_code)]
    pub fn remove(&self, key: &str) -> bool {
        match self.sessions.write() {
            Ok(mut map) => map.remove(key).is_some(),
            Err(_) => false,
        }
    }
}

impl Default for LqlSessionManager {
    fn default() -> Self {
        Self::new()
    }
}
