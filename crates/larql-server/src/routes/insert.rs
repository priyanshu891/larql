//! POST /v1/insert — routes through the LQL Compose executor.
//!
//! This handler is a thin adapter: it builds a `Statement::Insert` AST
//! node with `InsertMode::Compose` (by default) and dispatches it to a
//! persistent per-HTTP-session `larql_lql::Session`. The LQL executor
//! runs the full five-phase Compose pipeline — `plan_install`,
//! `capture_install_residuals`, `install_slots`, `balance_installed`,
//! `cross_fact_regression_check` — which is the only code path that
//! defends against multi-fact hijacking. The previous bespoke handler
//! bypassed the balance + regression phases and wrote at
//! `alpha = 0.25` (vs the LQL default of `0.1`), causing installed
//! facts to bleed onto template-matched neighbours and degrade prior
//! inserts.
//!
//! The server's `PatchedVindex` is swapped into the LQL session's
//! backend for the duration of each call (see
//! `Session::swap_patched_vindex`) so mutations land directly on the
//! server's state. The cross-fact accumulators (`installed_edges`,
//! `raw_install_residuals`, `decoy_residual_cache`) live on the
//! `Session` struct itself and persist across calls, keyed by
//! `X-Session-Id`.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::Deserialize;

use larql_lql::Session;
use larql_lql::ast::{InsertMode, Statement};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct InsertRequest {
    pub entity: String,
    pub relation: String,
    pub target: String,
    /// Pin the install to a single layer. Omit for the default band
    /// (upper-knowledge layers, matching LQL's planner).
    #[serde(default)]
    pub layer: Option<u32>,
    /// Per-layer down-vector multiplier. Omit to use the LQL default
    /// (0.1) — the validated value from `experiments/14_vindex_compilation`.
    /// The previous handler's hardcoded 0.25 was the primary cause of
    /// neighbour bleed.
    #[serde(default)]
    pub alpha: Option<f32>,
    /// Confidence score written into the feature metadata. Defaults to
    /// LQL's 0.9.
    #[serde(default)]
    pub confidence: Option<f32>,
    /// `"compose"` (default) runs the full guardrail pipeline. `"knn"`
    /// uses the retrieval-only KNN store (Architecture B) and skips the
    /// FFN overlay. Unknown values fall back to Compose.
    #[serde(default)]
    pub mode: Option<String>,
}

fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn parse_mode(mode: Option<&str>) -> InsertMode {
    match mode.map(|m| m.trim().to_ascii_lowercase()) {
        Some(ref s) if s == "knn" => InsertMode::Knn,
        _ => InsertMode::Compose,
    }
}

fn mode_str(mode: InsertMode) -> &'static str {
    match mode {
        InsertMode::Compose => "compose",
        InsertMode::Knn => "knn",
    }
}

/// Key into the LQL session manager. Requests without an `X-Session-Id`
/// header share a single `"__global__"` session so multi-insert
/// cross-fact state is still maintained for them.
fn lql_key(session_id: Option<&str>) -> String {
    match session_id {
        Some(s) => s.to_string(),
        None => "__global__".to_string(),
    }
}

/// Execute the insert inside the LQL session, swapping `patched` in
/// and out around `Session::execute`. Returns `(messages, inserted_ops)`
/// where `inserted_ops` is the delta in the session's recorded patch
/// operations.
fn run_lql_insert(
    session: &mut Session,
    patched: &mut larql_vindex::PatchedVindex,
    stmt: &Statement,
) -> Result<(Vec<String>, usize), ServerError> {
    let prev_ops = session.pending_patch_op_count();
    session
        .swap_patched_vindex(patched)
        .map_err(|e| ServerError::Internal(format!("LQL backend swap-in failed: {e}")))?;
    let exec_result = session.execute(stmt);
    // Always swap back — even on error — so the server's PatchedVindex
    // is restored and the LQL session returns to its resting state.
    let swap_back = session.swap_patched_vindex(patched);
    let messages = exec_result
        .map_err(|e| ServerError::Internal(format!("LQL INSERT failed: {e}")))?;
    swap_back
        .map_err(|e| ServerError::Internal(format!("LQL backend swap-out failed: {e}")))?;
    let delta = session.pending_patch_op_count().saturating_sub(prev_ops);
    Ok((messages, delta))
}

fn run_insert(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    sid: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();
    let mode = parse_mode(req.mode.as_deref());
    let key = lql_key(sid);

    let lql_arc = state
        .lql_sessions
        .get_or_create(&key, &model.path)
        .map_err(|e| ServerError::Internal(format!("LQL session setup failed: {e}")))?;
    let mut lql = lql_arc
        .lock()
        .map_err(|e| ServerError::Internal(format!("LQL session poisoned: {e}")))?;

    let stmt = Statement::Insert {
        entity: req.entity.clone(),
        relation: req.relation.clone(),
        target: req.target.clone(),
        layer: req.layer,
        confidence: req.confidence,
        alpha: req.alpha,
        mode,
    };

    let (messages, inserted_ops) = if let Some(sid) = sid {
        // Session-scoped: write into the per-session PatchedVindex,
        // creating it on first use from the shared base.
        let mut sessions = state.sessions.sessions_blocking_write();
        let now = std::time::Instant::now();
        let server_session = sessions
            .entry(sid.to_string())
            .or_insert_with(|| {
                let base = model.patched.blocking_read();
                crate::session::SessionState::new(base.base().clone(), now)
            });
        server_session.touch(now);
        run_lql_insert(&mut lql, &mut server_session.patched, &stmt)?
    } else {
        // Global: write into the shared model PatchedVindex.
        let mut patched = model.patched.blocking_write();
        run_lql_insert(&mut lql, &mut patched, &stmt)?
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "inserted": inserted_ops,
        "mode": mode_str(mode),
        "alpha": req.alpha,
        "confidence": req.confidence,
        "session": sid,
        "messages": messages,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    }))
}

pub async fn handle_insert(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_insert(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_insert_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_insert(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
