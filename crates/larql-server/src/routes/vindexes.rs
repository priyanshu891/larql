//! `/v1/vindexes` — workspace: list, load, unload, delete.
//!
//! Each slot registered in `AppState` is a cheap handle: metadata + path
//! without the weights until explicitly loaded.  `POST /v1/vindexes/{id}/load`
//! pre-warms; `POST /v1/vindexes/{id}/unload` drops weights back to disk.
//! `DELETE /v1/vindexes/{id}` is a deliberate 400 — removing a vindex
//! while the server is running risks stale in-flight handler refs.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::slot::SlotState;
use crate::state::AppState;

#[derive(Serialize, ToSchema)]
pub struct VindexInfo {
    pub id: String,
    /// Backwards-compat alias for `id` — some UIs still read this field.
    pub name: String,
    pub path: String,
    pub model: String,
    pub layers: usize,
    pub features: usize,
    pub extract_level: String,
    pub size_bytes: u64,
    pub state: SlotState,
    pub last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_stage: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct VindexListResponse {
    pub vindexes: Vec<VindexInfo>,
    pub vindex_dir: Option<String>,
}

#[utoipa::path(
    get,
    path = "/v1/vindexes",
    tag = "admin",
    responses(
        (status = 200, description = "List of registered vindex slots", body = VindexListResponse),
    ),
)]
pub async fn handle_list(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ServerError> {
    let mut vindexes = Vec::new();
    let slots = state.snapshot_slots();

    for slot in &slots {
        let total_features: usize = slot.config.layers.iter().map(|l| l.num_features).sum();
        let slot_state = slot.state().await;
        let last_error = slot.last_error().await;
        let load_stage = if slot_state == SlotState::Loading {
            slot.progress()
        } else {
            None
        };
        vindexes.push(VindexInfo {
            id: slot.id.clone(),
            name: slot.id.clone(),
            path: slot.path.display().to_string(),
            model: slot.config.model.clone(),
            layers: slot.config.num_layers,
            features: total_features,
            extract_level: format!("{:?}", slot.config.extract_level),
            size_bytes: slot.size_bytes,
            state: slot_state,
            last_error,
            load_stage,
        });
    }

    Ok(Json(VindexListResponse {
        vindexes,
        vindex_dir: state
            .vindex_dir
            .as_ref()
            .map(|p| p.display().to_string()),
    }))
}

// ── Load / unload ────────────────────────────────────────────────────────────

#[derive(Serialize, ToSchema)]
pub struct LoadResponse {
    pub id: String,
    pub state: SlotState,
    pub already_loaded: bool,
    pub load_ms: f64,
}

#[utoipa::path(
    post,
    path = "/v1/vindexes/{name}/load",
    tag = "admin",
    params(("name" = String, Path, description = "Vindex slot id")),
    responses(
        (status = 200, description = "Slot state after load", body = LoadResponse),
        (status = 404, description = "Unknown slot", body = crate::error::ErrorBody),
        (status = 500, description = "Load failed", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_load(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<LoadResponse>, ServerError> {
    let slot = state
        .find_slot(Some(&id))
        .ok_or_else(|| ServerError::NotFound(format!("vindex '{id}' not found")))?;
    let already_loaded = slot.state().await == SlotState::Loaded;
    let start = std::time::Instant::now();
    slot.get_or_load()
        .await
        .map_err(|e| ServerError::Internal(format!("load failed: {e}")))?;
    Ok(Json(LoadResponse {
        id: slot.id.clone(),
        state: slot.state().await,
        already_loaded,
        load_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

#[derive(Serialize, ToSchema)]
pub struct UnloadResponse {
    pub id: String,
    pub state: SlotState,
}

#[utoipa::path(
    post,
    path = "/v1/vindexes/{name}/unload",
    tag = "admin",
    params(("name" = String, Path, description = "Vindex slot id")),
    responses(
        (status = 200, description = "Slot state after unload", body = UnloadResponse),
        (status = 404, description = "Unknown slot", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_unload(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<UnloadResponse>, ServerError> {
    let slot = state
        .find_slot(Some(&id))
        .ok_or_else(|| ServerError::NotFound(format!("vindex '{id}' not found")))?;
    slot.unload().await;
    Ok(Json(UnloadResponse {
        id: slot.id.clone(),
        state: slot.state().await,
    }))
}

#[utoipa::path(
    delete,
    path = "/v1/vindexes/{name}",
    tag = "admin",
    params(("name" = String, Path, description = "Vindex slot id")),
    responses(
        (status = 400, description = "Deletion unsupported while server is running", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_delete(
    State(_state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    Err(ServerError::BadRequest(format!(
        "Cannot delete vindex '{name}' while the server is running. \
         Stop the server and use `larql rm {name}` from the CLI."
    )))
}
