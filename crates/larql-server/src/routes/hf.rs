//! `GET /v1/hf/search` — proxy HuggingFace model search.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct HfSearchParams {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    20
}

#[derive(Serialize, ToSchema)]
pub struct HfSearchResponse {
    pub models: Vec<HfModelInfo>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct HfModelInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
}

#[utoipa::path(
    get,
    path = "/v1/hf/search",
    tag = "admin",
    params(
        ("q" = String, Query, description = "Search query"),
        ("limit" = Option<usize>, Query, description = "Max results (capped at 50)"),
    ),
    responses(
        (status = 200, description = "HuggingFace model list", body = HfSearchResponse),
    ),
)]
pub async fn handle_hf_search(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<HfSearchParams>,
) -> Result<impl IntoResponse, ServerError> {
    if params.q.trim().is_empty() {
        return Ok(Json(HfSearchResponse { models: vec![] }));
    }

    let limit = params.limit.min(50);
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=text-generation&sort=downloads&direction=-1&limit={}",
        urlencoding::encode(&params.q),
        limit
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| ServerError::Internal(format!("http client error: {e}")))?;

    let resp = client
        .get(&url)
        .header("User-Agent", "larql-server")
        .send()
        .await
        .map_err(|e| ServerError::Internal(format!("HF API request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(ServerError::Internal(format!(
            "HF API returned status {}",
            resp.status()
        )));
    }

    let models: Vec<HfModelInfo> = resp
        .json()
        .await
        .map_err(|e| ServerError::Internal(format!("HF API parse error: {e}")))?;

    Ok(Json(HfSearchResponse { models }))
}
