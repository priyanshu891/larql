//! `POST /v1/lql` — execute an LQL statement via HTTP.
//!
//! Ported from the sibling `larql` repo.  Inlines a small subset of
//! DESCRIBE / SHOW / STATS directly over `LoadedModel` fields rather than
//! pulling in the full `larql-lql` evaluator — the HTTP surface only
//! needs the read-only query path.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize, Serialize, ToSchema)]
pub struct LqlRequest {
    pub statement: String,
}

#[derive(Deserialize)]
pub struct LqlQuery {
    /// Optional model id (`?model=<id>`). Required when the server has
    /// more than one slot registered.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct LqlResponse {
    pub output: Vec<String>,
    pub error: Option<String>,
    pub latency_ms: f64,
}

#[utoipa::path(
    post,
    path = "/v1/lql",
    tag = "browse",
    request_body = LqlRequest,
    params(("model" = Option<String>, Query, description = "Optional model id (required for multi-model servers)")),
    responses(
        (status = 200, description = "LQL result", body = LqlResponse),
        (status = 400, description = "Invalid request or ambiguous model", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_lql(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LqlQuery>,
    Json(req): Json<LqlRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start = std::time::Instant::now();
    let stmt = req.statement.trim().to_string();

    if stmt.is_empty() {
        return Err(ServerError::BadRequest("empty statement".into()));
    }

    let model = match q.model.as_deref() {
        Some(id) => state.resolve(Some(id)).await?,
        None => match state.resolve(None).await {
            Ok(m) => m,
            Err(ServerError::BadRequest(_)) => {
                let ids: Vec<String> = state
                    .snapshot_slots()
                    .iter()
                    .map(|s| s.id.clone())
                    .collect();
                return Err(ServerError::BadRequest(format!(
                    "server has {} models registered; pass `?model=<id>` on /v1/lql. \
                     Available: {}",
                    ids.len(),
                    ids.join(", ")
                )));
            }
            Err(e) => return Err(e),
        },
    };

    let result = tokio::task::spawn_blocking({
        let model = Arc::clone(&model);
        let stmt = stmt.clone();
        move || execute_lql_statement(&model, &stmt)
    })
    .await
    .map_err(|e| ServerError::Internal(format!("task join error: {e}")))?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(output) => Ok(Json(LqlResponse {
            output,
            error: None,
            latency_ms: elapsed,
        })),
        Err(msg) => Ok(Json(LqlResponse {
            output: vec![],
            error: Some(msg),
            latency_ms: elapsed,
        })),
    }
}

fn execute_lql_statement(model: &LoadedModel, stmt: &str) -> Result<Vec<String>, String> {
    use larql_vindex::ndarray;

    let upper = stmt.trim_end_matches(';').trim().to_uppercase();
    let first_word = upper.split_whitespace().next().unwrap_or("");

    match first_word {
        "DESCRIBE" => {
            let entity = extract_quoted(stmt).ok_or("DESCRIBE requires a quoted entity")?;
            let patched = model.patched.blocking_read();
            let hidden = model.embeddings.shape()[1];

            let encoding = model
                .tokenizer
                .encode(entity.as_str(), false)
                .map_err(|e| format!("tokenize error: {e}"))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if token_ids.is_empty() {
                return Ok(vec![format!("No tokens for entity: {entity}")]);
            }

            let query = if token_ids.len() == 1 {
                model
                    .embeddings
                    .row(token_ids[0] as usize)
                    .mapv(|v| v * model.embed_scale)
            } else {
                let mut avg = ndarray::Array1::<f32>::zeros(hidden);
                for &tok in &token_ids {
                    avg += &model
                        .embeddings
                        .row(tok as usize)
                        .mapv(|v| v * model.embed_scale);
                }
                avg /= token_ids.len() as f32;
                avg
            };

            let mut lines = vec![format!("Entity: {entity}")];
            let num_layers = model.config.num_layers;
            for layer in 0..num_layers {
                let results = patched.gate_knn(layer, &query, 5);
                for (feat, score) in results {
                    if score < 5.0 {
                        continue;
                    }
                    let label = match patched.feature_meta(layer, feat) {
                        Some(meta) => meta.top_token.clone(),
                        None => String::new(),
                    };
                    lines.push(format!("  L{layer:02} F{feat:04} score={score:.2} {label}"));
                }
            }
            if lines.len() == 1 {
                lines.push("  (no matching features)".into());
            }
            Ok(lines)
        }

        "SHOW" => {
            if upper.contains("LAYERS") {
                let num_layers = model.config.num_layers;
                let mut lines = vec![format!("Layers: {num_layers}")];
                for (i, l) in model.config.layers.iter().enumerate() {
                    lines.push(format!("  L{i:02}: {} features", l.num_features));
                }
                Ok(lines)
            } else if upper.contains("RELATIONS") {
                let mut lines = vec![];
                if model.probe_labels.is_empty() {
                    lines.push(
                        "No probe-confirmed relations. Use /v1/relations for inferred relations."
                            .into(),
                    );
                } else {
                    use std::collections::HashMap;
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    for label in model.probe_labels.values() {
                        *counts.entry(label.as_str()).or_insert(0) += 1;
                    }
                    let mut sorted: Vec<_> = counts.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(&a.1));
                    lines.push(format!("Probe-confirmed relations: {}", sorted.len()));
                    for (name, count) in sorted.iter().take(20) {
                        lines.push(format!("  {} (count={})", name, count));
                    }
                }
                Ok(lines)
            } else if upper.contains("PATCHES") {
                let patched = model.patched.blocking_read();
                let count = patched.num_patches();
                if count == 0 {
                    Ok(vec!["No patches applied.".into()])
                } else {
                    Ok(vec![format!("Applied patches: {count}")])
                }
            } else {
                Ok(vec![format!("SHOW subcommand not recognized: {stmt}")])
            }
        }

        "STATS" => {
            let total_features: usize =
                model.config.layers.iter().map(|l| l.num_features).sum();
            Ok(vec![
                format!("Model: {}", model.config.model),
                format!("Layers: {}", model.config.num_layers),
                format!("Features: {total_features}"),
                format!("Hidden size: {}", model.config.hidden_size),
                format!("Vocab size: {}", model.config.vocab_size),
                format!("Extract level: {:?}", model.config.extract_level),
            ])
        }

        _ => Err(format!(
            "Statement type '{first_word}' is not yet supported via the HTTP LQL endpoint. \
             Supported: DESCRIBE, SHOW, STATS."
        )),
    }
}

fn extract_quoted(s: &str) -> Option<String> {
    let start = s.find('"')?;
    let rest = &s[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}
