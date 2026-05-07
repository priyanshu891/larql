//! `/v1/vindexes/extract` — kick off a vindex build and stream progress.
//!
//! Flow:
//!   1. `POST /v1/vindexes/extract` — validates the request, spawns a
//!      blocking worker that runs `larql_vindex::build_vindex_streaming`,
//!      and returns `{ id }` immediately. Only one job runs at a time.
//!   2. `GET  /v1/vindexes/extract/status?id=<id>` — SSE stream. The
//!      handler replays every event recorded so far and then follows the
//!      broadcast channel for live events until the job terminates.
//!   3. On success the worker creates a `ModelSlot::discover` on the
//!      new output directory and registers it in `AppState.models` so the
//!      new vindex shows up in `/v1/vindexes` without a server restart.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, RwLock};
use std::time::Instant;

use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, Stream, StreamExt};
use tracing::{error, info, warn};
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::slot::{ModelSlot, SlotLoadOpts};
use crate::state::AppState;

/// Capacity of the per-job broadcast channel. 256 slots absorbs burst
/// activity without stalling the worker if a subscriber is slow to drain.
const EVENT_CHANNEL_CAPACITY: usize = 256;

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize, ToSchema)]
pub struct ExtractRequest {
    /// Model path or HuggingFace identifier. Resolved via
    /// `larql_models::resolve_model_path`, which walks the HF cache for
    /// `owner/name` inputs.
    pub model: String,
    /// Extract level: `browse | attention | inference | all`.
    /// Defaults to `inference`.
    #[serde(default = "default_level")]
    pub level: String,
    /// Output directory name (relative to the server's vindex dir).
    #[serde(default)]
    pub output: Option<String>,
}

fn default_level() -> String {
    "inference".into()
}

#[derive(Serialize, ToSchema)]
pub struct ExtractStartResponse {
    pub id: String,
    /// Relative SSE URL so the UI can subscribe without reconstructing it.
    pub status_url: String,
}

#[derive(Deserialize)]
pub struct StatusQuery {
    pub id: String,
}

/// Events streamed over SSE.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ExtractProgress {
    Progress {
        stage: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        component: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        layer: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total_layers: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        done: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total: Option<usize>,
    },
    Done {
        path: String,
        id: String,
        elapsed_ms: f64,
    },
    Error {
        message: String,
    },
}

// ── Job registry ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Running,
    Done,
    Failed,
}

pub struct ExtractJob {
    pub id: String,
    pub model: String,
    pub level: String,
    pub output_path: PathBuf,
    pub started_at: Instant,
    status: RwLock<JobStatus>,
    tx: broadcast::Sender<ExtractProgress>,
    events_log: StdMutex<Vec<ExtractProgress>>,
}

impl ExtractJob {
    /// Append + broadcast one event atomically, so late SSE subscribers
    /// can snapshot the log + subscribe under the same mutex without
    /// missing or duplicating any event.
    fn emit(&self, ev: ExtractProgress) {
        if let Ok(mut log) = self.events_log.lock() {
            // Soft cap to keep RSS bounded on very long builds.
            if log.len() >= 4000 {
                log.drain(0..1000);
            }
            log.push(ev.clone());
            let _ = self.tx.send(ev);
        }
    }

    pub fn status(&self) -> JobStatus {
        self.status.read().map(|g| *g).unwrap_or(JobStatus::Failed)
    }

    fn set_status(&self, s: JobStatus) {
        if let Ok(mut g) = self.status.write() {
            *g = s;
        }
    }
}

pub struct ExtractJobs {
    jobs: RwLock<std::collections::HashMap<String, Arc<ExtractJob>>>,
    current: RwLock<Option<String>>,
}

impl ExtractJobs {
    pub fn new() -> Self {
        Self {
            jobs: RwLock::new(std::collections::HashMap::new()),
            current: RwLock::new(None),
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<ExtractJob>> {
        self.jobs.read().ok()?.get(id).cloned()
    }

    fn claim(&self, job: Arc<ExtractJob>) -> Result<(), ServerError> {
        let mut current = self
            .current
            .write()
            .map_err(|_| ServerError::Internal("extract lock poisoned".into()))?;
        if let Some(cur_id) = current.as_ref() {
            if let Some(cur) = self.jobs.read().ok().and_then(|j| j.get(cur_id).cloned()) {
                if cur.status() == JobStatus::Running {
                    return Err(ServerError::BadRequest(format!(
                        "another extract ({}) is already running; wait for it or poll its status",
                        cur.id
                    )));
                }
            }
        }
        if let Ok(mut jobs) = self.jobs.write() {
            jobs.insert(job.id.clone(), Arc::clone(&job));
        }
        *current = Some(job.id.clone());
        Ok(())
    }
}

impl Default for ExtractJobs {
    fn default() -> Self {
        Self::new()
    }
}

// ── Callback bridging IndexBuildCallbacks → SSE events ────────────────────────

struct SseCallbacks {
    job: Arc<ExtractJob>,
    current_stage: String,
    last_layer_total: usize,
    last_progress_emit: Instant,
}

impl SseCallbacks {
    fn new(job: Arc<ExtractJob>) -> Self {
        Self {
            job,
            current_stage: String::new(),
            last_layer_total: 0,
            last_progress_emit: Instant::now()
                .checked_sub(std::time::Duration::from_secs(60))
                .unwrap_or_else(Instant::now),
        }
    }
}

impl larql_vindex::IndexBuildCallbacks for SseCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.current_stage = stage.to_string();
        self.last_progress_emit = Instant::now();
        self.job.emit(ExtractProgress::Progress {
            stage: stage.to_string(),
            component: None,
            layer: None,
            total_layers: None,
            done: None,
            total: None,
        });
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, total: usize) {
        self.last_layer_total = total;
        self.last_progress_emit = Instant::now();
        self.job.emit(ExtractProgress::Progress {
            stage: self.current_stage.clone(),
            component: Some(component.to_string()),
            layer: Some(layer),
            total_layers: Some(total),
            done: Some(0),
            total: None,
        });
    }

    fn on_feature_progress(&mut self, component: &str, layer: usize, done: usize, total: usize) {
        let is_first = done == 0;
        let is_last = total > 0 && done == total;
        if is_first
            || is_last
            || self.last_progress_emit.elapsed() >= std::time::Duration::from_millis(150)
        {
            self.last_progress_emit = Instant::now();
            self.job.emit(ExtractProgress::Progress {
                stage: self.current_stage.clone(),
                component: Some(component.to_string()),
                layer: Some(layer),
                total_layers: Some(self.last_layer_total),
                done: Some(done),
                total: if total > 0 { Some(total) } else { None },
            });
        }
    }

    fn on_layer_done(&mut self, component: &str, layer: usize, elapsed_ms: f64) {
        self.last_progress_emit = Instant::now();
        self.job.emit(ExtractProgress::Progress {
            stage: self.current_stage.clone(),
            component: Some(component.to_string()),
            layer: Some(layer),
            total_layers: Some(self.last_layer_total),
            done: None,
            total: Some(elapsed_ms.round() as usize),
        });
    }

    fn on_stage_done(&mut self, stage: &str, elapsed_ms: f64) {
        self.last_progress_emit = Instant::now();
        self.job.emit(ExtractProgress::Progress {
            stage: format!("{stage} done"),
            component: None,
            layer: None,
            total_layers: None,
            done: None,
            total: Some(elapsed_ms.round() as usize),
        });
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/v1/vindexes/extract",
    tag = "admin",
    request_body = ExtractRequest,
    responses(
        (status = 200, description = "Extract job queued", body = ExtractStartResponse),
        (status = 400, description = "Invalid request", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_extract(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractStartResponse>, ServerError> {
    let vindex_dir = state.vindex_dir.clone().ok_or_else(|| {
        ServerError::BadRequest(
            "server was started without a vindex directory; pass --vindex-dir or set \
             LARQL_VINDEX_DIR so extracts have somewhere to land"
                .into(),
        )
    })?;

    let level = parse_level(&req.level).map_err(ServerError::BadRequest)?;
    let model = req.model.trim().to_string();
    if model.is_empty() {
        return Err(ServerError::BadRequest("model is required".into()));
    }

    let output_rel = req
        .output
        .as_deref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_output_name(&model));
    let output_path = sanitize_output_path(&vindex_dir, &output_rel)?;

    let id = next_job_id();
    let (tx, _rx) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
    let job = Arc::new(ExtractJob {
        id: id.clone(),
        model: model.clone(),
        level: req.level.clone(),
        output_path: output_path.clone(),
        started_at: Instant::now(),
        status: RwLock::new(JobStatus::Running),
        tx,
        events_log: StdMutex::new(Vec::new()),
    });

    state.extract_jobs.claim(Arc::clone(&job))?;

    let state_for_worker = Arc::clone(&state);
    let job_for_worker = Arc::clone(&job);
    tokio::task::spawn_blocking(move || {
        run_extract(state_for_worker, job_for_worker, model, level, output_path);
    });

    let status_url = format!("/v1/vindexes/extract/status?id={}", id);
    info!(
        "extract: queued id={id} model={} level={}",
        req.model, req.level
    );
    Ok(Json(ExtractStartResponse { id, status_url }))
}

#[utoipa::path(
    get,
    path = "/v1/vindexes/extract/status",
    tag = "admin",
    params(("id" = String, Query, description = "Extract job id from POST /v1/vindexes/extract")),
    responses(
        (status = 200, description = "SSE stream of ExtractProgress events"),
        (status = 404, description = "Unknown job id", body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_extract_status(
    State(state): State<Arc<AppState>>,
    Query(q): Query<StatusQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ServerError> {
    let job = state
        .extract_jobs
        .get(&q.id)
        .ok_or_else(|| ServerError::NotFound(format!("extract job '{}' not found", q.id)))?;

    // Snapshot + subscribe atomically under the event-log mutex —
    // `ExtractJob::emit` holds the same mutex across the log append AND
    // the broadcast send, so any event is either (a) already in the
    // snapshot AND not yet sendable to our rx, or (b) not in the
    // snapshot AND landing in our rx after this block releases.
    let (replay, rx) = match job.events_log.lock() {
        Ok(log) => (log.clone(), job.tx.subscribe()),
        Err(_) => {
            return Err(ServerError::Internal(
                "extract event log lock poisoned".into(),
            ));
        }
    };
    let terminal = job.status() != JobStatus::Running;

    let replay_stream = tokio_stream::iter(replay.into_iter().map(Ok));
    let live_stream = BroadcastStream::new(rx).filter_map(|r| r.ok()).map(Ok);

    let combined: std::pin::Pin<
        Box<dyn Stream<Item = Result<ExtractProgress, Infallible>> + Send>,
    > = if terminal {
        Box::pin(replay_stream)
    } else {
        Box::pin(replay_stream.chain(live_stream))
    };

    let sse_stream = combined.map(|ev| {
        ev.and_then(|p| {
            Ok(Event::default().json_data(&p).unwrap_or_else(|_| {
                Event::default().data("{\"type\":\"error\",\"message\":\"serialize failed\"}")
            }))
        })
    });

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::default()))
}

// ── Worker ────────────────────────────────────────────────────────────────────

fn run_extract(
    state: Arc<AppState>,
    job: Arc<ExtractJob>,
    model: String,
    level: larql_vindex::ExtractLevel,
    output_path: PathBuf,
) {
    job.emit(ExtractProgress::Progress {
        stage: "resolving model path".into(),
        component: None,
        layer: None,
        total_layers: None,
        done: None,
        total: None,
    });

    // For bare HF ids always route through `download_hf_model`, even when
    // `resolve_model_path` returns OK — a prior partial download can leave
    // a snapshot dir with just config.json/tokenizer.json and no
    // safetensors, and our old logic would gleefully hand that back only
    // for `build_vindex_streaming` to panic with "no safetensors files".
    let model_path = if looks_like_hf_id(&model) {
        if let Err(e) = download_hf_model(&model, &job) {
            finalize_error(&job, format!("hf download: {e}"));
            return;
        }
        match larql_models::resolve_model_path(&model) {
            Ok(p) => p,
            Err(e) => {
                finalize_error(
                    &job,
                    format!("downloaded but resolve_model_path still failed: {e}"),
                );
                return;
            }
        }
    } else {
        match larql_models::resolve_model_path(&model) {
            Ok(p) => p,
            Err(e) => {
                finalize_error(&job, format!("resolve_model_path: {e}"));
                return;
            }
        }
    };

    job.emit(ExtractProgress::Progress {
        stage: "loading tokenizer".into(),
        component: None,
        layer: None,
        total_layers: None,
        done: None,
        total: None,
    });
    let tok_path = model_path.join("tokenizer.json");
    if !tok_path.exists() {
        finalize_error(
            &job,
            format!("tokenizer.json not found at {}", model_path.display()),
        );
        return;
    }
    let tokenizer = match larql_vindex::tokenizers::Tokenizer::from_file(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            finalize_error(&job, format!("failed to load tokenizer: {e}"));
            return;
        }
    };

    if let Err(e) = std::fs::create_dir_all(&output_path) {
        finalize_error(
            &job,
            format!("create_dir_all({}): {e}", output_path.display()),
        );
        return;
    }

    let mut callbacks = SseCallbacks::new(Arc::clone(&job));
    let weight_opts = larql_vindex::WriteWeightsOptions {
        level,
        ffn_compact: false,
    };
    let q4k_opts = larql_vindex::Q4kWriteOptions {
        down_q4k: false,
        feature_major_down: false,
    };

    let result = larql_vindex::build_vindex_streaming(
        &model_path,
        &tokenizer,
        &model,
        &output_path,
        /* down_top_k */ 10,
        level,
        larql_vindex::StorageDtype::F16,
        larql_vindex::QuantFormat::None,
        weight_opts,
        q4k_opts,
        /* drop_gate_vectors */ false,
        &mut callbacks,
    );

    match result {
        Ok(()) => {
            let elapsed_ms = job.started_at.elapsed().as_secs_f64() * 1000.0;
            let id = match ModelSlot::discover(output_path.clone(), SlotLoadOpts::default()) {
                Ok(slot) => {
                    let id = slot.id.clone();
                    if state.add_slot(Arc::new(slot)) {
                        info!(
                            "extract: registered new slot {} at {}",
                            id,
                            output_path.display()
                        );
                    } else {
                        warn!(
                            "extract: slot {} already registered — refresh instead",
                            id
                        );
                    }
                    id
                }
                Err(e) => {
                    warn!("extract: built vindex but discover failed: {e}");
                    finalize_error(&job, format!("built vindex but discover failed: {e}"));
                    return;
                }
            };

            job.emit(ExtractProgress::Done {
                path: output_path.display().to_string(),
                id,
                elapsed_ms,
            });
            job.set_status(JobStatus::Done);
        }
        Err(e) => {
            finalize_error(&job, format!("build_vindex_streaming: {e}"));
        }
    }
}

fn finalize_error(job: &Arc<ExtractJob>, message: String) {
    error!("extract {}: {}", job.id, message);
    job.emit(ExtractProgress::Error {
        message: message.clone(),
    });
    job.set_status(JobStatus::Failed);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_level(s: &str) -> Result<larql_vindex::ExtractLevel, String> {
    match s.to_lowercase().as_str() {
        "browse" => Ok(larql_vindex::ExtractLevel::Browse),
        "attention" | "attn" => Ok(larql_vindex::ExtractLevel::Attention),
        "inference" | "infer" => Ok(larql_vindex::ExtractLevel::Inference),
        "all" => Ok(larql_vindex::ExtractLevel::All),
        other => Err(format!(
            "unknown extract level '{other}' (expected: browse, attention, inference, all)"
        )),
    }
}

fn default_output_name(model: &str) -> String {
    let leaf = model.rsplit('/').next().unwrap_or(model);
    let leaf = leaf.trim_end_matches(".gguf");
    format!("{leaf}.vindex")
}

fn sanitize_output_path(vindex_dir: &Path, rel: &str) -> Result<PathBuf, ServerError> {
    if rel.is_empty() || rel.contains("..") || rel.starts_with('/') {
        return Err(ServerError::BadRequest(format!(
            "invalid output name '{rel}' — must be a plain directory name under the vindex dir"
        )));
    }
    Ok(vindex_dir.join(rel))
}

fn next_job_id() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let nonce: u32 = rand_like();
    format!("x-{t:x}-{nonce:x}")
}

fn rand_like() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = format!("{:?}", std::thread::current().id());
    let mut h: u32 = 0x9e37_79b9;
    for b in tid.as_bytes() {
        h = h.wrapping_mul(16777619) ^ (*b as u32);
    }
    n ^ h
}

// ── HuggingFace download pre-flight ───────────────────────────────────────────

fn looks_like_hf_id(s: &str) -> bool {
    if s.starts_with('/') || s.starts_with('.') {
        return false;
    }
    if std::path::Path::new(s).exists() {
        return false;
    }
    let parts: Vec<&str> = s.splitn(3, '/').collect();
    parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty()
}

fn is_needed_weight_file(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name == "model.safetensors.index.json"
        || name == "config.json"
        || name == "tokenizer.json"
        || name == "tokenizer_config.json"
        || name == "special_tokens_map.json"
        || name == "generation_config.json"
        || name == "tokenizer.model"
}

fn download_hf_model(model_id: &str, job: &Arc<ExtractJob>) -> Result<(), String> {
    job.emit(ExtractProgress::Progress {
        stage: "fetching HF repo info".into(),
        component: Some(model_id.to_string()),
        layer: None,
        total_layers: None,
        done: None,
        total: None,
    });

    let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf api init: {e}"))?;
    let repo = api.model(model_id.to_string());

    let cache = hf_hub::Cache::from_env();
    let cache_repo = cache.repo(hf_hub::Repo::model(model_id.to_string()));

    let info = repo
        .info()
        .map_err(|e| translate_hf_error(model_id, "repo info", e))?;

    let files: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|f| is_needed_weight_file(f))
        .collect();

    if files.is_empty() {
        return Err(format!(
            "repo {model_id} has no .safetensors / config.json / tokenizer.json siblings — \
             is this a model repo?"
        ));
    }

    if !files.iter().any(|f| f == "tokenizer.json") {
        return Err(format!(
            "repo {model_id} does not publish a `tokenizer.json` (uses legacy `vocab.json` + \
             `merges.txt`). The extractor requires the HF fast-tokenizer format — try a \
             different checkpoint of the same architecture, or convert the tokenizer manually \
             first."
        ));
    }

    info!(
        "extract: downloading {} files from hf://{}",
        files.len(),
        model_id
    );

    let total_files = files.len();
    for (idx, filename) in files.iter().enumerate() {
        if let Some(cached_path) = cache_repo.get(filename) {
            let size = std::fs::metadata(&cached_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            let mut progress =
                SseDownloadProgress::new(Arc::clone(job), filename.clone(), idx + 1, total_files);
            hf_hub::api::Progress::init(&mut progress, size, filename);
            hf_hub::api::Progress::finish(&mut progress);
            info!("extract: cache hit for {filename} ({size} bytes)");
            continue;
        }
        let progress =
            SseDownloadProgress::new(Arc::clone(job), filename.clone(), idx + 1, total_files);
        repo.download_with_progress(filename, progress)
            .map_err(|e| translate_hf_error(model_id, &format!("download {filename}"), e))?;
    }

    Ok(())
}

fn translate_hf_error<E: std::fmt::Display>(model_id: &str, context: &str, e: E) -> String {
    let raw = e.to_string();
    if raw.contains("403") {
        format!(
            "HuggingFace refused access to {model_id} ({context}: HTTP 403). \
             This usually means the model is gated — open https://huggingface.co/{model_id} \
             in a browser, accept the license, and confirm your `HF_TOKEN` (or \
             ~/.cache/huggingface/token) belongs to the account that accepted it."
        )
    } else if raw.contains("401") {
        format!(
            "HuggingFace rejected the request to {model_id} ({context}: HTTP 401). \
             Set `HF_TOKEN` or run `huggingface-cli login` so the extract worker can authenticate."
        )
    } else if raw.contains("404") {
        format!(
            "HuggingFace has no repo called {model_id} ({context}: HTTP 404). \
             Double-check the owner/name."
        )
    } else {
        format!("{context} on {model_id}: {raw}")
    }
}

struct SseDownloadProgress {
    job: Arc<ExtractJob>,
    filename: String,
    file_idx: usize,
    total_files: usize,
    total: usize,
    sent: usize,
    last_emit: Instant,
}

impl SseDownloadProgress {
    fn new(job: Arc<ExtractJob>, filename: String, file_idx: usize, total_files: usize) -> Self {
        Self {
            job,
            filename,
            file_idx,
            total_files,
            total: 0,
            sent: 0,
            last_emit: Instant::now(),
        }
    }

    fn emit(&self) {
        self.job.emit(ExtractProgress::Progress {
            stage: format!("downloading ({}/{})", self.file_idx, self.total_files),
            component: Some(self.filename.clone()),
            layer: None,
            total_layers: None,
            done: Some(self.sent),
            total: if self.total > 0 { Some(self.total) } else { None },
        });
    }
}

impl hf_hub::api::Progress for SseDownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total = size;
        self.sent = 0;
        if !filename.is_empty() {
            self.filename = filename.to_string();
        }
        self.last_emit = Instant::now();
        self.emit();
    }

    fn update(&mut self, size: usize) {
        self.sent = self.sent.saturating_add(size);
        if self.last_emit.elapsed() >= std::time::Duration::from_millis(100) {
            self.last_emit = Instant::now();
            self.emit();
        }
    }

    fn finish(&mut self) {
        self.sent = self.total.max(self.sent);
        self.emit();
    }
}
