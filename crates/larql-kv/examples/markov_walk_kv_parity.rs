//! Markov-RS walk-KV parity check.
//!
//! Drives baseline dense K/V and candidate walk-KV through the same
//! via-executor path, then compares logits at prefill and forced decode
//! steps. Candidate decode is forced with baseline greedy tokens so the
//! reported drift isolates K/V projection changes instead of sampling
//! divergence.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release -p larql-kv --example markov_walk_kv_parity -- \
//!     --vindex output/gemma3-4b-q4k-v2.vindex \
//!     --prompt "Write a concise technical note about residual streams and attention routing:" \
//!     --tokens 4 \
//!     --top-k-list 64,128 \
//!     --layers 5-20 \
//!     --select-at 4
//! ```

use std::path::PathBuf;

use larql_inference::ffn::NullFfn;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::layer_executor::LocalWalkExecutor;
use larql_kv::vindex_compare::metrics_from_logits;
use larql_kv::EngineKind;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

struct Args {
    vindex: PathBuf,
    prompt: String,
    tokens: usize,
    top_k_list: Vec<usize>,
    layers: String,
    select_at: usize,
    cos_min: f64,
}

struct StepLogits {
    label: String,
    forced_token: Option<u32>,
    next_token: u32,
    logits: Vec<f32>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        vindex: PathBuf::new(),
        prompt: "Write a concise technical note about residual streams and attention routing:"
            .into(),
        tokens: 4,
        top_k_list: vec![64, 128],
        layers: "5-20".into(),
        select_at: 4,
        cos_min: 0.999_999,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--vindex" => {
                i += 1;
                a.vindex = PathBuf::from(&argv[i]);
            }
            "--prompt" => {
                i += 1;
                a.prompt = argv[i].clone();
            }
            "--tokens" => {
                i += 1;
                a.tokens = argv[i].parse().expect("--tokens must be an integer");
            }
            "--top-k-list" => {
                i += 1;
                a.top_k_list = argv[i]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().expect("--top-k-list entries must be integers"))
                    .collect();
            }
            "--layers" => {
                i += 1;
                a.layers = argv[i].clone();
            }
            "--select-at" => {
                i += 1;
                a.select_at = argv[i].parse().expect("--select-at must be an integer");
            }
            "--cos-min" => {
                i += 1;
                a.cos_min = argv[i].parse().expect("--cos-min must be a float");
            }
            other => {
                eprintln!("unknown arg: {other}");
                usage_and_exit();
            }
        }
        i += 1;
    }
    if a.vindex.as_os_str().is_empty() || a.top_k_list.is_empty() {
        usage_and_exit();
    }
    a
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: markov_walk_kv_parity --vindex <path> [--prompt <text>] \
         [--tokens <N>] [--top-k-list 64,128] [--layers 5-20] \
         [--select-at 4] [--cos-min 0.999999]"
    );
    std::process::exit(2);
}

fn clear_walk_kv_env() {
    std::env::remove_var("LARQL_MARKOV_WALK_KV_TOPK");
    std::env::remove_var("LARQL_MARKOV_WALK_KV_LAYERS");
    std::env::remove_var("LARQL_MARKOV_WALK_KV_SELECT_AT");
}

fn set_walk_kv_env(top_k: usize, layers: &str, select_at: usize) {
    std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", top_k.to_string());
    std::env::set_var("LARQL_MARKOV_WALK_KV_LAYERS", layers);
    std::env::set_var("LARQL_MARKOV_WALK_KV_SELECT_AT", select_at.to_string());
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn token_prob(logits: &[f32], token: u32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let idx = token as usize;
    if idx >= logits.len() || denom == 0.0 {
        return f64::NAN;
    }
    ((logits[idx] - max) as f64).exp() / denom
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_baseline(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    tokens: usize,
) -> Vec<StepLogits> {
    clear_walk_kv_env();
    run_markov(weights, index, prompt_ids, tokens, None)
}

fn run_candidate(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    forced_tokens: &[u32],
    top_k: usize,
    layers: &str,
    select_at: usize,
) -> Vec<StepLogits> {
    set_walk_kv_env(top_k, layers, select_at);
    let out = run_markov(
        weights,
        index,
        prompt_ids,
        forced_tokens.len(),
        Some(forced_tokens),
    );
    clear_walk_kv_env();
    out
}

fn run_markov(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    prompt_ids: &[u32],
    tokens: usize,
    forced_tokens: Option<&[u32]>,
) -> Vec<StepLogits> {
    let compute_backend = larql_inference::cpu_backend();
    let engine_backend = larql_inference::cpu_engine_backend();
    let mut engine = EngineKind::MarkovResidual { window_size: None }.build(engine_backend);
    let executor = LocalWalkExecutor::new(compute_backend.as_ref());
    let ffn = NullFfn;

    let mut hidden = engine
        .prefill_quant_via_executor(weights, &executor, &ffn, index, prompt_ids)
        .expect("markov-rs prefill via executor failed");
    let mut logits = hidden_to_raw_logits(weights, &hidden);
    let mut next = argmax(&logits);
    let mut steps = Vec::with_capacity(tokens + 1);
    steps.push(StepLogits {
        label: "prefill".into(),
        forced_token: None,
        next_token: next,
        logits,
    });

    for step in 0..tokens {
        let token = forced_tokens.map_or(next, |ids| ids[step]);
        hidden = engine
            .decode_step_quant_via_executor(weights, &executor, &ffn, index, token)
            .unwrap_or_else(|err| {
                panic!("markov-rs decode via executor failed at step {step}: {err}")
            });
        logits = hidden_to_raw_logits(weights, &hidden);
        next = argmax(&logits);
        steps.push(StepLogits {
            label: format!("decode{}", step + 1),
            forced_token: Some(token),
            next_token: next,
            logits,
        });
    }
    steps
}

fn main() {
    let args = parse_args();

    eprintln!("[setup] loading vindex: {}", args.vindex.display());
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.vindex, &mut cb).expect("vindex load failed");
    index
        .load_attn_kquant(&args.vindex)
        .expect("attn q4k load failed");
    index
        .load_interleaved_kquant(&args.vindex)
        .expect("interleaved q4k load failed");

    eprintln!("[setup] loading model weights and tokenizer");
    let mut weights =
        larql_vindex::load_model_weights_kquant(&args.vindex, &mut cb).expect("weights failed");
    let tokenizer = larql_vindex::load_vindex_tokenizer(&args.vindex).expect("tokenizer failed");
    let prompt_ids = tokenizer
        .encode(args.prompt.as_str(), false)
        .expect("tokenize failed")
        .get_ids()
        .to_vec();

    eprintln!(
        "[run] baseline dense K/V, prompt tokens={}, forced decode steps={}",
        prompt_ids.len(),
        args.tokens
    );
    let baseline = run_baseline(&mut weights, &index, &prompt_ids, args.tokens);
    let forced_tokens: Vec<u32> = baseline
        .iter()
        .skip(1)
        .filter_map(|s| s.forced_token)
        .collect();

    let mut all_pass = true;
    for top_k in &args.top_k_list {
        eprintln!(
            "[run] walk-KV topK={} layers={} select_at={}",
            top_k, args.layers, args.select_at
        );
        let candidate = run_candidate(
            &mut weights,
            &index,
            &prompt_ids,
            &forced_tokens,
            *top_k,
            &args.layers,
            args.select_at,
        );

        println!();
        println!(
            "walk-KV topK={} layers={} select_at={} cos_min={:.6}",
            top_k, args.layers, args.select_at, args.cos_min
        );
        println!(
            "{:<9} {:>8} {:>8} {:>8} {:>6} {:>11} {:>7} {:>11} {:>11} {:>11} {:>10}",
            "step",
            "forced",
            "ref_next",
            "cand_next",
            "arg",
            "cos",
            "top5",
            "kl_sym",
            "p_ref",
            "p_cand",
            "max_abs"
        );
        println!("{}", "-".repeat(122));

        for (ref_step, cand_step) in baseline.iter().zip(candidate.iter()) {
            let metrics = metrics_from_logits(
                &args.prompt,
                prompt_ids.len(),
                &ref_step.logits,
                &cand_step.logits,
                5,
            );
            let forced = ref_step
                .forced_token
                .map(|t| t.to_string())
                .unwrap_or_else(|| "-".into());
            let ref_prob = token_prob(&ref_step.logits, ref_step.next_token);
            let cand_prob = token_prob(&cand_step.logits, ref_step.next_token);
            let max_abs = max_abs_diff(&ref_step.logits, &cand_step.logits);
            let arg = if metrics.argmax_match { "yes" } else { "no" };
            if !metrics.argmax_match || metrics.logit_cos < args.cos_min {
                all_pass = false;
            }
            println!(
                "{:<9} {:>8} {:>8} {:>8} {:>6} {:>11.9} {:>7.3} {:>11.4e} {:>11.6} {:>11.6} {:>10.4}",
                ref_step.label,
                forced,
                ref_step.next_token,
                cand_step.next_token,
                arg,
                metrics.logit_cos,
                metrics.top_k_jaccard,
                metrics.kl_symmetric,
                ref_prob,
                cand_prob,
                max_abs
            );
        }
    }

    clear_walk_kv_env();
    if all_pass {
        println!("\nPASS: all reported steps met argmax and cosine gates");
    } else {
        println!("\nFAIL: at least one reported step missed argmax or cosine gate");
        std::process::exit(1);
    }
}
