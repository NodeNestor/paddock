//! Benchmark our engine via the Generator trait (works for any family/backend
//! it supports). Currently: gpt-oss, qwen35 and granite on CUDA (GPU-only).
//! Note this is the IN-PROCESS lane - serving runs go over HTTP against a real
//! runner (`paddock-bench --endpoint`), which is what any family with a batch
//! lane actually wants measured.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use paddock_engine::generator::Generator;
use paddock_models::mapped::MappedGguf;

use crate::timings::Timings;

pub fn run(
    model_path: &Path,
    device: &str,
    pack: Option<&Path>,
    prompt: &[u32],
    decode_tokens: usize,
    warmup: usize,
) -> Result<Timings, String> {
    let map = MappedGguf::open(model_path).map_err(|e| e.to_string())?;
    let arch = map.gguf().architecture().unwrap_or("?").to_owned();

    let t_load = Instant::now();
    let mut eng: Box<dyn Generator> = build(&arch, device, &map, pack)?;
    let load = t_load.elapsed();

    // prefill: feed the whole prompt, time it; last token's forward is TTFT's tail
    let t_pref = Instant::now();
    eng.reset();
    let mut logits = Vec::new();
    for &tok in prompt {
        logits = eng.forward(tok).map_err(|e| e.to_string())?;
    }
    let prefill = t_pref.elapsed();

    // TTFT = prefill + one decode step (first generated token)
    let mut next = argmax(&logits);
    let t_first = Instant::now();
    logits = eng.forward(next).map_err(|e| e.to_string())?;
    let ttft = prefill + t_first.elapsed();

    // warmup decode (discarded)
    for _ in 0..warmup {
        next = argmax(&logits);
        logits = eng.forward(next).map_err(|e| e.to_string())?;
    }

    // timed decode
    let t_dec = Instant::now();
    for _ in 0..decode_tokens {
        next = argmax(&logits);
        logits = eng.forward(next).map_err(|e| e.to_string())?;
    }
    let decode = t_dec.elapsed();

    Ok(Timings {
        runner: format!("paddock/{arch}"),
        load,
        prefill_tokens: prompt.len(),
        prefill,
        ttft,
        decode_tokens,
        decode,
    })
}

fn build(
    arch: &str,
    device: &str,
    map: &MappedGguf,
    pack: Option<&Path>,
) -> Result<Box<dyn Generator>, String> {
    match (arch, device) {
        ("gpt-oss", "cuda") => {
            let pack = pack.ok_or("cuda device needs --pack")?;
            let exec = Arc::new(
                paddock_engine::gpu::GpuExecutor::new(0, pack).map_err(|e| e.to_string())?,
            );
            let m = paddock_engine::gpu_model::gpt_oss::GpuGptOss::load(exec, map, 4096)
                .map_err(|e| e.to_string())?;
            Ok(Box::new(m))
        }
        ("qwen35", "cuda") => {
            let pack = pack.ok_or("cuda device needs --pack")?;
            let exec = Arc::new(
                paddock_engine::gpu::GpuExecutor::new(0, pack).map_err(|e| e.to_string())?,
            );
            // env resolved here (a binary's config layer) - the engine takes
            // fp8_native as an explicit load option, never reads env itself
            let fp8 = std::env::var_os("PADDOCK_FP8_NATIVE").map(std::path::PathBuf::from);
            let m = paddock_engine::gpu_model::qwen35::GpuQwen35::load_with(
                exec,
                map,
                4096,
                fp8.as_deref(),
            )
            .map_err(|e| e.to_string())?;
            Ok(Box::new(m))
        }
        ("granite", "cuda") => {
            let pack = pack.ok_or("cuda device needs --pack")?;
            let exec = Arc::new(
                paddock_engine::gpu::GpuExecutor::new(0, pack).map_err(|e| e.to_string())?,
            );
            let m = paddock_engine::gpu_model::granite::GpuGranite::load(exec, map, 4096)
                .map_err(|e| e.to_string())?;
            Ok(Box::new(m))
        }
        _ => Err(format!("unsupported (arch={arch}, device={device})")),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}
