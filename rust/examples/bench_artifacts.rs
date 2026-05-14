//! Stand-alone bench: load an EDF, pick channel 0, run detect_artifacts
//! with sub-step timing enabled. Run with:
//!
//! ```text
//! DYNAMO_ARTIFACT_TIMING=1 cargo run --release \
//!     --example bench_artifacts -- /path/to/file.edf [channel_idx]
//! ```
//!
//! Reports total wall-clock + the per-band sub-step breakdown
//! (filter_design / filter / hilbert / smooth / detrend / zscore).

use std::env;
use std::time::Instant;

use dynamo_rs::artifacts::{detect_artifacts, ArtifactOpts};
use dynamo_rs::io::edf::read_edf_all;

fn main() {
    let mut args = env::args().skip(1);
    let edf = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: bench_artifacts <edf_path> [channel_idx]");
            std::process::exit(2);
        }
    };
    let ch_idx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    eprintln!("loading {} …", edf);
    let t = Instant::now();
    let edf_data = read_edf_all(&edf).expect("read_edf_all");
    let load_s = t.elapsed().as_secs_f64();
    let n_chan = edf_data.data.len();
    eprintln!(
        "loaded in {:.2}s — {} channel(s), data_record_duration={} s, num_data_records={}",
        load_s, n_chan, edf_data.header.data_record_duration, edf_data.header.num_data_records
    );
    for (i, sh) in edf_data.signals.iter().enumerate() {
        eprintln!(
            "  [{}] {} — fs={} Hz, N={} samples",
            i, sh.signal_labels, sh.sampling_frequency, edf_data.data[i].len()
        );
    }

    if ch_idx >= n_chan {
        eprintln!("channel index {} out of range", ch_idx);
        std::process::exit(2);
    }
    let data = &edf_data.data[ch_idx];
    let fs = edf_data.signals[ch_idx].sampling_frequency;
    eprintln!("\nrunning detect_artifacts on channel {} (N={}, fs={} Hz) …\n", ch_idx, data.len(), fs);

    let opts = ArtifactOpts::default();

    // Warm-up run (toggles allocator caches, JIT-like effects).
    let _ = detect_artifacts(data, fs, &opts);

    // Timed runs.
    let mut times = Vec::new();
    for i in 0..3 {
        let t = Instant::now();
        let mask = detect_artifacts(data, fs, &opts);
        let s = t.elapsed().as_secs_f64();
        let flagged = mask.iter().filter(|&&b| b).count();
        let pct = 100.0 * flagged as f64 / mask.len() as f64;
        eprintln!(
            "trial #{}: {:.3}s   flagged {} / {} samples ({:.2}%)",
            i + 1, s, flagged, mask.len(), pct
        );
        times.push(s);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("\nmean over {} trials: {:.3}s   best: {:.3}s", times.len(), mean, best);
}
