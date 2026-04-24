//! `dynamo` CLI — standalone Rust DYNAM-O extract runner.
//!
//! Today's scope (v1):
//!   Input:  a pre-computed multitaper spectrogram as three .npy files
//!           (spect.npy, stimes.npy, sfreqs.npy)
//!   Output: stats.csv with one row per detected TF-peak.
//!
//! The MATLAB / pydynamo pipelines both go much further — baseline
//! subtraction, pass-1/pass-2 spectrograms, refine, SO-power/phase
//! histograms — but each stage is available as a dynamo_rs module, so this
//! CLI will grow. The extract step is the hot path; the CLI already proves
//! the end-to-end Rust pipeline composes.
//!
//! Usage:
//!   dynamo extract --spect spect.npy --stimes stimes.npy \
//!                  --sfreqs sfreqs.npy --out stats.csv
//!
//!   dynamo extract --spect ... --merge-thresh 8 --trim-vol 0.9 ...
//!
//! Build:  cargo build --release --bin dynamo
//! Run:    ./target/release/dynamo extract --help

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;

use dynamo_rs::extract_pipeline::ExtractParams;
use dynamo_rs::pipeline::{
    default_extract_params, run_extract_from_spectrogram, write_stats_csv,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "DYNAM-O pure-Rust pipeline CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the TF-peak extraction pipeline on a pre-computed spectrogram.
    Extract(ExtractArgs),
}

#[derive(Parser, Debug)]
struct ExtractArgs {
    /// Path to (F, T) float64 .npy spectrogram (baseline already applied
    /// by the caller if desired). MATLAB column-major needs a transpose
    /// before .npy export; numpy C-order works as-is.
    #[arg(long)]
    spect: PathBuf,
    /// Path to (T,) float64 .npy time axis.
    #[arg(long)]
    stimes: PathBuf,
    /// Path to (F,) float64 .npy frequency axis.
    #[arg(long)]
    sfreqs: PathBuf,
    /// Path to write the peak stats CSV.
    #[arg(long)]
    out: PathBuf,
    /// (Optional) also write the F×T int64 label image as .npy.
    #[arg(long)]
    labels_out: Option<PathBuf>,

    // ------- ExtractParams overrides (defaults match runDYNAMO). -------
    #[arg(long, default_value_t = 30.0)]
    seg_time: f64,
    #[arg(long, default_value_t = 2)]
    downsample_f: u32,
    #[arg(long, default_value_t = 2)]
    downsample_t: u32,
    #[arg(long, default_value_t = 11.0)]
    merge_thresh: f64,
    #[arg(long, default_value_t = 0.8)]
    trim_vol: f64,
    #[arg(long, default_value_t = 0.5)]
    dur_min: f64,
    #[arg(long, default_value_t = 5.0)]
    dur_max: f64,
    #[arg(long, default_value_t = 2.0)]
    bw_min: f64,
    #[arg(long, default_value_t = 15.0)]
    bw_max: f64,
    /// `expand_labels` BFS distance. 0 = MATLAB-paint default (closes the
    /// +2% peak-count gap vs MATLAB); >0 = skimage-style fill.
    #[arg(long, default_value_t = 0)]
    expand_labels_distance: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Extract(args) => run_extract(args),
    }
}

fn run_extract(args: ExtractArgs) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();

    // Load .npy inputs. ReadNpyExt infers shape from the .npy header.
    let spect = Array2::<f64>::read_npy(std::fs::File::open(&args.spect)?)?;
    let stimes = Array1::<f64>::read_npy(std::fs::File::open(&args.stimes)?)?;
    let sfreqs = Array1::<f64>::read_npy(std::fs::File::open(&args.sfreqs)?)?;

    if spect.dim() != (sfreqs.len(), stimes.len()) {
        return Err(format!(
            "spect shape {:?} does not match (|sfreqs|={}, |stimes|={})",
            spect.dim(),
            sfreqs.len(),
            stimes.len()
        )
        .into());
    }

    eprintln!(
        "loaded spect {:?}, stimes.len={}, sfreqs.len={}",
        spect.dim(),
        stimes.len(),
        sfreqs.len()
    );

    let params = ExtractParams {
        seg_time: args.seg_time,
        downsample_f: args.downsample_f as usize,
        downsample_t: args.downsample_t as usize,
        merge_thresh: args.merge_thresh,
        max_merges: f64::INFINITY,
        trim_vol_thresh: args.trim_vol,
        trim_shift_val: f64::NAN,
        dur_min: args.dur_min,
        dur_max: args.dur_max,
        bw_min: args.bw_min,
        bw_max: args.bw_max,
        freq_min: default_extract_params().freq_min,
        freq_max: default_extract_params().freq_max,
        ht_db_min: default_extract_params().ht_db_min,
        expand_labels_distance: args.expand_labels_distance,
    };

    let (peaks, labels) = run_extract_from_spectrogram(
        spect.view(), stimes.view(), sfreqs.view(), &params,
    )?;

    eprintln!("extracted {} peaks in {:.1}s", peaks.len(), t0.elapsed().as_secs_f64());

    write_stats_csv(&peaks, &args.out)?;
    eprintln!("wrote {}", args.out.display());

    if let Some(lab_path) = &args.labels_out {
        use ndarray_npy::WriteNpyExt;
        let f = std::fs::File::create(lab_path)?;
        labels.write_npy(f)?;
        eprintln!("wrote {}", lab_path.display());
    }

    Ok(())
}
