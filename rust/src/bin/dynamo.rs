//! `dynamo` CLI: standalone Rust DYNAM-O extract runner.
//!
//! Scope:
//!   Input:  a pre-computed multitaper spectrogram as three .npy files
//!           (spect.npy, stimes.npy, sfreqs.npy)
//!   Output: stats.csv with one row per detected TF-peak.
//!
//! This binary runs the extract step only. The other stages of the MATLAB
//! and pydynamo pipelines (baseline subtraction, pass-1/pass-2
//! spectrograms, refine, SO-power/phase histograms) are available as
//! dynamo_rs modules but are not wired into this CLI.
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
    default_extract_params, run_extract_from_spectrogram, write_stats_csv, StatsCsvPrecision,
    StatsCsvProvenance,
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
    /// before .npy export. numpy C-order works as-is.
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
    /// Segment length in seconds.
    #[arg(long, default_value_t = 30.0)]
    seg_time: f64,
    /// Frequency downsample factor for the watershed.
    #[arg(long, default_value_t = 2)]
    downsample_f: u32,
    /// Time downsample factor for the watershed.
    #[arg(long, default_value_t = 2)]
    downsample_t: u32,
    /// Merge threshold: regions merge while the merge weight exceeds this.
    #[arg(long, default_value_t = 11.0)]
    merge_thresh: f64,
    /// Fraction of each peak's volume kept by trimming.
    #[arg(long, default_value_t = 0.8)]
    trim_vol: f64,
    /// Minimum peak duration in seconds.
    #[arg(long, default_value_t = 0.5)]
    dur_min: f64,
    /// Maximum peak duration in seconds.
    #[arg(long, default_value_t = 5.0)]
    dur_max: f64,
    /// Minimum peak bandwidth in Hz.
    #[arg(long, default_value_t = 2.0)]
    bw_min: f64,
    /// Maximum peak bandwidth in Hz.
    #[arg(long, default_value_t = 15.0)]
    bw_max: f64,
    /// Border-fill mode after upsampling the labels. 0 (default) paints
    /// in label order as MATLAB does. N > 0 runs a skimage-style
    /// `expand_labels` fill out to N pixels.
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

    // Parameters with no CLI flag keep the runDYNAMO defaults.
    let defaults = default_extract_params();
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
        freq_min: defaults.freq_min,
        freq_max: defaults.freq_max,
        ht_db_min: defaults.ht_db_min,
        expand_labels_distance: args.expand_labels_distance,
    };

    let (peaks, labels) = run_extract_from_spectrogram(
        spect.view(), stimes.view(), sfreqs.view(), &params,
    )?;

    eprintln!("extracted {} peaks in {:.1}s", peaks.len(), t0.elapsed().as_secs_f64());

    // Bare extract with no staging, so SO-power/phase/stage columns are empty.
    // Defaults match `OutputOpts` at fs=100 (time at 8 sig figs covers
    // recordings up to ~28 h with sub-sample precision).
    let precision = StatsCsvPrecision::default();
    let provenance = StatsCsvProvenance {
        writer: "dynamo-rs-bin",
        writer_version: dynamo_rs::build_info::VERSION,
        subject_id: None,
        // The kernel sits below the settings crate, so it has no
        // computation-semantics number to stamp.
        semantics: None,
    };
    write_stats_csv(&peaks, &[], &[], &[], precision, Some(&provenance), &args.out)?;
    eprintln!("wrote {}", args.out.display());

    if let Some(labels_path) = &args.labels_out {
        use ndarray_npy::WriteNpyExt;
        let labels_file = std::fs::File::create(labels_path)?;
        labels.write_npy(labels_file)?;
        eprintln!("wrote {}", labels_path.display());
    }

    Ok(())
}
