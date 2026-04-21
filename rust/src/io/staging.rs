//! Port of `read_staging.m` — sleep-stage CSV/TSV parser.
//!
//! Reads a delimited text file, takes two columns (time and stage string),
//! maps stage strings to numeric codes with the MATLAB default table, and
//! returns (times_seconds, stage_values).
//!
//! Stage number mapping (stage_numbers in the MATLAB source):
//!   artifact / 'A' / '6'           → 6
//!   wake / 'W' / '5'               → 5
//!   REM / 'R' / '4'                → 4
//!   N1 / 'Stage 1' / '1'           → 3
//!   N2 / 'Stage 2' / '2'           → 2
//!   N3 / 'Stage 3' / '3'           → 1
//!   Unk / 'U' / 'Unknown' / '0'    → 0
//!
//! Time column can be:
//!   (a) consecutive integer epoch numbers → times = values * epoch_dur
//!   (b) numeric seconds → times = values
//!   (c) time strings 'HH:MM:SS [AM|PM]' → seconds of day with midnight wrap

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub enum StagingError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StagingError::Io(e) => write!(f, "staging I/O: {}", e),
            StagingError::Format(s) => write!(f, "staging format: {}", s),
        }
    }
}

impl std::error::Error for StagingError {}

impl From<std::io::Error> for StagingError {
    fn from(e: std::io::Error) -> Self {
        StagingError::Io(e)
    }
}

/// Arguments matching `read_staging`.
#[derive(Debug, Clone)]
pub struct StagingOpts {
    pub time_col: usize,      // 1-based
    pub stage_col: usize,     // 1-based
    pub header_lines: usize,
    pub delimiter: char,
    pub epoch_dur: f64,
    /// Optional start-time 'HH:MM:SS [AM|PM]' string — only used for Case 3.
    pub start_time: Option<String>,
}

impl Default for StagingOpts {
    fn default() -> Self {
        Self {
            time_col: 1,
            stage_col: 2,
            header_lines: 0,
            delimiter: ',',
            epoch_dur: 30.0,
            start_time: None,
        }
    }
}

/// Parse a delimited line honoring simple quoted fields.
fn split_line(line: &str, delim: char) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    for c in line.chars() {
        if c == '"' {
            in_q = !in_q;
            continue;
        }
        if c == delim && !in_q {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    out.push(cur.trim().to_string());
    out
}

/// Stage-value lookup (lowercased strs → code).
fn stage_code(raw: &str) -> Option<i32> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    // Artifact = 6
    if matches!(s.as_str(), "art" | "artifact" | "a" | "6") {
        return Some(6);
    }
    // Wake = 5
    if matches!(s.as_str(), "wake" | "w" | "5") {
        return Some(5);
    }
    // REM = 4
    if matches!(s.as_str(), "rem" | "r" | "4") {
        return Some(4);
    }
    // N1 = 3
    if matches!(s.as_str(), "n1" | "stage 1" | "1") {
        return Some(3);
    }
    // N2 = 2
    if matches!(s.as_str(), "n2" | "stage 2" | "2") {
        return Some(2);
    }
    // N3 = 1
    if matches!(s.as_str(), "n3" | "stage 3" | "3") {
        return Some(1);
    }
    // Unknown = 0
    if matches!(s.as_str(), "unk" | "u" | "unknown" | "0") {
        return Some(0);
    }
    None
}

/// Parse 'HH:MM:SS[.ff] [AM|PM]' → seconds of day.
fn parse_time_of_day(t: &str) -> Option<f64> {
    let t = t.trim();
    if t.is_empty() {
        return None;
    }
    let mut pm = false;
    let mut is_ampm = false;
    let mut core = t.to_string();

    // Strip trailing AM/PM (case-insensitive).
    let upper = t.to_ascii_uppercase();
    if let Some(stripped) = upper.strip_suffix("AM") {
        is_ampm = true;
        core = stripped.trim().to_string();
    } else if let Some(stripped) = upper.strip_suffix("PM") {
        pm = true;
        is_ampm = true;
        core = stripped.trim().to_string();
    }

    let parts: Vec<&str> = core.split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let h: f64 = parts[0].trim().parse().ok()?;
    let m: f64 = parts[1].trim().parse().ok()?;
    let s: f64 = if parts.len() > 2 {
        parts[2].trim().parse().ok()?
    } else {
        0.0
    };
    let mut hour = h;
    if is_ampm {
        if hour == 12.0 {
            hour = 0.0;
        }
        if pm {
            hour += 12.0;
        }
    }
    Some(hour * 3600.0 + m * 60.0 + s)
}

/// Result of a staging-file parse: times in seconds + integer stage codes.
pub struct StagingOut {
    pub times: Vec<f64>,
    pub vals: Vec<f64>,
}

pub fn read_staging<P: AsRef<Path>>(
    path: P,
    opts: &StagingOpts,
) -> Result<StagingOut, StagingError> {
    let f = File::open(path.as_ref())?;
    let reader = BufReader::new(f);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if idx < opts.header_lines {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_line(&line, opts.delimiter);
        rows.push(fields);
    }

    if rows.is_empty() {
        return Err(StagingError::Format("no data rows".into()));
    }

    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if ncols < 2 {
        return Err(StagingError::Format(format!(
            "only {} column(s) detected — check delimiter / header_lines",
            ncols
        )));
    }
    if opts.time_col == 0 || opts.time_col > ncols {
        return Err(StagingError::Format(format!(
            "time_col {} out of range (ncols={})",
            opts.time_col, ncols
        )));
    }
    if opts.stage_col == 0 || opts.stage_col > ncols {
        return Err(StagingError::Format(format!(
            "stage_col {} out of range (ncols={})",
            opts.stage_col, ncols
        )));
    }

    let time_strs: Vec<String> = rows
        .iter()
        .map(|r| r.get(opts.time_col - 1).cloned().unwrap_or_default())
        .collect();
    let stage_strs: Vec<String> = rows
        .iter()
        .map(|r| r.get(opts.stage_col - 1).cloned().unwrap_or_default())
        .collect();

    // Convert time column.
    let numeric: Vec<Option<f64>> = time_strs
        .iter()
        .map(|s| s.trim().parse::<f64>().ok())
        .collect();
    let all_numeric = numeric.iter().all(|x| x.is_some());

    let times_all: Vec<f64>;
    if all_numeric {
        let vals: Vec<f64> = numeric.iter().map(|x| x.unwrap()).collect();
        let all_int = vals.iter().all(|v| v.fract() == 0.0);
        let sorted = vals.windows(2).all(|w| w[0] <= w[1]);
        if all_int && sorted && vals.len() >= 2 {
            // Median of diff == 1?
            let mut diffs: Vec<f64> = vals.windows(2).map(|w| w[1] - w[0]).collect();
            diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = diffs[diffs.len() / 2];
            if (med - 1.0).abs() < f64::EPSILON {
                if opts.epoch_dur <= 0.0 {
                    return Err(StagingError::Format("epoch_dur must be > 0".into()));
                }
                times_all = vals.iter().map(|v| v * opts.epoch_dur).collect();
            } else {
                times_all = vals;
            }
        } else {
            times_all = vals;
        }
    } else {
        // Case 3: time strings.
        let mut secs: Vec<f64> = Vec::with_capacity(time_strs.len());
        for (i, s) in time_strs.iter().enumerate() {
            match parse_time_of_day(s) {
                Some(v) => secs.push(v),
                None => {
                    return Err(StagingError::Format(format!(
                        "could not parse time string at row {}: '{}'",
                        i + 1,
                        s
                    )))
                }
            }
        }
        // Midnight wrap.
        let mut day_off = 0.0;
        let mut out = Vec::with_capacity(secs.len());
        let mut prev = None::<f64>;
        for v in secs.iter() {
            if let Some(p) = prev {
                if *v < p {
                    day_off += 86400.0;
                }
            }
            out.push(*v + day_off);
            prev = Some(*v);
        }

        // start_offset: times_seconds(1) - seconds(timeofday(datetime(start_time)))
        let start_offset = match &opts.start_time {
            Some(st) => {
                let st_sec = parse_time_of_day(st).ok_or_else(|| {
                    StagingError::Format(format!("bad start_time: {}", st))
                })?;
                let off = out[0] - st_sec;
                if off < 0.0 {
                    return Err(StagingError::Format(
                        "start_time is later than first time point".into(),
                    ));
                }
                off
            }
            None => 0.0,
        };

        let t0 = out[0];
        times_all = out.iter().map(|v| v - t0 + start_offset).collect();
    }

    // Stage codes; drop unmatched rows (MATLAB does `~unmatched_idx`).
    let mut times: Vec<f64> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for (t, s) in times_all.iter().zip(stage_strs.iter()) {
        if let Some(code) = stage_code(s) {
            times.push(*t);
            vals.push(code as f64);
        }
    }

    // MATLAB asserts unique time stamps.
    for w in times.windows(2) {
        if w[0] == w[1] {
            return Err(StagingError::Format(
                "multiple stages identified at the exact same time stamp".into(),
            ));
        }
    }

    // If start_time was provided and first time != 0, prepend (0, 0).
    if opts.start_time.is_some() && !times.is_empty() && times[0] != 0.0 {
        times.insert(0, 0.0);
        vals.insert(0, 0.0);
    }

    Ok(StagingOut { times, vals })
}
