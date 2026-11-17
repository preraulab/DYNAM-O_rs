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

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum StagingError {
    Io(std::io::Error),
    NotFound(std::path::PathBuf),
    ReadFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Format(String),
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StagingError::Io(e) => write!(f, "staging I/O: {}", e),
            StagingError::NotFound(p) => write!(f, "staging file not found: {}", p.display()),
            StagingError::ReadFailed { path, source } => {
                write!(f, "cannot read staging file {}: {}", path.display(), source)
            }
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
    /// Length of the recording in seconds, when the caller knows it — only
    /// used for Case 3 together with `start_time`. It lets the reader tell
    /// a first scored epoch after midnight from staging that starts before
    /// the recording does (see the next-day rule in `read_staging`).
    pub recording_duration: Option<f64>,
    /// Per-stage label aliases in canonical order
    /// `[artifact, wake, REM, N1, N2, N3, unknown]`. Each inner list is
    /// the set of strings (case-insensitive) that map to that stage —
    /// matches MATLAB `read_staging`'s `stage_vals` parameter. A
    /// length-7 list overrides the built-in default table; any other
    /// length (incl. empty) falls back to the defaults.
    pub stage_vals: Vec<Vec<String>>,
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
            recording_duration: None,
            stage_vals: Vec::new(),
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

/// MATLAB `read_staging` default `stage_vals` — 7 alias lists in
/// canonical order [artifact, wake, REM, N1, N2, N3, unknown].
const DEFAULT_STAGE_VALS: [&[&str]; 7] = [
    &["art", "artifact", "a", "6"],
    &["wake", "w", "5"],
    &["rem", "r", "4"],
    &["n1", "stage 1", "1"],
    &["n2", "stage 2", "2"],
    &["n3", "stage 3", "3"],
    &["unk", "u", "unknown", "0"],
];

/// Numeric code for each stage slot (MATLAB `stage_numbers`):
/// artifact=6, wake=5, REM=4, N1=3, N2=2, N3=1, unknown=0.
const STAGE_CODES: [i32; 7] = [6, 5, 4, 3, 2, 1, 0];

/// Build the case-insensitive alias→code lookup. A length-7 `custom`
/// list overrides the defaults; any other length falls back to the
/// MATLAB default table. When two stage slots share an alias the
/// later (higher-numbered) slot wins — matches MATLAB's overwrite order.
fn build_stage_map(custom: &[Vec<String>]) -> HashMap<String, i32> {
    let mut map: HashMap<String, i32> = HashMap::new();
    if custom.len() == 7 {
        for (i, aliases) in custom.iter().enumerate() {
            for a in aliases {
                let key = a.trim().to_ascii_lowercase();
                if !key.is_empty() {
                    map.insert(key, STAGE_CODES[i]);
                }
            }
        }
    } else {
        for (i, aliases) in DEFAULT_STAGE_VALS.iter().enumerate() {
            for a in *aliases {
                map.insert((*a).to_string(), STAGE_CODES[i]);
            }
        }
    }
    map
}

/// Strip a leading ISO 8601 / RFC 3339 date prefix, returning the
/// time-of-day portion. Handles both separators:
///   - `2014-06-26T01:29:29.420000` (T separator — strict ISO 8601)
///   - `2014-06-26 01:29:29.420000` (space separator — RFC 3339 §5.6)
///
/// The space form is only stripped when the part before the space
/// looks like a date (contains `-` or `/` and otherwise only digits/
/// separators), so a bare `"12:00:00 AM"` time-of-day is not
/// mis-handled — the AM/PM-side space stays intact and the AM/PM
/// branch downstream sees it.
fn strip_date_prefix(s: &str) -> &str {
    if let Some(i) = s.rfind('T') {
        return &s[i + 1..];
    }
    if let Some(i) = s.find(' ') {
        let prefix = &s[..i];
        let looks_date = (prefix.contains('-') || prefix.contains('/'))
            && prefix.chars().all(|c| c.is_ascii_digit() || c == '-' || c == '/');
        if looks_date {
            return s[i + 1..].trim_start();
        }
    }
    s
}

/// Strip a trailing timezone marker (`Z` or `+HH:MM` / `-HH:MM` /
/// `+HHMM` / `-HHMM`). We only care about wall-clock time, so any
/// timezone offset is silently dropped. Heuristic — only treats a
/// trailing `+`/`-` as a TZ marker when it's followed by digits (so
/// it doesn't strip a `-` inside a negative second field, which
/// wouldn't be valid anyway).
fn strip_tz_suffix(s: &str) -> &str {
    let s = s.strip_suffix('Z').unwrap_or(s);
    // Walk back from the end to find a `+` or `-` that's not at the
    // start. If the chars after it are all digits/colons, treat as TZ.
    let bytes = s.as_bytes();
    for i in (1..bytes.len()).rev() {
        let c = bytes[i];
        if c == b'+' || c == b'-' {
            let tail = &s[i + 1..];
            if !tail.is_empty()
                && tail.chars().all(|c| c.is_ascii_digit() || c == ':')
            {
                return &s[..i];
            }
            break;
        }
        // If we hit something that can't be inside a TZ suffix and
        // we haven't seen the +/- yet, give up.
        if !(c.is_ascii_digit() || c == b':') {
            break;
        }
    }
    s
}

/// Parse 'HH:MM:SS[.ff] [AM|PM]' → seconds of day. Also accepts EDF's
/// "hh.mm.ss" (dot-separated). ISO 8601 datetime inputs
/// (`2014-06-26T01:29:29.420000`, with or without trailing TZ) are
/// also accepted — the date and timezone are stripped before parsing,
/// and only the wall-clock time-of-day is returned. Returns `None`
/// for empty / unparseable input.
///
/// Public so the init wizard can validate EDF `recording_starttime` at
/// Step 1 (header-load time), rather than waiting until staging
/// alignment in Step 4 surfaces the error.
pub fn parse_time_of_day(t: &str) -> Option<f64> {
    let t = t.trim();
    if t.is_empty() {
        return None;
    }
    // ISO 8601 / RFC 3339 date-time inputs (`2014-06-26T01:29:29.420000`,
    // `2014-06-26 01:29:29.420000+05:30`) — we only care about the
    // wall-clock time-of-day; the date drops and the timezone is
    // ignored. The midnight-wrap detector in `read_staging` reconstructs
    // cross-day runs from time-of-day deltas, so dropping the date
    // here loses no information for the cohort use case.
    let t = strip_date_prefix(t);
    let t = strip_tz_suffix(t);

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

    // Staging clock times use "HH:MM:SS"; EDF start times use "hh.mm.ss" (dots,
    // per the EDF spec). Split on whichever separator is present — prefer ':'
    // so a fractional-second "HH:MM:SS.s" keeps its decimal.
    let sep = if core.contains(':') { ':' } else { '.' };
    let parts: Vec<&str> = core.split(sep).collect();
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
    // Read as bytes + lossy UTF-8 decode. Lab scoring software often writes
    // staging files in Windows-1252 / Latin-1 (degree signs, em-dashes in
    // the metadata header), which `read_to_string` rejects. The actual data
    // rows are pure ASCII (Wake/N1/N2/...), so a lossy decode produces
    // correct results — invalid bytes become U+FFFD and only show up in
    // the metadata header that we skip anyway.
    let raw = std::fs::read(path.as_ref()).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            StagingError::NotFound(path.as_ref().to_path_buf())
        } else {
            StagingError::ReadFailed { path: path.as_ref().to_path_buf(), source: e }
        }
    })?;
    let text = String::from_utf8_lossy(&raw);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if idx < opts.header_lines {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_line(line, opts.delimiter);
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
                    // Common failure mode: header_lines was one short and the
                    // column-header label (e.g. "Start Time") got fed in as
                    // data. Detect that and suggest the fix in the message.
                    let hint = if s.chars().any(|c| c.is_alphabetic()) {
                        " (looks like a column-header label — increase header_lines)"
                    } else {
                        ""
                    };
                    return Err(StagingError::Format(format!(
                        "could not parse time string at row {}: '{}'{}",
                        i + 1, s, hint
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
                    // `st` is the EDF's recording_starttime field. EDF spec
                    // is "hh.mm.ss"; common in-the-wild deviations are
                    // "HH:MM:SS", AM/PM suffixes, and partial fills with
                    // NULL padding ("22:07:2\0") from non-compliant writers.
                    StagingError::Format(format!(
                        "EDF recording_starttime is malformed ({:?}); expected hh.mm.ss or HH:MM:SS. \
                         If the EDF was de-identified or written by a non-spec writer, this field \
                         may have been corrupted",
                        st
                    ))
                })?;
                // Next-day rule. Clock times carry no date, so a first
                // staging row with an earlier clock time than the recording
                // start (start 21:44, first scored epoch 01:55) is read as
                // falling on the next day: the whole staging time vector
                // moves forward by 24 h. The row-to-row wrap above cannot
                // see this crossing because it happens before row 1.
                let first_row_is_before_start = out[0] < st_sec;
                if first_row_is_before_start {
                    for v in out.iter_mut() {
                        *v += 86400.0;
                    }
                }
                let off = out[0] - st_sec;
                if off < 0.0 {
                    return Err(StagingError::Format(
                        "start_time is later than first time point".into(),
                    ));
                }
                // The other way to get an earlier clock time is staging that
                // really does begin before the recording (start 21:44,
                // first row 21:40). Read as the next day, that row lands
                // almost 24 h in, past the end of the recording. When the
                // recording length is known, that reading is refused.
                if first_row_is_before_start {
                    if let Some(duration) = opts.recording_duration {
                        if off >= duration {
                            return Err(StagingError::Format(format!(
                                "the first staging time is earlier than the recording start time. \
                                 Read as the next day it would fall {:.2} h after the start, past \
                                 the end of the {:.2} h recording. Staging cannot begin before the \
                                 recording does: check the staging time column and the EDF start time",
                                off / 3600.0,
                                duration / 3600.0
                            )));
                        }
                    }
                }
                off
            }
            None => 0.0,
        };

        let t0 = out[0];
        times_all = out.iter().map(|v| v - t0 + start_offset).collect();
    }

    // Stage codes; drop unmatched rows (MATLAB does `~unmatched_idx`).
    let stage_map = build_stage_map(&opts.stage_vals);
    let mut times: Vec<f64> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for (t, s) in times_all.iter().zip(stage_strs.iter()) {
        let key = s.trim().to_ascii_lowercase();
        if let Some(&code) = stage_map.get(&key) {
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

#[cfg(test)]
mod tests {
    use super::parse_time_of_day;

    #[test]
    fn time_of_day_accepts_colon_and_dot_formats() {
        // EDF-spec start time "hh.mm.ss" (dots) — the format that regressed.
        assert_eq!(parse_time_of_day("22.12.52"), Some(22.0 * 3600.0 + 12.0 * 60.0 + 52.0));
        // Colon start time (some exporters write the non-standard form).
        assert_eq!(parse_time_of_day("21:46:23"), Some(21.0 * 3600.0 + 46.0 * 60.0 + 23.0));
        // Staging clock times with AM/PM.
        assert_eq!(parse_time_of_day("10:19:22 PM"), Some(22.0 * 3600.0 + 19.0 * 60.0 + 22.0));
        assert_eq!(parse_time_of_day("12:00:00 AM"), Some(0.0)); // midnight
        assert_eq!(parse_time_of_day("12:00:00 PM"), Some(12.0 * 3600.0)); // noon
        // Colon form keeps fractional seconds (separator stays ':').
        assert_eq!(parse_time_of_day("00:00:01.5"), Some(1.5));
        // Unparseable → None.
        assert_eq!(parse_time_of_day(""), None);
        assert_eq!(parse_time_of_day("notatime"), None);
    }

    /// ISO 8601 / RFC 3339 datetime inputs: date prefix and TZ suffix
    /// must both be stripped so the existing HH:MM:SS[.fff] path takes
    /// over. The actual report was a Compumedics-derived staging CSV
    /// whose time column contained values like
    /// `2014-06-26T01:29:29.420000` — date drops, microseconds keep
    /// their fractional component.
    #[test]
    fn time_of_day_accepts_iso8601_datetime() {
        let h_m_s = 1.0 * 3600.0 + 29.0 * 60.0 + 29.42;
        // T separator, no TZ — the form from the user report.
        assert_eq!(parse_time_of_day("2014-06-26T01:29:29.420000"), Some(h_m_s));
        // T separator, Z suffix (UTC).
        assert_eq!(parse_time_of_day("2014-06-26T01:29:29.420000Z"), Some(h_m_s));
        // T separator, +HH:MM TZ.
        assert_eq!(parse_time_of_day("2014-06-26T01:29:29.420000+05:30"), Some(h_m_s));
        // T separator, -HHMM TZ (no colon).
        assert_eq!(parse_time_of_day("2014-06-26T01:29:29.420000-0400"), Some(h_m_s));
        // Space separator (RFC 3339 §5.6).
        assert_eq!(parse_time_of_day("2014-06-26 01:29:29.420000"), Some(h_m_s));
        // / date separator + space.
        assert_eq!(parse_time_of_day("2014/06/26 01:29:29.420000"), Some(h_m_s));
        // Date only (no time) is unparseable.
        assert_eq!(parse_time_of_day("2014-06-26"), None);
    }

    /// The space-separator branch in `strip_date_prefix` must NOT
    /// strip the AM/PM space of a bare time-of-day. Regression guard
    /// for `12:00:00 AM` etc.
    /// Write a two-column staging file with clock times and read it back
    /// against the given recording start and length.
    fn read_clock_staging(
        file_tag: &str,
        clock_times: &[&str],
        start_time: &str,
        recording_duration: Option<f64>,
    ) -> Result<super::StagingOut, super::StagingError> {
        let path = std::env::temp_dir().join(format!(
            "dynamo_staging_next_day_{}_{}.csv",
            file_tag,
            std::process::id()
        ));
        let mut text = String::new();
        for clock_time in clock_times {
            text.push_str(clock_time);
            text.push_str(",N2\n");
        }
        std::fs::write(&path, text).expect("write staging file");
        let opts = super::StagingOpts {
            start_time: Some(start_time.to_string()),
            recording_duration,
            ..Default::default()
        };
        let result = super::read_staging(&path, &opts);
        let _ = std::fs::remove_file(&path);
        result
    }

    #[test]
    fn first_scored_epoch_after_midnight_is_read_as_the_next_day() {
        // Recording starts 21:44, scoring starts 01:55: 4 h 11 min in.
        let eight_hours = 8.0 * 3600.0;
        let staging = read_clock_staging(
            "after_midnight",
            &["01:55:00", "01:55:30", "01:56:00"],
            "21:44:00",
            Some(eight_hours),
        )
        .expect("an after-midnight start is valid");
        let first_scored = 4.0 * 3600.0 + 11.0 * 60.0;
        assert!(staging.times.contains(&first_scored), "times: {:?}", staging.times);
        assert!(staging.times.contains(&(first_scored + 60.0)));
    }

    #[test]
    fn staging_that_begins_before_the_recording_is_refused() {
        // Recording starts 21:44, staging starts 21:40. Read as the next
        // day this is 23 h 56 min in, past the end of an 8 h recording.
        let eight_hours = 8.0 * 3600.0;
        let result = read_clock_staging(
            "before_start",
            &["21:40:00", "21:40:30", "21:41:00"],
            "21:44:00",
            Some(eight_hours),
        );
        let message = match result {
            Ok(_) => panic!("staging before the recording start must be an error"),
            Err(e) => format!("{}", e),
        };
        assert!(message.contains("earlier than the recording start time"), "{}", message);
    }

    #[test]
    fn without_a_recording_length_an_earlier_clock_time_is_the_next_day() {
        // The reader cannot tell the two cases apart without the length,
        // so it keeps the next-day reading.
        let staging = read_clock_staging(
            "no_length",
            &["21:40:00", "21:40:30"],
            "21:44:00",
            None,
        )
        .expect("no recording length, no refusal");
        let next_day_offset = 86400.0 - 4.0 * 60.0;
        assert!(staging.times.contains(&next_day_offset), "times: {:?}", staging.times);
    }

    #[test]
    fn time_of_day_ampm_with_space_unchanged() {
        assert_eq!(parse_time_of_day("12:00:00 AM"), Some(0.0));
        assert_eq!(parse_time_of_day("12:00:00 PM"), Some(12.0 * 3600.0));
        assert_eq!(parse_time_of_day("10:19:22 PM"), Some(22.0 * 3600.0 + 19.0 * 60.0 + 22.0));
    }
}
