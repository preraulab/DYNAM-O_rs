//! Loading a recording for a run: one channel of an EDF (with channel
//! derivations and references) plus its staging, aligned to the EDF clock.
//!
//! The DYNAM-O app, its command-line tool and the Python wrapper all load
//! recordings through [`load_edf_bundle`], so a run starts from the same
//! samples and hypnogram everywhere.

/// A recording could not be loaded. The message names the step ("EDF:",
/// "select_channel:", "staging:") and the reason.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

/// Canonical EDF + staging loader, shared by the GUI job runner and the
/// headless `dynamo-cli` so both produce byte-identical pipeline inputs
/// (same channel/reference selection, staging clock-alignment to the EDF
/// recording start, no-staging all-N2 synthesis, and stage-label mapping).
#[allow(clippy::too_many_arguments)]
pub fn load_edf_bundle(
    edf_path: &std::path::Path,
    staging_path: Option<&std::path::Path>,
    channel: &str,
    references: &[String],
    time_col: usize,
    stage_col: usize,
    header_lines: usize,
    delimiter: &str,
    epoch_dur: f64,
    stage_labels: &[Vec<String>],
) -> Result<(Vec<f64>, f64, Vec<f64>, Vec<f64>), LoadError> {
    // Single-channel convenience wrapper: read the signals this channel is
    // made from, build the channel, load or synthesize staging. For
    // multi-channel work prefer one `read_edf_signals` for all the channels
    // and `select_channel` per channel, so a subject's EDF (possibly on a
    // slow network drive) is read only once.
    let reference_strs: Vec<&str> = references.iter().map(|r| r.as_str()).collect();
    let edf_data = match leaf_labels_for_read(&[channel], &reference_strs) {
        Ok(labels) => {
            let label_strs: Vec<&str> = labels.iter().map(|l| l.as_str()).collect();
            read_edf_signals(edf_path, &label_strs)?
        }
        // The channel expression could not be parsed. Read every signal and
        // let `select_channel` below report the problem in its own words.
        Err(_) => read_edf_data(edf_path)?,
    };
    let (data, fs) = select_channel(&edf_data, channel, references)?;
    let (stage_times, stage_vals) = match staging_path {
        Some(sp) => load_staging_file(
            &edf_data.header,
            sp,
            time_col,
            stage_col,
            header_lines,
            delimiter,
            epoch_dur,
            stage_labels,
        )?,
        None => synth_n2_staging(data.len(), fs, epoch_dur),
    };
    Ok((data, fs, stage_times, stage_vals))
}

/// Read an entire EDF (header + all channels) once — the expensive, often
/// network-bound step. Pair with [`select_channel`] to process many channels
/// from a single read, matching DYNAM-O's load-subject-then-loop-channels
/// structure instead of re-reading the file per channel.
///
/// Wraps the dynamo_rs EDF toolbox (Rust port of MATLAB `read_EDF.m`).
pub fn read_edf_data(
    edf_path: &std::path::Path,
) -> Result<crate::io::edf::EdfData, LoadError> {
    crate::io::edf::read_edf_all(edf_path).map_err(|e| LoadError(format!("EDF: {}", e)))
}

/// Partial-load: read only the signals whose labels are in `wanted_labels`
/// (case-insensitive, trimmed). Labels not found in the EDF header are
/// silently absent from the result — `select_channel` will surface them
/// downstream as `channel not found, skipping`, matching MATLAB
/// `read_EDF`'s `'Channels'` warn-and-skip behavior.
///
/// Same `EdfData` shape as [`read_edf_data`]; just smaller in memory.
/// Use this whenever the cohort's channels are literal EDF labels (the
/// dominant production case) to avoid loading 4 unused signals when the
/// user only asked for 4 of 8.
pub fn read_edf_signals(
    edf_path: &std::path::Path,
    wanted_labels: &[&str],
) -> Result<crate::io::edf::EdfData, LoadError> {
    crate::io::edf::read_edf_with_signals(edf_path, wanted_labels)
        .map_err(|e| LoadError(format!("EDF: {}", e)))
}

/// Leaf labels to load for a set of channel expressions, INCLUDING the raw
/// expression strings themselves. A channel that is a literal EDF label — a
/// pre-derived signal like `Cz-Pz` — must be loaded whole, not split into its
/// arithmetic leaves (`Cz` − `Pz`), so the reader's literal fast-path in
/// `select_channel` can find it. `collect_leaf_labels` is label-agnostic and
/// splits `Cz-Pz` into `Cz`,`Pz`. With those leaves alone,
/// `read_edf_with_signals` would never load the literal signal and the channel
/// would be (wrongly) reported "not found".
///
/// So the list is a safe superset: it also requests the raw channel/reference
/// strings.
/// `read_edf_with_signals` matches labels case-insensitively and only loads
/// ones that actually exist, so a literal channel loads whole while a genuine
/// expression's raw string (e.g. `mean(C3, C4)`) simply matches nothing.
pub fn leaf_labels_for_read(chan_exprs: &[&str], ref_strs: &[&str]) -> Result<Vec<String>, String> {
    let mut out = crate::io::expr::collect_leaf_labels(chan_exprs, ref_strs)
        .map_err(|e| e.to_string())?;
    // Append a label unless it is blank or already listed (case-insensitive).
    fn push_uniq(out: &mut Vec<String>, label: &str) {
        let trimmed = label.trim();
        let already_listed = out.iter().any(|x| x.eq_ignore_ascii_case(trimmed));
        if !trimmed.is_empty() && !already_listed {
            out.push(trimmed.to_string());
        }
    }
    for &chan_expr in chan_exprs {
        push_uniq(&mut out, chan_expr);
    }
    for &reference in ref_strs {
        // A reference is `NAME = expr`. Only the expression side can be a label.
        let body = match reference.find('=') {
            Some(eq_pos) => &reference[eq_pos + 1..],
            None => reference,
        };
        push_uniq(&mut out, body);
    }
    Ok(out)
}

#[cfg(test)]
mod leaf_labels_for_read_tests {
    use super::leaf_labels_for_read;

    #[test]
    fn literal_dashed_label_is_requested_whole() {
        // A pre-derived bipolar channel like "Cz-Pz" must appear verbatim in the
        // read set (so the reader's literal fast-path loads it), NOT only as its
        // arithmetic leaves Cz / Pz.
        let out = leaf_labels_for_read(&["Cz-Pz"], &[]).unwrap();
        assert!(out.iter().any(|s| s.eq_ignore_ascii_case("Cz-Pz")),
            "literal 'Cz-Pz' missing from read set: {:?}", out);
    }

    #[test]
    fn genuine_expression_still_collects_leaves() {
        // A real derivation keeps its leaves; the raw string is added too but is
        // harmless (won't match a real signal).
        let out = leaf_labels_for_read(&["C3-A2"], &[]).unwrap();
        assert!(out.iter().any(|s| s.eq_ignore_ascii_case("C3")), "{:?}", out);
        assert!(out.iter().any(|s| s.eq_ignore_ascii_case("A2")), "{:?}", out);
    }

    #[test]
    fn reference_body_literal_is_requested() {
        // A reference whose body is a literal label (e.g. "REF = Cz-Pz") also
        // gets the literal into the read set.
        let out = leaf_labels_for_read(&["REF"], &["REF = Cz-Pz"]).unwrap();
        assert!(out.iter().any(|s| s.eq_ignore_ascii_case("Cz-Pz")), "{:?}", out);
    }
}

/// Select one channel's `(data, fs)` from already-loaded EDF data — cheap,
/// purely in-memory. The channel string can be a literal label (with or without
/// brackets / hyphens), an expression like `mean([C3], [C4])`, or a derivation
/// referencing named entries in `references`; `select_channel_with_refs` covers
/// all of those (including the bare-label fast-path).
pub fn select_channel(
    edf: &crate::io::edf::EdfData,
    channel: &str,
    references: &[String],
) -> Result<(Vec<f64>, f64), LoadError> {
    let (sig_header, data) = crate::io::edf::select_channel_with_refs(edf, channel, references)
        .map_err(|e| LoadError(format!("select_channel: {}", e)))?;
    Ok((data, sig_header.sampling_frequency))
}

/// Decode a staging-delimiter string to the single separator character
/// the parser uses. THE alias table — settings files and UIs store
/// human spellings ("tab", "\\t", "comma", "space"); every consumer
/// must decode them identically or a `"tab"` from a saved
/// batch_settings.json silently splits fields on the letter `t`.
/// Unrecognized strings use their first character; empty defaults to
/// `,`.
pub fn normalize_delim(d: &str) -> char {
    // No trim: a literal tab or space IS the delimiter.
    match d {
        "tab" | "\\t" | "\t" => '\t',
        "comma" => ',',
        "space" => ' ',
        other => other.chars().next().unwrap_or(','),
    }
}

#[cfg(test)]
mod delim_tests {
    use super::normalize_delim;

    #[test]
    fn aliases_decode_and_literals_pass_through() {
        assert_eq!(normalize_delim("tab"), '\t');
        assert_eq!(normalize_delim("\\t"), '\t');
        assert_eq!(normalize_delim("\t"), '\t');
        assert_eq!(normalize_delim("comma"), ',');
        assert_eq!(normalize_delim("space"), ' ');
        assert_eq!(normalize_delim(","), ',');
        assert_eq!(normalize_delim(";"), ';');
        assert_eq!(normalize_delim(""), ',');
    }
}

/// Length of the recording in seconds from the EDF header (number of data
/// records times the record length), or `None` when the header does not
/// state the record count.
pub fn recording_duration_s(header: &crate::io::edf::EdfHeader) -> Option<f64> {
    if header.num_data_records <= 0 || header.data_record_duration <= 0.0 {
        return None;
    }
    Some(header.num_data_records as f64 * header.data_record_duration)
}

/// Read a staging file once for a subject (channel-independent) and align
/// its clock times to the EDF recording start. The EDF header supplies the
/// start time and the recording length, which the staging reader uses to
/// refuse staging that begins before the recording does.
///
/// MATLAB `load_data` passes `header.recording_starttime` to `read_staging`
/// the same way, so staging times become absolute from EDF sample 0. Without
/// the start time the staging is shifted by the gap between recording start
/// and the first scored epoch, and every downstream "in scored range?"
/// filter bites the wrong window.
#[allow(clippy::too_many_arguments)]
pub fn load_staging_file(
    edf_header: &crate::io::edf::EdfHeader,
    staging_path: &std::path::Path,
    time_col: usize,
    stage_col: usize,
    header_lines: usize,
    delimiter: &str,
    epoch_dur: f64,
    stage_labels: &[Vec<String>],
) -> Result<(Vec<f64>, Vec<f64>), LoadError> {
    let opts = crate::io::staging::StagingOpts {
        time_col,
        stage_col,
        header_lines,
        delimiter: normalize_delim(delimiter),
        epoch_dur,
        start_time: Some(edf_header.recording_starttime.trim().to_string()),
        recording_duration: recording_duration_s(edf_header),
        stage_vals: stage_labels.to_vec(),
    };
    let staging = crate::io::staging::read_staging(staging_path, &opts)
        .map_err(|e| LoadError(format!("staging: {}", e)))?;
    Ok((staging.times, staging.vals))
}

/// No-staging mode: synthesize an all-N2 hypnogram (stage code 2) — one epoch
/// every `epoch_dur` seconds across the whole recording, so every sample scores
/// as N2.
pub fn synth_n2_staging(n_samples: usize, fs: f64, epoch_dur: f64) -> (Vec<f64>, Vec<f64>) {
    let duration = n_samples as f64 / fs;
    let n = ((duration / epoch_dur).ceil() as usize).max(1);
    let times: Vec<f64> = (0..n).map(|i| i as f64 * epoch_dur).collect();
    (times, vec![2.0_f64; n])
}
