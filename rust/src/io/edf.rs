//! EDF / EDF+ reader ported from `read_EDF_mex.c`.
//!
//! The format is little-endian with a fixed 256-byte main header followed
//! by 256 bytes of per-signal header per signal, and then int16 data records.
//! Digital samples are linearly mapped to physical units:
//!
//! ```text
//! scale  = (physical_max - physical_min) / (digital_max - digital_min)
//! offset = physical_min - digital_min * scale
//! phys   = raw_i16 * scale + offset
//! ```
//!
//! Channel selection matches labels case-insensitively after trimming
//! surrounding spaces and NUL bytes. "ChA-ChB" requests a rereferenced
//! (subtraction) channel.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Main EDF header.
#[derive(Debug, Clone, Default)]
pub struct EdfHeader {
    pub edf_ver: String,
    pub patient_id: String,
    pub local_rec_id: String,
    pub recording_startdate: String,
    pub recording_starttime: String,
    pub num_header_bytes: i32,
    pub num_data_records: i32,
    pub data_record_duration: f64,
    pub num_signals: i32,
}

/// Per-signal header (after decoding + digital→physical scaling params).
#[derive(Debug, Clone, Default)]
pub struct SignalHeader {
    pub signal_labels: String,
    pub transducer_type: String,
    pub physical_dimension: String,
    pub physical_min: f64,
    pub physical_max: f64,
    pub digital_min: f64,
    pub digital_max: f64,
    pub prefiltering: String,
    pub samples_in_record: i32,
    pub sampling_frequency: f64,
}

/// All-signals decode result.
pub struct EdfData {
    pub header: EdfHeader,
    pub signals: Vec<SignalHeader>,
    /// One f64 vector per output channel (length = samples_in_record * num_data_records).
    pub data: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub enum EdfError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for EdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdfError::Io(e) => write!(f, "EDF I/O error: {}", e),
            EdfError::Format(s) => write!(f, "EDF format error: {}", s),
        }
    }
}

impl std::error::Error for EdfError {}

impl From<std::io::Error> for EdfError {
    fn from(e: std::io::Error) -> Self {
        EdfError::Io(e)
    }
}

/// Trim leading/trailing spaces and NULs (matches MATLAB `strtrim` + mex trim_string).
fn trim_edf(s: &[u8]) -> String {
    let mut start = 0usize;
    let mut end = s.len();
    while start < end && (s[start] == b' ' || s[start] == 0) {
        start += 1;
    }
    while end > start && (s[end - 1] == b' ' || s[end - 1] == 0) {
        end -= 1;
    }
    String::from_utf8_lossy(&s[start..end]).into_owned()
}

fn parse_int(s: &str) -> i32 {
    // MATLAB `atoi` equivalent: parse leading int, silently accept trailing.
    let t = s.trim();
    // Try full, fall back to leading-digit parse.
    if let Ok(v) = t.parse::<i32>() {
        return v;
    }
    let mut acc = String::new();
    let mut it = t.chars();
    if let Some(c) = it.next() {
        if c == '-' || c == '+' || c.is_ascii_digit() {
            acc.push(c);
        } else {
            return 0;
        }
    }
    for c in it {
        if c.is_ascii_digit() {
            acc.push(c);
        } else {
            break;
        }
    }
    acc.parse::<i32>().unwrap_or(0)
}

fn parse_float(s: &str) -> f64 {
    let t = s.trim();
    t.parse::<f64>().unwrap_or_else(|_| {
        // Try leading numeric prefix.
        let mut acc = String::new();
        for c in t.chars() {
            if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
                acc.push(c);
            } else {
                break;
            }
        }
        acc.parse::<f64>().unwrap_or(0.0)
    })
}

/// Read the main 256-byte EDF header.
pub fn read_main_header(f: &mut File) -> Result<EdfHeader, EdfError> {
    let mut buf = [0u8; 256];
    f.read_exact(&mut buf)?;
    let mut h = EdfHeader::default();
    h.edf_ver = trim_edf(&buf[0..8]);
    h.patient_id = trim_edf(&buf[8..88]);
    h.local_rec_id = trim_edf(&buf[88..168]);
    // Date/time: do NOT trim — keep 8-char fixed field (mex keeps raw).
    h.recording_startdate = String::from_utf8_lossy(&buf[168..176]).into_owned();
    h.recording_starttime = String::from_utf8_lossy(&buf[176..184]).into_owned();
    h.num_header_bytes = parse_int(&String::from_utf8_lossy(&buf[184..192]));
    h.num_data_records = parse_int(&String::from_utf8_lossy(&buf[236..244]));
    h.data_record_duration = parse_float(&String::from_utf8_lossy(&buf[244..252]));
    h.num_signals = parse_int(&String::from_utf8_lossy(&buf[252..256]));
    Ok(h)
}

/// Read per-signal headers (laid out field-major: all labels, then all
/// transducer_types, etc.). File position must be at byte 256.
pub fn read_signal_headers(f: &mut File, nsig: usize) -> Result<Vec<SignalHeader>, EdfError> {
    let mut sh = vec![SignalHeader::default(); nsig];

    // 16-char labels
    let mut buf16 = vec![0u8; 16];
    for s in sh.iter_mut() {
        f.read_exact(&mut buf16)?;
        s.signal_labels = trim_edf(&buf16);
    }
    // 80-char transducer
    let mut buf80 = vec![0u8; 80];
    for s in sh.iter_mut() {
        f.read_exact(&mut buf80)?;
        s.transducer_type = trim_edf(&buf80);
    }
    // 8-char physical_dimension
    let mut buf8 = vec![0u8; 8];
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.physical_dimension = trim_edf(&buf8);
    }
    // 8-char physical_min
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.physical_min = parse_float(&String::from_utf8_lossy(&buf8));
    }
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.physical_max = parse_float(&String::from_utf8_lossy(&buf8));
    }
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.digital_min = parse_float(&String::from_utf8_lossy(&buf8));
    }
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.digital_max = parse_float(&String::from_utf8_lossy(&buf8));
    }
    // 80-char prefiltering
    for s in sh.iter_mut() {
        f.read_exact(&mut buf80)?;
        s.prefiltering = trim_edf(&buf80);
    }
    // 8-char samples_in_record
    for s in sh.iter_mut() {
        f.read_exact(&mut buf8)?;
        s.samples_in_record = parse_int(&String::from_utf8_lossy(&buf8));
    }
    // 32-char reserved
    let mut buf32 = vec![0u8; 32];
    for _ in 0..nsig {
        f.read_exact(&mut buf32)?;
    }

    Ok(sh)
}

/// Read just the EDF headers (main + per-signal) and apply the same
/// post-processing `read_edf_all` performs — without reading sample
/// data. Specifically:
///   1. Compute `sampling_frequency = samples_in_record /
///      data_record_duration` for each signal.
///   2. If `num_data_records == -1` (EDF convention for "unknown"),
///      derive it from `file_size − num_header_bytes` divided by the
///      bytes-per-record (`2 · Σ samples_in_record`).
///
/// This is what `read_EDF.m` does in its header-only branch — UI
/// previews need both quantities to display duration + fs.
pub fn read_edf_header<P: AsRef<Path>>(path: P) -> Result<(EdfHeader, Vec<SignalHeader>), EdfError> {
    let mut f = File::open(&path)?;
    let mut header = read_main_header(&mut f)?;
    if header.num_signals <= 0 {
        return Err(EdfError::Format(format!(
            "invalid num_signals: {}",
            header.num_signals
        )));
    }
    let mut sh = read_signal_headers(&mut f, header.num_signals as usize)?;

    if header.data_record_duration > 0.0 {
        for s in sh.iter_mut() {
            s.sampling_frequency = s.samples_in_record as f64 / header.data_record_duration;
        }
    }

    if header.num_data_records <= 0 {
        let total_samp_per_rec: u64 = sh.iter().map(|s| s.samples_in_record.max(0) as u64).sum();
        let bytes_per_rec = total_samp_per_rec * 2;
        if bytes_per_rec > 0 {
            let fsize = f.seek(SeekFrom::End(0))?;
            let data_bytes = fsize.saturating_sub(header.num_header_bytes as u64);
            let actual_records = (data_bytes / bytes_per_rec) as i32;
            if actual_records >= 1 {
                header.num_data_records = actual_records;
            }
        }
    }

    Ok((header, sh))
}

/// Read the whole EDF, decoding every signal. Returns (header, signals, data).
pub fn read_edf_all<P: AsRef<Path>>(path: P) -> Result<EdfData, EdfError> {
    let mut f = File::open(&path)?;
    let mut header = read_main_header(&mut f)?;
    if header.num_signals <= 0 {
        return Err(EdfError::Format(format!(
            "invalid num_signals: {}",
            header.num_signals
        )));
    }
    let nsig = header.num_signals as usize;
    let mut sh = read_signal_headers(&mut f, nsig)?;

    let total_samp_per_rec: u64 = sh.iter().map(|s| s.samples_in_record as u64).sum();
    if total_samp_per_rec == 0 {
        return Err(EdfError::Format("total samples per record is zero".into()));
    }
    let bytes_per_rec = total_samp_per_rec * 2;

    // Sampling frequency (needs data_record_duration).
    if header.data_record_duration <= 0.0 {
        return Err(EdfError::Format(format!(
            "bad data_record_duration {}",
            header.data_record_duration
        )));
    }
    for s in sh.iter_mut() {
        s.sampling_frequency = s.samples_in_record as f64 / header.data_record_duration;
    }

    // File-size derived num_data_records (mex fix_num_records behavior).
    let fsize = f.seek(SeekFrom::End(0))?;
    let data_bytes = fsize.saturating_sub(header.num_header_bytes as u64);
    let actual_records = (data_bytes / bytes_per_rec) as i32;
    if actual_records < 1 {
        return Err(EdfError::Format(
            "file does not contain complete data records".into(),
        ));
    }
    if header.num_data_records <= 0 || header.num_data_records != actual_records {
        header.num_data_records = actual_records;
    }

    // Read all data bytes.
    f.seek(SeekFrom::Start(header.num_header_bytes as u64))?;
    let mut raw = vec![0u8; (bytes_per_rec as usize) * (header.num_data_records as usize)];
    f.read_exact(&mut raw)?;

    // Per-signal sample offset within one record.
    let mut sig_offset = vec![0u64; nsig];
    let mut acc: u64 = 0;
    for (i, s) in sh.iter().enumerate() {
        sig_offset[i] = acc;
        acc += s.samples_in_record as u64;
    }

    let ndr = header.num_data_records as u64;
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(nsig);
    for s in 0..nsig {
        let spr = sh[s].samples_in_record as u64;
        let len = (spr * ndr) as usize;
        let mut out = Vec::with_capacity(len);

        let d_range = sh[s].digital_max - sh[s].digital_min;
        let p_range = sh[s].physical_max - sh[s].physical_min;
        if d_range == 0.0 {
            return Err(EdfError::Format(format!(
                "digital_max == digital_min for signal '{}'",
                sh[s].signal_labels
            )));
        }
        let scale = p_range / d_range;
        let offs = sh[s].physical_min - sh[s].digital_min * scale;

        for r in 0..ndr {
            let base = r * total_samp_per_rec + sig_offset[s];
            for k in 0..spr {
                let bidx = 2 * ((base + k) as usize);
                let v = i16::from_le_bytes([raw[bidx], raw[bidx + 1]]);
                out.push(v as f64 * scale + offs);
            }
        }
        data.push(out);
    }

    Ok(EdfData { header, signals: sh, data })
}

/// Select a channel by label.
///
///   - Exact label match (case-insensitive, trimmed) → plain channel.
///   - Otherwise try each '-' as leftmost A-B split; both halves must match
///     a file label → returns A − B (element-wise).
///
/// Returns (selected_signal_header, signal_values).
/// On reref the header is a clone of A with `signal_labels` replaced by `label`.
pub fn select_channel(
    edf: &EdfData,
    label: &str,
) -> Result<(SignalHeader, Vec<f64>), EdfError> {
    let label = label.trim();

    // Direct match.
    if let Some(idx) = find_label(&edf.signals, label) {
        return Ok((edf.signals[idx].clone(), edf.data[idx].clone()));
    }

    // A-B reref.
    let bytes = label.as_bytes();
    for d in 1..bytes.len() {
        if bytes[d] != b'-' {
            continue;
        }
        let a = label[..d].trim();
        let b = label[d + 1..].trim();
        if a.is_empty() || b.is_empty() {
            continue;
        }
        if let (Some(ia), Some(ib)) = (find_label(&edf.signals, a), find_label(&edf.signals, b)) {
            let da = &edf.data[ia];
            let db = &edf.data[ib];
            if da.len() != db.len() {
                return Err(EdfError::Format(format!(
                    "reref channels '{}' and '{}' have different lengths ({} vs {})",
                    a,
                    b,
                    da.len(),
                    db.len()
                )));
            }
            let diff: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x - y).collect();
            let mut sh = edf.signals[ia].clone();
            sh.signal_labels = label.to_string();
            return Ok((sh, diff));
        }
    }

    Err(EdfError::Format(format!(
        "channel '{}' not found in EDF (available: {})",
        label,
        edf.signals
            .iter()
            .map(|s| s.signal_labels.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

fn find_label(sigs: &[SignalHeader], name: &str) -> Option<usize> {
    let name_lc = name.to_ascii_lowercase();
    sigs.iter()
        .position(|s| s.signal_labels.trim().to_ascii_lowercase() == name_lc)
}

/// Channel-selection with named-reference support. Ports MATLAB
/// `read_EDF.m`'s `References` + `Channels` two-pass model:
///
/// 1. Parse and topo-sort `references` (each one a `"NAME = expr"` line).
/// 2. Evaluate each reference against the EDF's raw channels (and any
///    earlier reference results), building a `signals` table.
/// 3. Parse `channel`, resolve its `SignalRef::Leaf` names against the
///    table, and return the resulting linear combination.
///
/// The returned `SignalHeader` is synthesised from the first leaf
/// channel's header (with `signal_labels` set to `channel`). All leaves
/// must share the same sample count — read_EDF's uniform-fs rule.
pub fn select_channel_with_refs(
    edf: &EdfData,
    channel: &str,
    references: &[String],
) -> Result<(SignalHeader, Vec<f64>), EdfError> {
    use super::expr::{evaluate, parse, parse_named, resolve_references, ExprAst, SignalRef, Term};

    // Channel fast-path: bare literal label match, even with refs
    // present. EDF labels can contain '-', '[' / ']', spaces — under
    // the new bracket-as-grouping parser semantics, those would mis-
    // parse if we always routed through the expression parser.
    if find_label(&edf.signals, channel.trim()).is_some() {
        return select_channel(edf, channel);
    }

    // Parse references; topo-sort. Each ref's body gets a literal-
    // label fast-path: if the textual body matches an EDF channel
    // name exactly, replace the parsed AST with a single-leaf AST so
    // the evaluator looks up the literal label directly.
    let mut parsed_refs: Vec<(String, ExprAst)> = Vec::new();
    for r in references {
        let (name, ast) = parse_named(r)
            .map_err(|e| EdfError::Format(format!("reference '{}': {}", r, e)))?;
        let body = match r.find('=') { Some(i) => r[i + 1..].trim(), None => r.trim() };
        let ast = if find_label(&edf.signals, body).is_some() {
            ExprAst { terms: vec![Term { coeff: 1.0, signal: Some(SignalRef::Leaf(body.to_string())) }] }
        } else {
            ast
        };
        parsed_refs.push((name, ast));
    }
    let ordered = resolve_references(&parsed_refs)
        .map_err(|e| EdfError::Format(format!("reference resolution: {}", e)))?;

    // Seed signals table with file channels.
    let mut signals: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for sh in &edf.signals {
        let key = sh.signal_labels.trim().to_string();
        let idx = find_label(&edf.signals, &key).unwrap();
        signals.insert(key, edf.data[idx].clone());
    }

    // Evaluate refs in dependency order.
    for (name, ast) in &ordered {
        let rewritten = rewrite_refs(ast, &signals_keys_only(&signals), &ordered);
        let v = evaluate(&rewritten, &signals)
            .map_err(|e| EdfError::Format(format!("evaluating ref '{}': {}", name, e)))?;
        signals.insert(name.clone(), v);
    }

    // Parse the channel expression.
    let channel = channel.trim();
    // Bare label fast-path again — covers `select_channel_with_refs(edf, "M", &[…M…])`.
    if let Some(v) = signals.get(channel) {
        // Synthesise header from any leaf. Use file channel if it's
        // a file label; otherwise fall back to the first file channel.
        let sh = match find_label(&edf.signals, channel) {
            Some(idx) => {
                let mut sh = edf.signals[idx].clone();
                sh.signal_labels = channel.to_string();
                sh
            }
            None => {
                let mut sh = edf.signals[0].clone();
                sh.signal_labels = channel.to_string();
                sh
            }
        };
        return Ok((sh, v.clone()));
    }

    let ast = parse(channel).map_err(|e| EdfError::Format(format!("channel '{}': {}", channel, e)))?;
    let rewritten = rewrite_refs(&ast, &signals_keys_only(&signals), &ordered);
    let v = evaluate(&rewritten, &signals)
        .map_err(|e| EdfError::Format(format!("evaluating channel '{}': {}", channel, e)))?;

    // Synthesise the header from the first leaf encountered.
    let header_leaf = ast.terms.iter().find_map(|t| t.signal.as_ref().map(|s| s.name().to_string()));
    let mut sh = match header_leaf.and_then(|n| find_label(&edf.signals, &n)) {
        Some(idx) => edf.signals[idx].clone(),
        None => edf.signals[0].clone(),
    };
    sh.signal_labels = channel.to_string();
    Ok((sh, v))
}

fn signals_keys_only(map: &std::collections::HashMap<String, Vec<f64>>) -> std::collections::HashSet<String> {
    map.keys().cloned().collect()
}

/// SignalRef::Leaf names that are also known refs become SignalRef::Named.
/// Kept simple — the evaluator looks up by name regardless of kind, but
/// preserving the distinction helps error messages.
fn rewrite_refs(
    ast: &super::expr::ExprAst,
    known_signals: &std::collections::HashSet<String>,
    ordered_refs: &[(String, super::expr::ExprAst)],
) -> super::expr::ExprAst {
    use super::expr::{ExprAst, SignalRef, Term};
    let ref_names: std::collections::HashSet<&str> =
        ordered_refs.iter().map(|(n, _)| n.as_str()).collect();
    let _ = known_signals; // currently unused; future: error early on missing
    let terms = ast.terms.iter().map(|t| {
        let new_sig = t.signal.as_ref().map(|s| {
            let n = s.name();
            if ref_names.contains(n) { SignalRef::Named(n.to_string()) } else { SignalRef::Leaf(n.to_string()) }
        });
        Term { coeff: t.coeff, signal: new_sig }
    }).collect();
    ExprAst { terms }
}
