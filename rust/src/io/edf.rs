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
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

/// One of the two readers the EDF parsers consume — a raw file handle
/// for plain `.edf`, or an in-memory cursor over decompressed bytes for
/// `.edf.gz`. `EdfReader: Read + Seek` so the rest of the parser code
/// is identical in either branch.
enum EdfReader {
    File(File),
    Mem(Cursor<Vec<u8>>),
}

impl Read for EdfReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            EdfReader::File(f) => f.read(buf),
            EdfReader::Mem(c) => c.read(buf),
        }
    }
}

impl Seek for EdfReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            EdfReader::File(f) => f.seek(pos),
            EdfReader::Mem(c) => c.seek(pos),
        }
    }
}

/// Open an EDF file, transparently decompressing if the path ends in
/// `.gz` (case-insensitive). Mirrors the lab convention from
/// `preraulab_utilities/EDF_toolbox/read_EDF.m`: `.edf` files are read
/// straight from disk, `.edf.gz` files are decompressed (in memory,
/// matching the MEX path's semantics — the pure-MATLAB path uses
/// `gunzip → tempname` to a temp file). `.edf.zst` is supported by the
/// MATLAB toolbox but not by this reader yet; pre-decompress with
/// `zstd -d` if you need it.
fn path_is_gz(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
}

/// Read the main header + per-signal headers, applying the
/// data_record_duration → sampling_frequency derivation. Works on any
/// `R: Read`, so it serves both the seekable `EdfReader` path and the
/// streaming `BufReader<GzDecoder<File>>` path. Does NOT do the
/// num_data_records-from-file-size fallback — that needs Seek, so the
/// caller handles it on the seekable readers.
fn read_header_pair<R: Read>(r: &mut R) -> Result<(EdfHeader, Vec<SignalHeader>), EdfError> {
    let header = read_main_header(r)?;
    if header.num_signals <= 0 {
        return Err(EdfError::Format(format!(
            "invalid num_signals: {}",
            header.num_signals
        )));
    }
    let mut sh = read_signal_headers(r, header.num_signals as usize)?;
    if header.data_record_duration > 0.0 {
        for s in sh.iter_mut() {
            s.sampling_frequency = s.samples_in_record as f64 / header.data_record_duration;
        }
    }
    Ok((header, sh))
}

fn open_edf_reader(path: &Path) -> Result<EdfReader, EdfError> {
    let is_gz = path_is_gz(path);
    let f = File::open(path)?;
    if is_gz {
        let mut buf = Vec::new();
        flate2::read::GzDecoder::new(f)
            .read_to_end(&mut buf)
            .map_err(|e| EdfError::Format(format!("gzip decompress {}: {}", path.display(), e)))?;
        Ok(EdfReader::Mem(Cursor::new(buf)))
    } else {
        Ok(EdfReader::File(f))
    }
}

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

/// Read the main 256-byte EDF header. Sequential `read_exact` only —
/// no `Seek` required, so this can run against a streaming
/// `GzDecoder<File>` wrapped in `BufReader` and only pull the header
/// bytes off the gzipped stream (no full decompress needed for a
/// header-only call).
pub fn read_main_header<R: Read>(f: &mut R) -> Result<EdfHeader, EdfError> {
    let mut buf = [0u8; 256];
    f.read_exact(&mut buf)?;
    let mut h = EdfHeader::default();
    h.edf_ver = trim_edf(&buf[0..8]);
    h.patient_id = trim_edf(&buf[8..88]);
    h.local_rec_id = trim_edf(&buf[88..168]);
    // Date/time: do NOT trim — keep 8-char fixed field (mex keeps raw).
    // Use `trim_edf` (not raw `from_utf8_lossy`) so trailing NULL / space
    // padding gets stripped — some lab EDF writers leave the date/time
    // fields half-filled like `"22:07:2\0"`, which the EDF spec permits
    // (NULL is a conventional pad character) but downstream parsers
    // choke on. trim_edf is already used for patient_id / local_rec_id
    // via decode_str-style stripping; these two were the holdouts.
    h.recording_startdate = trim_edf(&buf[168..176]);
    h.recording_starttime = trim_edf(&buf[176..184]);
    h.num_header_bytes = parse_int(&String::from_utf8_lossy(&buf[184..192]));
    h.num_data_records = parse_int(&String::from_utf8_lossy(&buf[236..244]));
    h.data_record_duration = parse_float(&String::from_utf8_lossy(&buf[244..252]));
    h.num_signals = parse_int(&String::from_utf8_lossy(&buf[252..256]));
    Ok(h)
}

/// Read per-signal headers (laid out field-major: all labels, then all
/// transducer_types, etc.). File position must be at byte 256.
///
/// **Bulk-read implementation.** The signal-header block is exactly
/// `256 * nsig` bytes, laid out in 11 field-major sub-arrays. The
/// previous implementation issued ~11N sequential `read_exact` calls
/// against a raw `File`. On NFS each call is a network round-trip, so
/// for a 30-signal EDF this was ~330 round-trips per file — easily 10×
/// slower than MATLAB's `read_EDF` MEX, which reads through a buffered
/// libc `FILE*`. Reading the whole block in one syscall and parsing in
/// memory matches the MEX's effective behaviour.
pub fn read_signal_headers<R: Read>(f: &mut R, nsig: usize) -> Result<Vec<SignalHeader>, EdfError> {
    let block_size = 256 * nsig;
    let mut block = vec![0u8; block_size];
    f.read_exact(&mut block)?;

    let mut sh = vec![SignalHeader::default(); nsig];
    let mut off = 0;

    // 16-char labels. Some exports leave every label blank — give those
    // signals deterministic positional names (`ch1`, `ch2`, …) so the
    // rest of the stack (label pickers, channel expressions, run-time
    // selection) can address them: the same synthesis runs on every
    // read, so `ch3` always resolves to the third signal of that file.
    for (i, s) in sh.iter_mut().enumerate() {
        s.signal_labels = trim_edf(&block[off..off + 16]);
        if s.signal_labels.is_empty() {
            s.signal_labels = format!("ch{}", i + 1);
        }
        off += 16;
    }
    // 80-char transducer
    for s in sh.iter_mut() {
        s.transducer_type = trim_edf(&block[off..off + 80]);
        off += 80;
    }
    // 8-char physical_dimension
    for s in sh.iter_mut() {
        s.physical_dimension = trim_edf(&block[off..off + 8]);
        off += 8;
    }
    // 8-char physical_min
    for s in sh.iter_mut() {
        s.physical_min = parse_float(&String::from_utf8_lossy(&block[off..off + 8]));
        off += 8;
    }
    // 8-char physical_max
    for s in sh.iter_mut() {
        s.physical_max = parse_float(&String::from_utf8_lossy(&block[off..off + 8]));
        off += 8;
    }
    // 8-char digital_min
    for s in sh.iter_mut() {
        s.digital_min = parse_float(&String::from_utf8_lossy(&block[off..off + 8]));
        off += 8;
    }
    // 8-char digital_max
    for s in sh.iter_mut() {
        s.digital_max = parse_float(&String::from_utf8_lossy(&block[off..off + 8]));
        off += 8;
    }
    // 80-char prefiltering
    for s in sh.iter_mut() {
        s.prefiltering = trim_edf(&block[off..off + 80]);
        off += 80;
    }
    // 8-char samples_in_record
    for s in sh.iter_mut() {
        s.samples_in_record = parse_int(&String::from_utf8_lossy(&block[off..off + 8]));
        off += 8;
    }
    // 32-char reserved per signal — skipped.
    off += 32 * nsig;
    debug_assert_eq!(off, block_size);

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
    let path = path.as_ref();
    let is_gz = path_is_gz(path);

    // Streaming path for .gz: only inflate the header bytes
    // (~256 + 256N) instead of the whole night. On an overnight
    // .edf.gz this is the difference between reading 512 bytes off the
    // decompressor and inflating 100-500 MB just to peek at the
    // header. Falls through to the full-decompress path below only if
    // the rare num_data_records-fixup is needed.
    if is_gz {
        let f = File::open(path)?;
        let mut r = std::io::BufReader::new(flate2::read::GzDecoder::new(f));
        let (header, sh) = read_header_pair(&mut r)?;
        if header.num_data_records > 0 {
            return Ok((header, sh));
        }
        // fall through to the seekable path so the fixup can run
    }

    let mut f = open_edf_reader(path)?;
    let (mut header, sh) = read_header_pair(&mut f)?;

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
    let mut f = open_edf_reader(path.as_ref())?;
    let mut header = read_main_header(&mut f)?;
    if header.num_signals <= 0 {
        return Err(EdfError::Format(format!(
            "invalid num_signals: {}",
            header.num_signals
        )));
    }
    let nsig = header.num_signals as usize;
    let mut sh = read_signal_headers(&mut f, nsig)?;
    validate_decodable_headers(&sh)?;

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

/// Partial-load: read only the signals whose labels (case-insensitive,
/// trimmed) appear in `wanted_labels`. Labels not present in the EDF
/// header are silently absent from the result — the caller decides how
/// to report misses (the CLI's downstream `select_channel` call surfaces
/// them as `channel not found, skipping`, matching the per-channel-miss
/// behaviour of MATLAB `read_EDF`'s `'Channels'` arg).
///
/// Memory profile: same constant-size raw-record buffer
/// (`bytes_per_rec`) regardless of `wanted_labels.len()`, but `data` is
/// only allocated for wanted signals. So loading 4 of 8 channels halves
/// the decoded f64 storage. Disk bandwidth is unchanged (EDF interleaves
/// signals within each record; a record-by-record sequential read is
/// faster than per-signal seeks on any modern storage).
///
/// Otherwise identical to [`read_edf_all`]: same num-records fixup, same
/// i16 → f64 scaling, same `EdfData` shape with `signals` parallel to `data`.
pub fn read_edf_with_signals<P: AsRef<Path>>(
    path: P,
    wanted_labels: &[&str],
) -> Result<EdfData, EdfError> {
    let path = path.as_ref();
    let is_gz = path_is_gz(path);

    // Fast path for `.gz` with a header that already declares a positive
    // `num_data_records`: stream-decompress so we never hold the full
    // payload in memory. The inflate WORK is unchanged (gzip is
    // sequential and EDF interleaves channels per record, so every
    // byte still has to be inflated to read past the unwanted
    // signals in each record), but instead of a 100-500 MB Vec<u8>
    // held for the duration of the read, the working set is the
    // ~10 KB per-record scratch buffer. That matters when the
    // wizard's 4-loader parallel header scan + Phase-1 workers run
    // concurrently — RAM pressure drops by orders of magnitude.
    //
    // Falls through to the full-decompress + Seek path below when the
    // header's `num_data_records` isn't trustworthy — that branch
    // re-derives the count from `file_size / bytes_per_rec`, which
    // needs Seek.
    if is_gz {
        let f = File::open(path)?;
        let mut r = std::io::BufReader::new(flate2::read::GzDecoder::new(f));
        let (mut header, mut sh) = read_header_pair(&mut r)?;
        let bytes_per_rec = compute_bytes_per_rec(&sh)?;
        validate_record_duration(&header)?;
        // Sampling-frequency derivation already done by read_header_pair.
        if header.num_data_records > 0 {
            let wanted_idx = map_wanted_indices(&sh, wanted_labels);
            let sig_byte_offset = signal_byte_offsets(&sh);
            let data = decode_records(
                &mut r,
                header.num_data_records as u64,
                bytes_per_rec as usize,
                &wanted_idx,
                &sh,
                &sig_byte_offset,
            )?;
            let signals: Vec<SignalHeader> =
                wanted_idx.iter().map(|&i| sh[i].clone()).collect();
            let _ = (&mut header, &mut sh); // silence unused-mut on the streaming branch
            return Ok(EdfData { header, signals, data });
        }
        // Fall through to full-decompress so we can Seek-to-end for the
        // num_data_records fixup. Rare path — valid EDFs declare a
        // positive count in the header.
    }

    let mut f = open_edf_reader(path)?;
    let mut header = read_main_header(&mut f)?;
    if header.num_signals <= 0 {
        return Err(EdfError::Format(format!(
            "invalid num_signals: {}",
            header.num_signals
        )));
    }
    let nsig = header.num_signals as usize;
    let mut sh = read_signal_headers(&mut f, nsig)?;

    let bytes_per_rec = compute_bytes_per_rec(&sh)?;
    validate_record_duration(&header)?;
    for s in sh.iter_mut() {
        s.sampling_frequency = s.samples_in_record as f64 / header.data_record_duration;
    }

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

    let wanted_idx = map_wanted_indices(&sh, wanted_labels);
    let sig_byte_offset = signal_byte_offsets(&sh);

    f.seek(SeekFrom::Start(header.num_header_bytes as u64))?;
    let data = decode_records(
        &mut f,
        header.num_data_records as u64,
        bytes_per_rec as usize,
        &wanted_idx,
        &sh,
        &sig_byte_offset,
    )?;

    let signals: Vec<SignalHeader> = wanted_idx.iter().map(|&i| sh[i].clone()).collect();
    Ok(EdfData { header, signals, data })
}

/// Reject signal headers that can't be safely decoded. A negative
/// `samples_in_record` — from a truncated or bit-flipped header (common on
/// interrupted NFS transfers) — sign-extends to a huge value when cast to
/// `u64`/`usize`, producing oversized record sizes and byte offsets that index
/// out of bounds and panic mid-decode, taking down the whole batch. Catch it
/// here and fail this one file cleanly instead. Valid EDFs always declare a
/// non-negative samples-per-record.
fn validate_decodable_headers(sh: &[SignalHeader]) -> Result<(), EdfError> {
    for s in sh {
        if s.samples_in_record < 0 {
            return Err(EdfError::Format(format!(
                "signal '{}' declares a negative samples-per-record ({})",
                s.signal_labels, s.samples_in_record
            )));
        }
    }
    Ok(())
}

/// Total bytes per data record (sum of samples_in_record * 2 across
/// signals). Errors when zero — the EDF has no usable data.
fn compute_bytes_per_rec(sh: &[SignalHeader]) -> Result<u64, EdfError> {
    validate_decodable_headers(sh)?;
    let total: u64 = sh.iter().map(|s| s.samples_in_record as u64).sum();
    if total == 0 {
        return Err(EdfError::Format("total samples per record is zero".into()));
    }
    Ok(total * 2)
}

fn validate_record_duration(header: &EdfHeader) -> Result<(), EdfError> {
    if header.data_record_duration <= 0.0 {
        return Err(EdfError::Format(format!(
            "bad data_record_duration {}",
            header.data_record_duration
        )));
    }
    Ok(())
}

/// Map a `wanted_labels` list to indices into `sh`, case-insensitive +
/// trimmed. Order of `wanted_labels` preserved; duplicates dedup'd;
/// labels not present in the EDF silently absent (caller decides how
/// to report misses).
fn map_wanted_indices(sh: &[SignalHeader], wanted_labels: &[&str]) -> Vec<usize> {
    let header_keys: Vec<String> = sh
        .iter()
        .map(|s| s.signal_labels.trim().to_ascii_lowercase())
        .collect();
    let mut wanted_idx: Vec<usize> = Vec::new();
    for w in wanted_labels {
        let key = w.trim().to_ascii_lowercase();
        if let Some(i) = header_keys.iter().position(|l| l == &key) {
            if !wanted_idx.contains(&i) {
                wanted_idx.push(i);
            }
        }
    }
    wanted_idx
}

/// Per-signal byte offset within one data record (cumulative).
fn signal_byte_offsets(sh: &[SignalHeader]) -> Vec<u64> {
    let mut off = vec![0u64; sh.len()];
    let mut acc: u64 = 0;
    for (i, s) in sh.iter().enumerate() {
        off[i] = acc;
        acc += s.samples_in_record as u64 * 2;
    }
    off
}

/// Decode `ndr` records from `r` sequentially, materialising only the
/// signals in `wanted_idx`. `<R: Read>` so both the seekable path
/// (`File` / `Cursor`) and the streaming gz path
/// (`BufReader<GzDecoder<File>>`) share the same loop. The per-record
/// scratch buffer is the only large allocation; output vectors are
/// pre-sized to `samples_in_record * ndr` for each wanted signal.
fn decode_records<R: Read>(
    r: &mut R,
    ndr: u64,
    bytes_per_rec: usize,
    wanted_idx: &[usize],
    sh: &[SignalHeader],
    sig_byte_offset: &[u64],
) -> Result<Vec<Vec<f64>>, EdfError> {
    let mut data: Vec<Vec<f64>> = wanted_idx
        .iter()
        .map(|&i| Vec::with_capacity((sh[i].samples_in_record as usize) * (ndr as usize)))
        .collect();
    let mut rec_buf = vec![0u8; bytes_per_rec];
    for _r in 0..ndr {
        r.read_exact(&mut rec_buf)?;
        for (out_idx, &sig_idx) in wanted_idx.iter().enumerate() {
            let off = sig_byte_offset[sig_idx] as usize;
            let spr = sh[sig_idx].samples_in_record as usize;
            let d_range = sh[sig_idx].digital_max - sh[sig_idx].digital_min;
            let p_range = sh[sig_idx].physical_max - sh[sig_idx].physical_min;
            if d_range == 0.0 {
                return Err(EdfError::Format(format!(
                    "digital_max == digital_min for signal '{}'",
                    sh[sig_idx].signal_labels
                )));
            }
            let scale = p_range / d_range;
            let offs = sh[sig_idx].physical_min - sh[sig_idx].digital_min * scale;
            for k in 0..spr {
                let bidx = off + 2 * k;
                let v = i16::from_le_bytes([rec_buf[bidx], rec_buf[bidx + 1]]);
                data[out_idx].push(v as f64 * scale + offs);
            }
        }
    }
    Ok(data)
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

    // Evaluate refs in dependency order. A reference that cannot be
    // evaluated against THIS file (its channels are absent — common in
    // cohorts that mix montage naming schemes) is recorded rather than
    // fatal: only the channels that actually use it fail, with the
    // recorded cause attached, and a mean(...) over it just skips it.
    let mut ref_errors: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (name, ast) in &ordered {
        let rewritten = rewrite_refs(ast, &signals_keys_only(&signals), &ordered);
        match evaluate(&rewritten, &signals) {
            Ok(v) => { signals.insert(name.clone(), v); }
            Err(e) => { ref_errors.insert(name.clone(), e.to_string()); }
        }
    }
    let attach_ref_causes = |msg: String| -> String {
        let used: Vec<String> = ref_errors
            .iter()
            .filter(|(n, _)| msg.contains(n.as_str()))
            .map(|(n, e)| format!("{}: {}", n, e))
            .collect();
        if used.is_empty() { msg } else { format!("{} (reference not available — {})", msg, used.join("; ")) }
    };

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
        .map_err(|e| EdfError::Format(attach_ref_causes(format!("evaluating channel '{}': {}", channel, e))))?;

    // Synthesise the header from the first leaf encountered (signals()
    // recurses into mean arguments, so a channel led by a mean still
    // finds a real file channel).
    let header_leaf = ast.signals().into_iter().next();
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
    fn rewrite_terms(
        terms: &[Term],
        ref_names: &std::collections::HashSet<&str>,
    ) -> Vec<Term> {
        terms.iter().map(|t| {
            let new_sig = t.signal.as_ref().map(|s| match s {
                // Recurse into mean arguments so a ref used inside a
                // mean resolves through the signals table like any other.
                SignalRef::Mean(args) => SignalRef::Mean(
                    args.iter()
                        .map(|a| ExprAst { terms: rewrite_terms(&a.terms, ref_names) })
                        .collect(),
                ),
                _ => {
                    let n = s.name();
                    if ref_names.contains(n) { SignalRef::Named(n.to_string()) } else { SignalRef::Leaf(n.to_string()) }
                }
            });
            Term { coeff: t.coeff, signal: new_sig }
        }).collect()
    }
    ExprAst { terms: rewrite_terms(&ast.terms, &ref_names) }
}

#[cfg(test)]
mod gz_tests {
    //! Verify `read_edf_*` reads a gzipped EDF as if it were plain.
    //! Round-trip a synthesised minimal-valid EDF (1 signal, 1 record,
    //! 10 samples) under both `.edf` and `.edf.gz` suffixes and assert
    //! the parsed headers match.
    use super::*;
    use std::io::Write;

    /// Minimal valid EDF: 1 signal "EEG", 1 record of 10 i16 samples.
    /// 512-byte header (256 main + 256 per-signal) + 20 bytes of data
    /// = 532 bytes total. ASCII field encoding follows the EDF spec.
    fn synth_minimal_edf() -> Vec<u8> {
        let mut buf = Vec::new();
        // ---- main header (256 bytes) ----
        buf.extend_from_slice(b"0       ");          // version (8)
        buf.extend(std::iter::repeat(b' ').take(80)); // patient_id (80)
        buf.extend(std::iter::repeat(b' ').take(80)); // local_rec_id (80)
        buf.extend_from_slice(b"01.01.20");           // startdate (8)
        buf.extend_from_slice(b"12.00.00");           // starttime (8)
        buf.extend_from_slice(b"512     ");           // num_header_bytes (8)
        buf.extend(std::iter::repeat(b' ').take(44)); // reserved (44)
        buf.extend_from_slice(b"1       ");           // num_data_records (8)
        buf.extend_from_slice(b"1.0     ");           // data_record_duration (8)
        buf.extend_from_slice(b"1   ");               // num_signals (4)
        assert_eq!(buf.len(), 256);
        // ---- signal-header block (256 bytes for 1 signal) ----
        buf.extend_from_slice(b"EEG             ");   // label (16)
        buf.extend(std::iter::repeat(b' ').take(80)); // transducer (80)
        buf.extend_from_slice(b"uV      ");           // phys_dim (8)
        buf.extend_from_slice(b"-200    ");           // phys_min (8)
        buf.extend_from_slice(b"200     ");           // phys_max (8)
        buf.extend_from_slice(b"-32768  ");           // dig_min (8)
        buf.extend_from_slice(b"32767   ");           // dig_max (8)
        buf.extend(std::iter::repeat(b' ').take(80)); // prefilter (80)
        buf.extend_from_slice(b"10      ");           // samples_in_record (8)
        buf.extend(std::iter::repeat(b' ').take(32)); // reserved (32)
        assert_eq!(buf.len(), 512);
        // ---- data: 10 i16 zeros (20 bytes) ----
        buf.extend(std::iter::repeat(0u8).take(20));
        buf
    }

    fn unique_temp_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let pid = std::process::id();
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir();
        (
            dir.join(format!("dynamo-edf-test-{}-{}.edf", pid, ns)),
            dir.join(format!("dynamo-edf-test-{}-{}.edf.gz", pid, ns)),
        )
    }

    #[test]
    fn read_edf_header_treats_gz_as_plain() {
        let bytes = synth_minimal_edf();
        let (plain, gzipped) = unique_temp_paths();

        std::fs::write(&plain, &bytes).expect("write plain");
        {
            let f = std::fs::File::create(&gzipped).expect("create gz");
            let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            enc.write_all(&bytes).expect("gz write");
            enc.finish().expect("gz finish");
        }

        let (h_plain, sh_plain) = read_edf_header(&plain).expect("plain read");
        let (h_gz, sh_gz) = read_edf_header(&gzipped).expect("gz read");

        assert_eq!(h_plain.num_signals, 1);
        assert_eq!(h_gz.num_signals, 1);
        assert_eq!(h_plain.num_data_records, h_gz.num_data_records);
        assert_eq!(h_plain.data_record_duration, h_gz.data_record_duration);
        assert_eq!(h_plain.recording_startdate, h_gz.recording_startdate);
        assert_eq!(sh_plain.len(), sh_gz.len());
        assert_eq!(sh_plain[0].signal_labels, "EEG");
        assert_eq!(sh_gz[0].signal_labels, "EEG");

        let _ = std::fs::remove_file(&plain);
        let _ = std::fs::remove_file(&gzipped);
    }

    /// Header reads on .gz must NOT inflate the entire payload — they
    /// should stream just the header bytes off the GzDecoder and stop.
    /// Test it by gzipping a valid 512-byte header followed by ~2 MB
    /// of zero bytes (compresses fine; would take measurable time to
    /// fully inflate). The header read should succeed quickly without
    /// the heavy buffer ever being allocated.
    ///
    /// Also pins that the parsed header values match the in-memory
    /// truth, so we don't regress correctness while chasing speed.
    #[test]
    fn read_edf_header_streams_gz_without_full_inflate() {
        let mut bytes = synth_minimal_edf();
        // Pad with 2 MiB of zeros — would be visible in inflate time
        // if we fully decompressed. Streaming stops at byte 532.
        bytes.extend(std::iter::repeat(0u8).take(2 * 1024 * 1024));

        let pid = std::process::id();
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir()
            .join(format!("dynamo-edf-test-bigpad-{}-{}.edf.gz", pid, ns));
        {
            let f = std::fs::File::create(&path).expect("create");
            let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            enc.write_all(&bytes).expect("gz write");
            enc.finish().expect("gz finish");
        }

        let t0 = std::time::Instant::now();
        let (h, sh) = read_edf_header(&path).expect("streaming header read");
        let elapsed = t0.elapsed();

        assert_eq!(h.num_signals, 1);
        assert_eq!(sh[0].signal_labels, "EEG");
        // Header-only on a 2 MiB-padded gz should be near-instant if
        // streaming; the assertion is generous (250 ms) so a loaded CI
        // box doesn't false-flag, but a full inflate of 2 MiB of zeros
        // would normally be sub-ms anyway — the real win is on the
        // 100-500 MB overnight EDF case where the difference is many
        // seconds vs near-instant.
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "header read took {}ms — looks like the full payload was inflated",
            elapsed.as_millis()
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Multi-signal multi-record EDF round-tripped through both
    /// `.edf` (full-decompress path on plain file) and `.edf.gz`
    /// (streaming path). The decoded signal arrays must be
    /// byte-identical so the streaming optimization can't drift in
    /// the f64 values, sample count, or signal selection order.
    /// Specifically exercises the wanted_labels subset by asking
    /// for only signal #2 of 3.
    #[test]
    fn read_edf_with_signals_streaming_matches_plain() {
        // 3 signals (A, B, C) × 5 records × samples_in_record=4 →
        // 5 records of (3 × 4 × 2 bytes) = 120 bytes payload after
        // the 256 + 3*256 = 1024-byte header.
        let mut bytes = Vec::new();
        // ---- main header ----
        bytes.extend_from_slice(b"0       ");
        bytes.extend(std::iter::repeat(b' ').take(80));
        bytes.extend(std::iter::repeat(b' ').take(80));
        bytes.extend_from_slice(b"01.01.20");
        bytes.extend_from_slice(b"12.00.00");
        bytes.extend_from_slice(b"1024    ");           // num_header_bytes = 256 + 3*256
        bytes.extend(std::iter::repeat(b' ').take(44));
        bytes.extend_from_slice(b"5       ");           // num_data_records = 5
        bytes.extend_from_slice(b"1.0     ");
        bytes.extend_from_slice(b"3   ");               // num_signals = 3
        assert_eq!(bytes.len(), 256);
        // ---- signal headers: 3 of them, field-major layout ----
        // labels (16 bytes each × 3)
        bytes.extend_from_slice(b"A               ");
        bytes.extend_from_slice(b"B               ");
        bytes.extend_from_slice(b"C               ");
        // transducer (80 × 3)
        for _ in 0..3 { bytes.extend(std::iter::repeat(b' ').take(80)); }
        // phys_dim (8 × 3)
        for _ in 0..3 { bytes.extend_from_slice(b"uV      "); }
        // phys_min, phys_max, dig_min, dig_max — all the same per
        // signal for simplicity; gives scale=1, offs=0.
        for _ in 0..3 { bytes.extend_from_slice(b"-32768  "); } // phys_min
        for _ in 0..3 { bytes.extend_from_slice(b"32767   "); } // phys_max
        for _ in 0..3 { bytes.extend_from_slice(b"-32768  "); } // dig_min
        for _ in 0..3 { bytes.extend_from_slice(b"32767   "); } // dig_max
        // prefilter (80 × 3)
        for _ in 0..3 { bytes.extend(std::iter::repeat(b' ').take(80)); }
        // samples_in_record = 4 (8 × 3)
        for _ in 0..3 { bytes.extend_from_slice(b"4       "); }
        // reserved (32 × 3)
        for _ in 0..3 { bytes.extend(std::iter::repeat(b' ').take(32)); }
        assert_eq!(bytes.len(), 1024);
        // ---- data: 5 records, each [A(4 i16) | B(4 i16) | C(4 i16)] ----
        for r in 0..5i16 {
            // A: 100 + r * 10 + k
            for k in 0..4i16 { bytes.extend_from_slice(&(100 + r * 10 + k).to_le_bytes()); }
            // B: 200 + r * 10 + k
            for k in 0..4i16 { bytes.extend_from_slice(&(200 + r * 10 + k).to_le_bytes()); }
            // C: 300 + r * 10 + k
            for k in 0..4i16 { bytes.extend_from_slice(&(300 + r * 10 + k).to_le_bytes()); }
        }
        assert_eq!(bytes.len(), 1024 + 5 * 3 * 4 * 2);

        let (plain, gzipped) = unique_temp_paths();
        std::fs::write(&plain, &bytes).expect("plain write");
        {
            let f = std::fs::File::create(&gzipped).expect("create gz");
            let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            enc.write_all(&bytes).expect("gz write");
            enc.finish().expect("gz finish");
        }

        // Request only signal "B" — exercises wanted_labels filtering on both paths.
        let wanted = ["B"];
        let plain_out = read_edf_with_signals(&plain, &wanted).expect("plain read");
        let gz_out = read_edf_with_signals(&gzipped, &wanted).expect("gz streaming read");

        assert_eq!(plain_out.signals.len(), 1);
        assert_eq!(gz_out.signals.len(), 1);
        assert_eq!(plain_out.signals[0].signal_labels, "B");
        assert_eq!(gz_out.signals[0].signal_labels, "B");
        assert_eq!(plain_out.data.len(), 1);
        assert_eq!(gz_out.data.len(), 1);
        assert_eq!(plain_out.data[0], gz_out.data[0],
            "streaming gz must produce byte-identical data to plain");
        // Spot-check the values: signal B at record 0, sample 0 = 200.
        assert_eq!(plain_out.data[0][0], 200.0);
        // Record 4, sample 3 = 200 + 40 + 3 = 243.
        assert_eq!(plain_out.data[0][19], 243.0);

        let _ = std::fs::remove_file(&plain);
        let _ = std::fs::remove_file(&gzipped);
    }

    /// Mixed-case extension `.GZ` should also trigger decompression
    /// (read_EDF.m uses `endsWith(...,'IgnoreCase',true)`).
    #[test]
    fn read_edf_header_handles_uppercase_gz() {
        let bytes = synth_minimal_edf();
        let pid = std::process::id();
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir()
            .join(format!("dynamo-edf-test-upper-{}-{}.edf.GZ", pid, ns));
        {
            let f = std::fs::File::create(&path).expect("create");
            let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            enc.write_all(&bytes).expect("gz write");
            enc.finish().expect("gz finish");
        }
        let (h, _) = read_edf_header(&path).expect("uppercase .GZ read");
        assert_eq!(h.num_signals, 1);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod select_refs_tests {
    //! `select_channel_with_refs` over a synthetic in-memory EDF —
    //! exercises the heterogeneous-cohort case: a reference mean that
    //! lists every montage spelling, evaluated in a file that has only
    //! some of them.
    use super::*;

    fn edf_with(channels: &[(&str, Vec<f64>)]) -> EdfData {
        EdfData {
            header: EdfHeader::default(),
            signals: channels
                .iter()
                .map(|(name, _)| SignalHeader {
                    signal_labels: name.to_string(),
                    samples_in_record: 2,
                    sampling_frequency: 1.0,
                    ..SignalHeader::default()
                })
                .collect(),
            data: channels.iter().map(|(_, d)| d.clone()).collect(),
        }
    }

    #[test]
    fn ref_mean_uses_only_the_spellings_this_file_has() {
        // M1 lists two spellings of the same montage; this file carries
        // only the literal "C3-A2". M1 must evaluate as the mean of the
        // one available channel, and the rereferenced output must work.
        let edf = edf_with(&[("C3-A2", vec![10.0, 20.0]), ("O1-A2", vec![4.0, 6.0])]);
        let refs = vec!["M1 = mean($C3-A2$, $[C3-A2 - B]$, $O1-A2$, $[O1-A2 - B]$)".to_string()];
        let (_, v) = select_channel_with_refs(&edf, "$C3-A2$ - M1", &refs).unwrap();
        // M1 = (C3-A2 + O1-A2)/2 = [7, 13]; C3-A2 - M1 = [3, 7].
        assert_eq!(v, vec![3.0, 7.0]);
    }

    #[test]
    fn unavailable_ref_fails_only_channels_that_use_it() {
        // M2's channels are absent from this file entirely. A channel
        // using M1 still works; one using M2 fails with the cause named.
        let edf = edf_with(&[("C3-A2", vec![1.0, 2.0])]);
        let refs = vec![
            "M1 = mean($C3-A2$)".to_string(),
            "M2 = mean($C4-A1$, $[C4-A1 - B]$)".to_string(),
        ];
        let (_, ok) = select_channel_with_refs(&edf, "$C3-A2$ - M1", &refs).unwrap();
        assert_eq!(ok, vec![0.0, 0.0]);
        let err = select_channel_with_refs(&edf, "$C3-A2$ - M2", &refs).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("M2"), "{msg}");
        assert!(msg.contains("reference not available"), "{msg}");
    }
}

#[cfg(test)]
mod blank_label_tests {
    use super::*;

    #[test]
    fn blank_signal_labels_get_positional_names() {
        // One 256-byte signal-header block with an all-space label field:
        // the reader must synthesize "ch1" so the signal stays addressable.
        let mut block = vec![b' '; 256];
        // samples_in_record field (8 chars) sits at offset 16+80+8+8+8+8+8+80 = 216.
        block[216..224].copy_from_slice(b"200     ");
        let mut cur = std::io::Cursor::new(block);
        let sh = read_signal_headers(&mut cur, 1).unwrap();
        assert_eq!(sh[0].signal_labels, "ch1");
        assert_eq!(sh[0].samples_in_record, 200);
    }

    #[test]
    fn real_labels_are_untouched() {
        let mut block = vec![b' '; 256];
        block[0..3].copy_from_slice(b"CA ");
        let mut cur = std::io::Cursor::new(block);
        let sh = read_signal_headers(&mut cur, 1).unwrap();
        assert_eq!(sh[0].signal_labels, "CA");
    }
}
