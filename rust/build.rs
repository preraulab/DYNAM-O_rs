//! Build script: generate `include/dynamo_rs.h` from the `#[no_mangle]
//! extern "C"` surface declared in `src/c_api.rs`.
//!
//! This is best-effort: a cbindgen failure should NOT break the crate build,
//! because the Rust library build is still valid without the header.
//! We emit a `cargo:warning` in that case.

fn main() {
    println!("cargo:rerun-if-changed=src/c_api.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../data_matlab_filters");

    emit_git_sha();

    let crate_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(v) => v,
        Err(_) => return,
    };
    stage_filter_cache(std::path::Path::new(&crate_dir));

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var_os("CARGO_FEATURE_PYTHON").is_none()
    {
        println!("cargo:rustc-link-arg-cdylib=-Wl,-install_name,@rpath/libdynamo_rs.dylib");
    }

    // Embed the canonical SO-phase SOS filters into the binary so a relocated
    // build (no source tree, no DYNAMO_FILTER_CACHE) still loads the MATLAB
    // elliptic design instead of silently degrading to a cheby1 fallback.
    generate_embedded_filters(&crate_dir);

    // The C header is written into the source tree. A consumer that builds
    // this crate from a read-only or pinned checkout (a git submodule, a
    // container layer) sets DYNAMO_SKIP_C_HEADER=1 to leave the tree alone.
    println!("cargo:rerun-if-env-changed=DYNAMO_SKIP_C_HEADER");
    if std::env::var_os("DYNAMO_SKIP_C_HEADER").is_some() {
        return;
    }

    let include_dir = format!("{crate_dir}/include");
    let out = format!("{include_dir}/dynamo_rs.h");
    if std::fs::create_dir_all(&include_dir).is_err() {
        println!("cargo:warning=failed to create {include_dir}");
        return;
    }

    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        cpp_compat: true,
        no_includes: true,
        sys_includes: vec!["stddef.h".to_string(), "stdint.h".to_string()],
        header: Some(
            "/* Auto-generated C ABI header for dynamo_rs. Do not edit. */\n\
             /* Generated from src/c_api.rs by build.rs via cbindgen. */"
                .to_string(),
        ),
        include_guard: Some("DYNAMO_RS_H".to_string()),
        ..cbindgen::Config::default()
    };

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&out);
        }
        Err(e) => {
            println!("cargo:warning=cbindgen failed: {e}");
        }
    }
}

/// Embed the git commit as `DYNAMO_GIT_SHA` so `build_info::VERSION` ties
/// every kernel build to an exact source state (grammar:
/// `<semver>+<sha12>[.dirty]`, DesktopApp OUTPUT_FORMAT.md §8.1). Fail-soft
/// to `unknown` when git metadata is unavailable (source tarball builds).
fn emit_git_sha() {
    let sha = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let ver = if dirty { format!("{sha}.dirty") } else { sha };
    println!("cargo:rustc-env=DYNAMO_GIT_SHA={ver}");

    // Re-run when HEAD moves (commit / pull / checkout); logs/HEAD is appended
    // on every HEAD update, so it catches new commits on the same branch.
    for p in ["logs/HEAD", "HEAD"] {
        if let Some(path) = git(&["rev-parse", "--git-path", p]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

fn git(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn stage_filter_cache(crate_dir: &std::path::Path) {
    let source_dir = crate_dir.join("..").join("data_matlab_filters");
    let out_dir = std::env::var_os("OUT_DIR")
        .map(std::path::PathBuf::from)
        .expect("Cargo did not set OUT_DIR");

    let mut entries: Vec<_> = std::fs::read_dir(&source_dir)
        .unwrap_or_else(|e| {
            panic!(
                "failed to read filter cache at {}: {e}",
                source_dir.display()
            )
        })
        .map(|entry| {
            entry.unwrap_or_else(|e| {
                panic!(
                    "failed to read a filter-cache entry at {}: {e}",
                    source_dir.display()
                )
            })
        })
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "npy"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    assert_eq!(
        entries.len(),
        48,
        "expected 48 .npy filter-cache files in {}; found {}",
        source_dir.display(),
        entries.len()
    );

    for entry in std::fs::read_dir(&out_dir)
        .unwrap_or_else(|e| panic!("failed to inspect Cargo OUT_DIR: {e}"))
    {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read a Cargo OUT_DIR entry: {e}"))
            .path();
        if path.extension().is_some_and(|ext| ext == "npy") {
            std::fs::remove_file(&path)
                .unwrap_or_else(|e| panic!("failed to remove stale {}: {e}", path.display()));
        }
    }
    for entry in entries {
        let destination = out_dir.join(entry.file_name());
        std::fs::copy(entry.path(), &destination).unwrap_or_else(|e| {
            panic!(
                "failed to stage filter cache {}: {e}",
                destination.display()
            )
        });
    }
}

/// Write `$OUT_DIR/embedded_filters.rs` — a sorted `EMBEDDED_FILTERS` table of
/// `(filename, include_bytes!(abs_path))` for every `*.npy` in the sibling
/// `data_matlab_filters/` dir. `filter_cache.rs` `include!`s this and uses it as
/// the last on-disk-cache fallback before the cheby1 design.
fn generate_embedded_filters(crate_dir: &str) {
    let out_dir = match std::env::var("OUT_DIR") {
        Ok(v) => v,
        Err(_) => return,
    };
    let filters_dir = std::path::Path::new(crate_dir).join("..").join("data_matlab_filters");
    let mut entries: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&filters_dir) {
        let mut names: Vec<String> = rd
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".npy"))
            .collect();
        names.sort();
        for name in names {
            // Absolute path so include_bytes! resolves from OUT_DIR.
            let abs = filters_dir.join(&name);
            entries.push(format!(
                "    ({:?}, include_bytes!({:?})),",
                name,
                abs.display().to_string(),
            ));
        }
    } else {
        println!("cargo:warning=data_matlab_filters not found; embedded filter set will be empty");
    }
    let body = format!(
        "/// (filename, raw .npy bytes) for the canonical SO-phase SOS filters, \
         generated by build.rs.\npub static EMBEDDED_FILTERS: &[(&str, &[u8])] = &[\n{}\n];\n",
        entries.join("\n"),
    );
    let dest = std::path::Path::new(&out_dir).join("embedded_filters.rs");
    if let Err(e) = std::fs::write(&dest, body) {
        println!("cargo:warning=failed to write embedded_filters.rs: {e}");
    }
}
