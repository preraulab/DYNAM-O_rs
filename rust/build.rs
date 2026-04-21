//! Build script: generate `include/dynamo_rs.h` from the `#[no_mangle]
//! extern "C"` surface declared in `src/c_api.rs`.
//!
//! This is best-effort: a cbindgen failure should NOT break the crate build,
//! because the pure-Rust staticlib/cdylib is still valid without the header.
//! We emit a `cargo:warning` in that case.

fn main() {
    println!("cargo:rerun-if-changed=src/c_api.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let crate_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(v) => v,
        Err(_) => return,
    };
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
