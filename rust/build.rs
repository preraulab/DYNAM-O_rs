// No C build — watershed is implemented in pure Rust (matlab_watershed.rs).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
