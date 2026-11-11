//! Records the compiler version and build profile for benchmark reports.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!("cargo:rustc-env=GPSS_BENCH_RUSTC={}", version.trim());
    let profile = std::env::var("PROFILE").unwrap_or_default();
    println!("cargo:rustc-env=GPSS_BENCH_PROFILE={profile}");
    println!("cargo:rerun-if-changed=build.rs");
}
