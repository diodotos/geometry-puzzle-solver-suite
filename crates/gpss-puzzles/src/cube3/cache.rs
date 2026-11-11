//! On-disk cache for generated 3x3x3 tables (Korf pattern databases, two-phase
//! pruning tables). Tables are build artefacts: they are generated on first use, saved
//! here, and never committed.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Directory for cached tables: `$GPSS_CACHE_DIR`, else `$XDG_CACHE_HOME/gpss`, else
/// `$HOME/.cache/gpss`. `GPSS_CACHE_DIR=""` disables caching.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("GPSS_CACHE_DIR") {
        return (!d.is_empty()).then(|| PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d).join("gpss"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache").join("gpss"))
}

/// Write `path` through a uniquely named temporary file and a rename, so concurrent
/// processes never see a half-written cache file.
pub(crate) fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut io::BufWriter<fs::File>) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp{}-{seq}", std::process::id()));
    let result = (|| {
        let mut f = io::BufWriter::new(fs::File::create(&tmp)?);
        write(&mut f)?;
        f.flush()?;
        drop(f);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

pub(crate) fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// 64-bit FNV-1a, enough to detect truncated or corrupted cache files.
pub(crate) struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    pub fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
}
