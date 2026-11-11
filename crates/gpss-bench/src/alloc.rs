//! A counting global allocator for peak-heap measurements.
//!
//! Wraps [`std::alloc::System`] and tracks live and peak heap bytes with atomics. It is
//! *not* installed by the library: a binary or test opts in with
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: gpss_bench::alloc::CountingAlloc = gpss_bench::alloc::CountingAlloc;
//! ```
//!
//! and then calls [`measure`]. When it is not installed, [`is_active`] is `false` and
//! memory figures are reported as unavailable instead of as zero.
//!
//! Heap bytes are what the program asked for, which is machine-independent for a given
//! build; they exclude allocator overhead, stack and code, which process-level peak RSS
//! (`conformance/harness.py`) includes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// The counting allocator. Zero-sized; all state is global.
pub struct CountingAlloc;

#[inline]
fn add(n: usize) {
    let now = CURRENT.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged; the
// counters are side bookkeeping and never affect the returned pointers.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            CURRENT.fetch_sub(layout.size(), Relaxed);
            add(new_size);
        }
        p
    }
}

/// Mark the allocator as installed. [`measure`] calls this implicitly by probing.
fn probe() -> bool {
    if ACTIVE.load(Relaxed) {
        return true;
    }
    let before = CURRENT.load(Relaxed);
    let v = std::hint::black_box(vec![0u8; 64]);
    let installed = CURRENT.load(Relaxed) >= before + 64;
    drop(v);
    if installed {
        ACTIVE.store(true, Relaxed);
    }
    installed
}

/// Whether [`CountingAlloc`] is the global allocator of this process.
pub fn is_active() -> bool {
    probe()
}

/// Live heap bytes right now (0 if the allocator is not installed).
pub fn current_bytes() -> usize {
    CURRENT.load(Relaxed)
}

/// Heap usage of one closure call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct HeapUsage {
    /// Highest live heap during the call, above the level at entry.
    pub peak_bytes: usize,
    /// Heap still live after the call (including the returned value), above entry level.
    /// Negative if the call freed more than it allocated.
    pub retained_bytes: i64,
}

/// Run `f` and report its heap usage, or `None` for the usage if the counting allocator is
/// not installed. Single-threaded measurements only: allocations by other threads during
/// the call are counted too.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Option<HeapUsage>) {
    if !probe() {
        return (f(), None);
    }
    let before = CURRENT.load(Relaxed);
    PEAK.store(before, Relaxed);
    let out = f();
    let peak = PEAK.load(Relaxed);
    let after = CURRENT.load(Relaxed);
    let usage = HeapUsage {
        peak_bytes: peak.saturating_sub(before),
        retained_bytes: after as i64 - before as i64,
    };
    (out, Some(usage))
}
