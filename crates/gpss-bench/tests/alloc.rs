//! The counting allocator measures what a closure allocates.

use gpss_bench::alloc::{self, CountingAlloc};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[test]
fn measures_peak_and_retained_heap() {
    assert!(alloc::is_active());
    let ((), usage) = alloc::measure(|| {
        let v = std::hint::black_box(vec![1u8; 1 << 20]);
        drop(v);
    });
    let usage = usage.unwrap();
    assert!(usage.peak_bytes >= 1 << 20, "{usage:?}");
    assert!(usage.retained_bytes.abs() < 4096, "{usage:?}");

    let (kept, usage) = alloc::measure(|| vec![0u64; 1000]);
    assert!(usage.unwrap().retained_bytes >= 8000);
    drop(kept);
}
