//! Small order statistics used by the timing reports.

/// Nearest-rank percentile (`q` in `0.0..=1.0`) of an ascending slice; `None` if empty.
pub fn percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    debug_assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
    let rank = (q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

/// Median (nearest-rank, lower middle for even lengths); `None` if empty.
pub fn median(sorted: &[f64]) -> Option<f64> {
    percentile(sorted, 0.5)
}

/// Sort a vector of finite floats ascending.
pub fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank() {
        let v = sorted(vec![5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(median(&v), Some(3.0));
        assert_eq!(percentile(&v, 0.0), Some(1.0));
        assert_eq!(percentile(&v, 0.95), Some(5.0));
        assert_eq!(percentile(&v, 1.0), Some(5.0));
        assert_eq!(median(&[1.0, 2.0]), Some(1.0));
        assert_eq!(median(&[]), None);
    }
}
