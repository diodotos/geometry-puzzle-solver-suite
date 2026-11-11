//! Kociemba coordinates for the 3x3x3 [Koc].
//!
//! Each coordinate maps a sub-aspect of a [`Cube3State`] to a dense integer, and has a
//! matching `set_*` that builds some state with that coordinate (other aspects
//! arbitrary). Move tables are built by `set`, apply a move, `get`.

use super::Cube3State;

pub const N_TWIST: usize = 2187; // 3^7
pub const N_FLIP: usize = 2048; // 2^11
pub const N_SLICE: usize = 495; // C(12,4)
pub const N_CPERM: usize = 40320; // 8!
pub const N_UDEPERM: usize = 40320; // 8! (phase 2 only)
pub const N_SPERM: usize = 24; // 4!

/// Slice coordinate of the solved cube (UD-slice edges `FR FL BL BR` at positions 8..12).
pub const SLICE_SOLVED: u16 = 494;

const fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut r = 1;
    let mut i = 0;
    while i < k {
        r = r * (n - i) / (i + 1);
        i += 1;
    }
    r
}

/// Corner-orientation coordinate in `0..N_TWIST` (base 3 over the first seven corners).
pub fn twist(s: &Cube3State) -> u16 {
    s.co[..7].iter().fold(0u16, |a, &c| a * 3 + c as u16)
}

/// Set corner orientations from a twist coordinate; the eighth corner keeps the sum 0 mod 3.
pub fn set_twist(s: &mut Cube3State, mut t: u16) {
    let mut sum = 0;
    for i in (0..7).rev() {
        s.co[i] = (t % 3) as u8;
        sum += s.co[i];
        t /= 3;
    }
    s.co[7] = (3 - sum % 3) % 3;
}

/// Edge-orientation coordinate in `0..N_FLIP` (base 2 over the first eleven edges).
pub fn flip(s: &Cube3State) -> u16 {
    s.eo[..11].iter().fold(0u16, |a, &e| a * 2 + e as u16)
}

/// Set edge orientations from a flip coordinate; the twelfth edge keeps the sum even.
pub fn set_flip(s: &mut Cube3State, mut f: u16) {
    let mut sum = 0;
    for i in (0..11).rev() {
        s.eo[i] = (f % 2) as u8;
        sum += s.eo[i];
        f /= 2;
    }
    s.eo[11] = sum % 2;
}

/// Which 4 positions hold the UD-slice edges (combinatorial number system rank).
pub fn slice(s: &Cube3State) -> u16 {
    let mut idx = 0;
    let mut k = 0;
    for (pos, &e) in s.ep.iter().enumerate() {
        if e >= 8 {
            k += 1;
            idx += binom(pos, k);
        }
    }
    idx as u16
}

/// Place the UD-slice edges at the positions encoded by `idx`; other edges fill the rest in order.
pub fn set_slice(s: &mut Cube3State, idx: u16) {
    let mut idx = idx as usize;
    let mut is_slice = [false; 12];
    for k in (1..=4).rev() {
        let mut p = k - 1;
        while binom(p + 1, k) <= idx {
            p += 1;
        }
        idx -= binom(p, k);
        is_slice[p] = true;
    }
    let (mut a, mut b) = (0u8, 8u8);
    for (pos, &sl) in is_slice.iter().enumerate() {
        if sl {
            s.ep[pos] = b;
            b += 1;
        } else {
            s.ep[pos] = a;
            a += 1;
        }
    }
}

/// Lehmer rank of a permutation of `0..n` (identity = 0).
pub fn perm_rank(p: &[u8]) -> u32 {
    let n = p.len();
    let mut r = 0u32;
    for i in 0..n {
        let smaller = p[i + 1..].iter().filter(|&&x| x < p[i]).count() as u32;
        r = r * (n - i) as u32 + smaller;
    }
    r
}

/// Inverse of [`perm_rank`]: write the permutation of rank `r` into `out`.
pub fn perm_unrank(mut r: u32, out: &mut [u8]) {
    let n = out.len();
    let mut digits = vec![0u32; n];
    for i in (0..n).rev() {
        let base = (n - i) as u32;
        digits[i] = r % base;
        r /= base;
    }
    let mut avail: Vec<u8> = (0..n as u8).collect();
    for i in 0..n {
        out[i] = avail.remove(digits[i] as usize);
    }
}

/// Corner-permutation coordinate in `0..N_CPERM`.
pub fn cperm(s: &Cube3State) -> u16 {
    perm_rank(&s.cp) as u16
}

/// Set the corner permutation from its coordinate.
pub fn set_cperm(s: &mut Cube3State, r: u16) {
    perm_unrank(r as u32, &mut s.cp);
}

/// Permutation of the eight U/D-layer edges (valid inside phase 2's subgroup).
pub fn udeperm(s: &Cube3State) -> u16 {
    perm_rank(&s.ep[..8]) as u16
}

/// Set the U/D-layer edge permutation from its coordinate.
pub fn set_udeperm(s: &mut Cube3State, r: u16) {
    perm_unrank(r as u32, &mut s.ep[..8]);
}

/// Permutation of the four UD-slice edges among themselves (phase 2).
pub fn sperm(s: &Cube3State) -> u8 {
    let p: Vec<u8> = s.ep[8..].iter().map(|&e| e.wrapping_sub(8)).collect();
    perm_rank(&p) as u8
}

/// Set the slice-edge permutation from its coordinate.
pub fn set_sperm(s: &mut Cube3State, r: u8) {
    let mut p = [0u8; 4];
    perm_unrank(r as u32, &mut p);
    for (dst, &x) in s.ep[8..].iter_mut().zip(&p) {
        *dst = x + 8;
    }
}
