use super::coord::*;
use super::*;
use crate::cube::{invert_sequence, parse_sequence, random_scramble};
use gpss_core::search::{IdaStar, SearchLimits};
use gpss_core::ZeroHeuristic;
use rand::{rngs::StdRng, SeedableRng};

fn seq(s: &str) -> Vec<FaceTurn> {
    parse_sequence(s).unwrap()
}

#[test]
fn move_order_matches_move_index() {
    for (i, m) in FaceTurn::all().enumerate() {
        assert_eq!(move_index(m), i);
    }
}

#[test]
fn corners_agree_with_cube2() {
    for m in FaceTurn::all() {
        let c3 = transform(m);
        let c2 = crate::cube2::transform(m);
        assert_eq!(c3.cp, c2.cp, "{m}");
        assert_eq!(c3.co, c2.co, "{m}");
    }
}

#[test]
fn face_turns_have_order_four_and_preserve_invariants() {
    for m in FaceTurn::all() {
        let mut s = Cube3State::SOLVED;
        let order = if m.turn == crate::cube::Turn::Half {
            2
        } else {
            4
        };
        for k in 1..=order {
            s = s.then(transform(m));
            assert!(s.is_valid());
            assert_eq!(s == Cube3State::SOLVED, k == order, "{m} x{k}");
        }
    }
}

#[test]
fn known_orders() {
    let sexy = seq("R U R' U'");
    let mut s = Cube3State::SOLVED;
    for k in 1..=6 {
        s = s.then_all(&sexy);
        assert_eq!(s == Cube3State::SOLVED, k == 6);
    }
    // (R U) has order 105.
    let ru = seq("R U");
    let mut s = Cube3State::SOLVED;
    for k in 1..=105 {
        s = s.then_all(&ru);
        assert_eq!(s == Cube3State::SOLVED, k == 105, "k={k}");
    }
}

#[test]
fn superflip_flips_every_edge_in_place() {
    let s = Cube3State::SOLVED.then_all(&seq("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"));
    assert_eq!(s.cp, Cube3State::SOLVED.cp);
    assert_eq!(s.co, [0; 8]);
    assert_eq!(s.ep, Cube3State::SOLVED.ep);
    assert_eq!(s.eo, [1; 12]);
}

#[test]
fn out_of_range_orientations_are_invalid() {
    let mut s = Cube3State::SOLVED;
    s.co[0] = 3;
    assert!(!s.is_valid());
    let mut s = Cube3State::SOLVED;
    s.eo[0] = 2;
    assert!(!s.is_valid());
}

#[test]
fn scramble_then_inverse_is_identity() {
    let mut rng = StdRng::seed_from_u64(7);
    for len in [0, 1, 5, 40] {
        let m = random_scramble(&mut rng, len);
        assert_eq!(
            Cube3State::SOLVED
                .then_all(&m)
                .then_all(&invert_sequence(&m)),
            Cube3State::SOLVED
        );
    }
}

#[test]
fn coordinates_round_trip() {
    let mut s = Cube3State::SOLVED;
    for i in 0..N_TWIST as u16 {
        set_twist(&mut s, i);
        assert_eq!(twist(&s), i);
    }
    for i in 0..N_FLIP as u16 {
        set_flip(&mut s, i);
        assert_eq!(flip(&s), i);
    }
    for i in 0..N_SLICE as u16 {
        set_slice(&mut s, i);
        assert_eq!(slice(&s), i);
    }
    for i in (0..N_CPERM as u16).step_by(7) {
        set_cperm(&mut s, i);
        assert_eq!(cperm(&s), i);
    }
    for i in 0..N_SPERM as u8 {
        set_sperm(&mut s, i);
        assert_eq!(sperm(&s), i);
    }
    let solved = Cube3State::SOLVED;
    assert_eq!(
        (twist(&solved), flip(&solved), slice(&solved)),
        (0, 0, SLICE_SOLVED)
    );
    assert_eq!(
        (cperm(&solved), udeperm(&solved), sperm(&solved)),
        (0, 0, 0)
    );
}

#[test]
fn generic_idastar_solves_short_scrambles_optimally() {
    let cube = Cube3::new();
    let solver = IdaStar::new(ZeroHeuristic);
    for (s, d) in [("", 0), ("R", 1), ("R U", 2), ("R U F'", 3)] {
        let sol = cube
            .solve_scramble(&seq(s), &solver, &SearchLimits::default())
            .unwrap();
        assert_eq!(sol.moves.len(), d, "{s}");
        assert_eq!(
            cube.scrambled(&seq(s)).then_all(&sol.moves),
            Cube3State::SOLVED
        );
    }
}

#[test]
fn twophase_solves_random_scrambles_within_target() {
    let cube = Cube3::new();
    let solver = TwoPhase::default();
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..30 {
        let scramble = random_scramble(&mut rng, 30);
        let sol = cube
            .solve_scramble(&scramble, &solver, &SearchLimits::default())
            .unwrap();
        assert_eq!(
            cube.scrambled(&scramble).then_all(&sol.moves),
            Cube3State::SOLVED
        );
        assert!(
            sol.moves.len() <= 20,
            "{} moves {:?}",
            sol.moves.len(),
            sol.stats
        );
        assert!(!sol.optimal);
    }
}

#[test]
fn twophase_edge_cases() {
    let cube = Cube3::new();
    let solver = TwoPhase::default();
    let limits = SearchLimits::default();
    assert!(cube
        .solve_scramble(&[], &solver, &limits)
        .unwrap()
        .moves
        .is_empty());
    let sol = cube.solve_scramble(&seq("R2"), &solver, &limits).unwrap();
    assert_eq!(sol.moves.len(), 1);
    let superflip = seq("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2");
    let sol = cube.solve_scramble(&superflip, &solver, &limits).unwrap();
    assert_eq!(
        cube.scrambled(&superflip).then_all(&sol.moves),
        Cube3State::SOLVED
    );
    let one = SearchLimits {
        max_expanded: Some(1),
        ..SearchLimits::default()
    };
    let sol = cube.solve_scramble(&seq("R2"), &solver, &one).unwrap();
    assert_eq!(sol.moves.len(), 1);
    assert!(sol.stats.expanded <= 1);
    let zero = SearchLimits {
        max_expanded: Some(0),
        ..SearchLimits::default()
    };
    assert!(cube.solve_scramble(&seq("R2"), &solver, &zero).is_err());
    let strict = SearchLimits {
        max_depth: Some(2),
        ..SearchLimits::default()
    };
    assert!(cube
        .solve_scramble(&seq("R U F"), &solver, &strict)
        .is_err());
}

/// Builds (or loads from the cache) the full 87 MB Korf tables; run with
/// `cargo test --release -- --ignored korf`.
#[test]
#[ignore = "builds 87 MB of pattern databases; run in release mode"]
fn korf_tables_and_optimal_solves() {
    use super::korf::{self, N_CORNERS, N_EDGES6};
    let t = korf::tables();
    // Distance distribution of the corner group in the half-turn metric; the maximum of 11
    // matches Korf (1997).
    assert_eq!(
        t.corners.histogram(),
        vec![
            1, 18, 243, 2874, 28000, 205416, 1168516, 5402628, 20776176, 45391616, 15139616, 64736
        ]
    );
    for e in [&t.edges_a, &t.edges_b] {
        let h = e.histogram();
        assert_eq!(h.iter().sum::<u64>(), N_EDGES6 as u64);
        assert_eq!(h[0], 1);
    }
    // The two edge sets are not related by a symmetry that preserves the edge-orientation
    // convention, so their distributions differ; both were checked to sum to 12!/6! x 2^6.
    assert_eq!(
        t.edges_a.histogram(),
        vec![1, 18, 230, 2747, 30847, 308783, 2508618, 13189082, 23497569, 3039786, 239]
    );
    assert_eq!(t.corners.histogram().iter().sum::<u64>(), N_CORNERS as u64);

    let cube = Cube3::new();
    let solver = korf::solver();
    let mut rng = StdRng::seed_from_u64(97);
    for len in [1usize, 4, 8, 10, 11] {
        let scramble = random_scramble(&mut rng, len);
        let sol = cube
            .solve_scramble(&scramble, &solver, &SearchLimits::default())
            .unwrap();
        assert!(sol.moves.len() <= len);
        let s = cube.scrambled(&scramble).then_all(&sol.moves);
        assert_eq!(s, Cube3State::SOLVED);
        // Admissible and consistent with the found optimum.
        assert!(t.estimate(&cube.scrambled(&scramble)) as usize <= sol.moves.len());
    }
}

#[test]
fn encoding_is_injective_and_in_range() {
    use gpss_core::encode::Encodable;
    let cube = Cube3::new();
    let spec = cube.encoding();
    assert_eq!(
        (spec.puzzle.as_str(), spec.len, spec.classes),
        ("cube3", 20, 24)
    );
    let mut seen = std::collections::HashMap::new();
    let mut frontier = vec![cube.solved()];
    for _ in 0..3 {
        let mut next = Vec::new();
        for s in &frontier {
            for &m in cube.moves() {
                next.push(cube.apply(s, m));
            }
        }
        frontier = next;
        for s in &frontier {
            let mut code = [0u8; 20];
            cube.encode(s, &mut code);
            assert!(code.iter().all(|&c| u16::from(c) < spec.classes));
            assert_eq!(
                *seen.entry(code).or_insert(*s),
                *s,
                "two states share an encoding"
            );
        }
    }
    let mut solved = [0u8; 20];
    cube.encode(&cube.solved(), &mut solved);
    assert_eq!(&solved[..8], &[0, 3, 6, 9, 12, 15, 18, 21]);
    assert_eq!(&solved[8..], &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
}
