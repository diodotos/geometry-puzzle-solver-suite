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

    // The symmetry/inverse lookups only ever raise the plain estimate, stay admissible,
    // and `estimate_within` agrees with `estimate` whenever the latter fits the budget.
    use gpss_core::Heuristic;
    let (plain, full) = (KorfHeuristic::plain(), KorfHeuristic::new());
    let st = super::sym::tables();
    for len in 0..=40 {
        let s = cube.scrambled(&random_scramble(&mut rng, len));
        let (hp, hf) = (plain.estimate(&s), full.estimate(&s));
        assert!(hp <= hf && hf as usize <= len);
        // Distance is invariant under conjugation and inversion, so every variant's
        // estimate is a lower bound for the original.
        for v in [s.inverse(), st.conjugate(&s, super::sym::urf3())] {
            assert!(plain.estimate(&v) as usize <= len);
        }
        for b in 0..=hf + 1 {
            let w = full.estimate_within(&s, b);
            assert!(w <= hf);
            if hf <= b {
                assert_eq!(w, hf);
            } else {
                assert!(w > b);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Symmetries (`cube3::sym`)

mod symmetry {
    use super::super::sym::{self, Facelets, N_SYM, N_SYM_D4H};
    use super::*;
    use proptest::prelude::*;
    use rand::Rng;

    fn random_state(seed: u64, len: usize) -> Cube3State {
        Cube3State::SOLVED.then_all(&random_scramble(&mut StdRng::seed_from_u64(seed), len))
    }

    /// The geometric model behind the symmetry tables agrees with the hand-written move
    /// tables: turning a layer of the sticker picture is the same as the cubie move.
    #[test]
    fn facelet_model_matches_move_tables() {
        for face in 0..6u8 {
            let solved = Facelets::from_state(&Cube3State::SOLVED);
            let mv = FaceTurn::all().nth(3 * face as usize).unwrap();
            assert_eq!(
                solved.turn(face),
                Facelets::from_state(transform(mv)),
                "{mv}"
            );
            let s = random_state(face as u64, 25);
            assert_eq!(
                Facelets::from_state(&s).turn(face),
                Facelets::from_state(&s.then(transform(mv))),
                "{mv} after a scramble"
            );
        }
    }

    #[test]
    fn symmetry_group_structure() {
        let t = sym::tables();
        // 48 distinct face maps, closed under composition, with inverses.
        for a in 0..N_SYM {
            for b in 0..N_SYM {
                assert_eq!(a == b, t.faces[a] == t.faces[b]);
            }
            assert_eq!(t.mult[a][t.inv[a] as usize], 0);
            assert_eq!(t.mult[0][a] as usize, a);
        }
        // Exactly 24 reflections; the odd-numbered ones.
        for s in 0..N_SYM {
            assert_eq!(t.is_reflection(s), s % 2 == 1, "{s}");
        }
        // 0..16 is the subgroup keeping the U-D axis (U goes to U or D).
        for s in 0..N_SYM {
            let ud = matches!(t.faces[s][0], 0 | 3);
            assert_eq!(ud, s < N_SYM_D4H, "{s}");
        }
        for a in 0..N_SYM_D4H {
            for b in 0..N_SYM_D4H {
                assert!((t.mult[a][b] as usize) < N_SYM_D4H);
            }
        }
    }

    /// Conjugation turns quarter turns into quarter turns (anticlockwise ones under a
    /// reflection) and half turns into half turns.
    #[test]
    fn conjugation_maps_moves_to_moves() {
        let t = sym::tables();
        let moves: Vec<FaceTurn> = FaceTurn::all().collect();
        for (m, mv) in moves.iter().enumerate() {
            for s in 0..N_SYM {
                let c = moves[t.move_conj[m][s] as usize];
                assert_eq!(sym::conjugate(transform(*mv), s), *transform(c));
                let expected_turn = if t.is_reflection(s) {
                    mv.inverse().turn
                } else {
                    mv.turn
                };
                assert_eq!(c.turn, expected_turn, "{mv} under {s}");
                assert_eq!(c.face as u8, t.faces[s][mv.face as usize]);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Conjugation is a left group action: `conj(conj(c, a), b) == conj(c, b . a)`,
        /// and the identity symmetry does nothing.
        #[test]
        fn conjugation_is_a_group_action(seed in any::<u64>(), len in 0usize..40,
                                         a in 0..N_SYM, b in 0..N_SYM) {
            let t = sym::tables();
            let c = random_state(seed, len);
            prop_assert_eq!(t.conjugate(&c, 0), c);
            prop_assert_eq!(
                t.conjugate(&t.conjugate(&c, a), b),
                t.conjugate(&c, t.mult[b][a] as usize)
            );
            prop_assert_eq!(t.conjugate(&t.conjugate(&c, a), t.inv[a] as usize), c);
        }

        /// Conjugation by a fixed symmetry is an automorphism: it respects composition
        /// and inverses, fixes the solved cube and keeps states valid.
        #[test]
        fn conjugation_is_an_automorphism(s1 in any::<u64>(), s2 in any::<u64>(),
                                          len in 0usize..30, s in 0..N_SYM) {
            let t = sym::tables();
            let (a, b) = (random_state(s1, len), random_state(s2, 30 - len));
            prop_assert_eq!(
                t.conjugate(&a.then(&b), s),
                t.conjugate(&a, s).then(&t.conjugate(&b, s))
            );
            prop_assert_eq!(t.conjugate(&a.inverse(), s), t.conjugate(&a, s).inverse());
            prop_assert_eq!(t.conjugate(&Cube3State::SOLVED, s), Cube3State::SOLVED);
            prop_assert!(t.conjugate(&a, s).is_valid());
        }

        /// Conjugating a scramble is the same as playing the conjugated moves, so the
        /// conjugate is exactly as far from solved.
        #[test]
        fn conjugate_of_scramble_is_scramble_of_conjugates(seed in any::<u64>(),
                                                           len in 0usize..30, s in 0..N_SYM) {
            let t = sym::tables();
            let scramble = random_scramble(&mut StdRng::seed_from_u64(seed), len);
            let moves: Vec<FaceTurn> = FaceTurn::all().collect();
            let conj: Vec<FaceTurn> = scramble
                .iter()
                .map(|&m| moves[t.move_conj[move_index(m)][s] as usize])
                .collect();
            prop_assert_eq!(
                t.conjugate(&Cube3State::SOLVED.then_all(&scramble), s),
                Cube3State::SOLVED.then_all(&conj)
            );
        }

        /// The two-phase coordinates are carried along by the 16 `D4h` symmetries: the
        /// coordinate of a conjugate depends only on the coordinate of the original, which
        /// is what lets `symcoord` tabulate conjugation on coordinates alone.
        #[test]
        fn coordinates_are_consistent_under_d4h(seed in any::<u64>(), len in 0usize..40,
                                                s in 0..N_SYM_D4H) {
            let t = sym::tables();
            let c = random_state(seed, len);
            let conj = t.conjugate(&c, s);
            let mut x = Cube3State::SOLVED;
            set_twist(&mut x, twist(&c));
            prop_assert_eq!(twist(&t.conjugate(&x, s)), twist(&conj));
            let mut x = Cube3State::SOLVED;
            set_slice(&mut x, slice(&c));
            set_flip(&mut x, flip(&c));
            let xc = t.conjugate(&x, s);
            prop_assert_eq!((slice(&xc), flip(&xc)), (slice(&conj), flip(&conj)));
            let mut x = Cube3State::SOLVED;
            set_cperm(&mut x, cperm(&c));
            prop_assert_eq!(cperm(&t.conjugate(&x, s)), cperm(&conj));
            let mut x = Cube3State::SOLVED;
            set_slice_sorted(&mut x, slice_sorted(&c));
            prop_assert_eq!(slice(&x), slice(&c));
            prop_assert_eq!(slice_sorted(&x), slice_sorted(&c));
        }

        /// Inside `G1` the U/D-edge and slice permutations are consistent too, and
        /// `slice_sorted` reduces to `sperm`.
        #[test]
        fn g1_coordinates_are_consistent_under_d4h(seed in any::<u64>(), len in 0usize..30,
                                                   s in 0..N_SYM_D4H) {
            let t = sym::tables();
            let g1: Vec<FaceTurn> = "U U2 U' D D2 D' R2 L2 F2 B2"
                .split(' ')
                .map(|m| crate::cube::parse_sequence(m).unwrap()[0])
                .collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let c = Cube3State::SOLVED
                .then_all(&(0..len).map(|_| g1[rng.gen_range(0..10)]).collect::<Vec<_>>());
            let conj = t.conjugate(&c, s);
            prop_assert_eq!((twist(&conj), flip(&conj), slice(&conj)), (0, 0, SLICE_SOLVED));
            prop_assert_eq!(slice_sorted(&c) as usize % N_SPERM, sperm(&c) as usize);
            let mut x = Cube3State::SOLVED;
            set_udeperm(&mut x, udeperm(&c));
            prop_assert_eq!(udeperm(&t.conjugate(&x, s)), udeperm(&conj));
        }

        /// The symmetry-reduced phase-1 table gives the same exact distance for all 16
        /// `D4h` conjugates, never more than the scramble length or the phase-1 diameter
        /// (12), and agrees with a brute-force check at small distances.
        #[test]
        fn phase1_distance_is_symmetric_and_exact(seed in any::<u64>(), len in 0usize..30,
                                                  s in 0..N_SYM_D4H) {
            use super::super::twophase::phase1_distance;
            let t = sym::tables();
            let c = random_state(seed, len);
            let d = phase1_distance(&c);
            prop_assert!(d <= len.min(12));
            prop_assert_eq!(phase1_distance(&t.conjugate(&c, s)), d);
            prop_assert_eq!(phase1_distance(&c.inverse().inverse()), d);
            if len <= 1 {
                prop_assert_eq!(d, if c.co == [0; 8] && c.eo == [0; 12]
                    && c.ep[8..].iter().all(|&e| e >= 8) { 0 } else { 1 });
            }
        }

        /// Likewise for the phase-2 corner x U/D-edge table, inside `G1`.
        #[test]
        fn phase2_distance_is_symmetric(seed in any::<u64>(), len in 0usize..30,
                                        s in 0..N_SYM_D4H) {
            use super::super::twophase::phase2_corner_edge_distance;
            let t = sym::tables();
            let g1: Vec<FaceTurn> = "U U2 U' D D2 D' R2 L2 F2 B2"
                .split(' ')
                .map(|m| crate::cube::parse_sequence(m).unwrap()[0])
                .collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let c = Cube3State::SOLVED
                .then_all(&(0..len).map(|_| g1[rng.gen_range(0..10)]).collect::<Vec<_>>());
            let d = phase2_corner_edge_distance(&c);
            prop_assert!(d <= len);
            prop_assert_eq!(phase2_corner_edge_distance(&t.conjugate(&c, s)), d);
        }

        #[test]
        fn inverse_is_a_two_sided_inverse(seed in any::<u64>(), len in 0usize..40) {
            let scramble = random_scramble(&mut StdRng::seed_from_u64(seed), len);
            let c = Cube3State::SOLVED.then_all(&scramble);
            prop_assert_eq!(c.then(&c.inverse()), Cube3State::SOLVED);
            prop_assert_eq!(c.inverse().then(&c), Cube3State::SOLVED);
            prop_assert_eq!(c.inverse(), Cube3State::SOLVED.then_all(&invert_sequence(&scramble)));
        }
    }
}
