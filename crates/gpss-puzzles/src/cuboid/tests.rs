use super::*;
use gpss_core::search::{Bfs, IdaStar};
use gpss_core::{Heuristic, ZeroHeuristic};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cuboid(s: &str) -> Cuboid {
    Cuboid::new(s.parse().unwrap())
}

fn order(c: &Cuboid, mv: LayerTurn) -> usize {
    let mut s = c.apply(&c.solved(), mv);
    let mut n = 1;
    while s != c.solved() {
        s = c.apply(&s, mv);
        n += 1;
    }
    n
}

#[test]
fn dims_parse_and_display() {
    let d: Dims = "2x3x4".parse().unwrap();
    assert_eq!(d.layers(Axis::X), 2);
    assert_eq!(d.layers(Axis::Z), 3);
    assert_eq!(d.layers(Axis::Y), 4);
    assert_eq!(d.to_string(), "2x3x4");
    for bad in ["", "3x3", "1x3x3", "3x3x3x3", "axbxc", "3x3x33", "3x-3x3"] {
        assert!(bad.parse::<Dims>().is_err(), "{bad}");
    }
}

#[test]
fn sticker_and_move_counts() {
    let c = cuboid("3x3x3");
    assert_eq!(c.sticker_count(), 54);
    assert_eq!(c.moves().len(), 27);
    let domino = cuboid("3x3x2");
    assert_eq!(domino.sticker_count(), 2 * 9 + 4 * 6);
    // U-D axis: 2 layers x 3 turns; R-L and F-B axes: 3 layers x half turn only.
    assert_eq!(domino.moves().len(), 6 + 3 + 3);
    assert_eq!(cuboid("2x3x4").moves().len(), 2 + 3 + 4);
}

#[test]
fn move_orders_and_inverses() {
    for dims in ["2x2x2", "3x3x3", "4x4x4", "3x3x2", "2x3x4"] {
        let c = cuboid(dims);
        let s = c.scrambled(&c.random_scramble(&mut StdRng::seed_from_u64(1), 20));
        for &mv in c.moves() {
            assert_eq!(
                order(&c, mv),
                if mv.turns == 2 { 2 } else { 4 },
                "{dims} {mv}"
            );
            assert_eq!(c.apply(&c.apply(&s, mv), c.inverse(mv)), s, "{dims} {mv}");
        }
    }
}

#[test]
fn notation_round_trips_and_negative_faces() {
    let c = cuboid("4x4x4");
    for &mv in c.moves() {
        assert_eq!(c.parse_move(&mv.to_string()).unwrap(), mv);
    }
    // On an N-layer axis, kX (negative face) is layer N-k turned the other way.
    assert_eq!(
        c.parse_move("L").unwrap(),
        LayerTurn {
            axis: Axis::X,
            layer: 3,
            turns: 3
        }
    );
    assert_eq!(
        c.parse_move("2D2").unwrap(),
        LayerTurn {
            axis: Axis::Y,
            layer: 2,
            turns: 2
        }
    );
    // Turning every layer the same way is a whole-puzzle rotation: still solved.
    let rotation = c.parse_sequence("R 2R 3R 4R").unwrap();
    assert!(c.is_solved(&c.scrambled(&rotation)));
    assert!(!c.is_solved(&c.scrambled(&rotation[..3])));
    assert_eq!(
        c.scrambled(&c.parse_sequence("R L'").unwrap()),
        c.scrambled(&c.parse_sequence("L' R").unwrap())
    );
}

#[test]
fn invalid_tokens_are_rejected() {
    let domino = cuboid("3x3x2");
    assert!(matches!(
        domino.parse_move("R"),
        Err(ParseLayerTurnError::QuarterNotAllowed(_))
    ));
    assert!(domino.parse_move("R2").is_ok());
    assert!(domino.parse_move("U'").is_ok());
    assert!(matches!(
        domino.parse_move("3U"),
        Err(ParseLayerTurnError::LayerOutOfRange(_))
    ));
    assert!(matches!(
        domino.parse_move("0R2"),
        Err(ParseLayerTurnError::LayerOutOfRange(_))
    ));
    for bad in ["", "X", "R3", "R''", "2", "r", "Rw"] {
        assert!(
            matches!(domino.parse_move(bad), Err(ParseLayerTurnError::Syntax(_))),
            "{bad:?}"
        );
    }
}

#[test]
fn face_turns_match_cube3_model() {
    use crate::cube::parse_sequence;
    use crate::cube3::Cube3;
    // Same scramble on both models: solved-ness of every prefix must agree, and the
    // 3x3x3 solution from the specialised model must solve the generic one.
    let c = cuboid("3x3x3");
    let scramble = "R U F' D2 L B' U2 R' F L2 D' B2";
    let solution = Cube3::new()
        .solve_scramble(
            &parse_sequence(scramble).unwrap(),
            &crate::cube3::TwoPhase::default(),
            &SearchLimits::default(),
        )
        .unwrap();
    let all = format!(
        "{scramble} {}",
        gpss_core::puzzle::format_moves(&solution.moves)
    );
    assert!(c.is_solved(&c.scrambled(&c.parse_sequence(&all).unwrap())));
    assert!(!c.is_solved(&c.scrambled(&c.parse_sequence(scramble).unwrap())));
}

#[test]
fn optimal_search_matches_cube2_lengths() {
    use crate::cube::parse_sequence;
    use crate::cube2::{Cube2, Cube2Heuristic};
    let generic = cuboid("2x2x2");
    let cube2 = Cube2::new();
    let exact = IdaStar::new(Cube2Heuristic::Dual.build(&cube2));
    let mut rng = StdRng::seed_from_u64(7);
    for len in 1..=5 {
        let scramble = generic.random_scramble(&mut rng, len);
        let text = gpss_core::puzzle::format_moves(&scramble);
        // Generic 2x2x2 layer turns `2R`, `2U`, `2F` are L', D', B' relative to the frame;
        // the cube2 notation needs face letters, so translate via the negative faces.
        let face_text = text
            .replace("2R2", "L2")
            .replace("2U2", "D2")
            .replace("2F2", "B2")
            .replace("2R'", "L")
            .replace("2U'", "D")
            .replace("2F'", "B")
            .replace("2R", "L'")
            .replace("2U", "D'")
            .replace("2F", "B'");
        let want = cube2
            .solve_scramble(
                &parse_sequence(&face_text).unwrap(),
                &exact,
                &SearchLimits::default(),
            )
            .unwrap()
            .moves
            .len();
        let got = generic
            .solve_scramble(
                &scramble,
                &IdaStar::new(ZeroHeuristic),
                &SearchLimits::default(),
            )
            .unwrap();
        assert!(got.optimal);
        assert_eq!(got.moves.len(), want, "{text}");
        let end = generic.apply_all(&generic.scrambled(&scramble), &got.moves);
        assert!(generic.is_solved(&end));
    }
}

#[test]
fn bfs_and_idastar_agree_on_cuboids() {
    let mut rng = StdRng::seed_from_u64(3);
    for dims in ["3x3x2", "2x2x3", "2x3x4", "4x4x4", "3x3x3"] {
        let c = cuboid(dims);
        for len in [0, 1, 2, 3] {
            let s = c.scrambled(&c.random_scramble(&mut rng, len));
            let a = Bfs.solve(&c, &s, &SearchLimits::default()).unwrap();
            let b = IdaStar::new(ZeroHeuristic)
                .solve(&c, &s, &SearchLimits::default())
                .unwrap();
            assert_eq!(a.moves.len(), b.moves.len(), "{dims}");
            assert!(a.moves.len() <= len);
            assert!(c.is_solved(&c.apply_all(&s, &b.moves)));
        }
    }
}

fn face_turn_text(seq: &[crate::cube::FaceTurn]) -> String {
    seq.iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn corner_projection_matches_cube2() {
    let cube2 = crate::cube2::Cube2::new();
    let mut rng = StdRng::seed_from_u64(11);
    for dims in ["2x2x2", "3x3x3", "4x4x4", "5x5x5"] {
        let c = cuboid(dims);
        assert_eq!(
            c.corner_state(&c.solved()),
            Some(crate::cube2::Cube2State::SOLVED)
        );
        for len in 0..16 {
            let seq = crate::cube::random_scramble(&mut rng, len);
            let mut moves = Vec::new();
            for mv in c.parse_sequence(&face_turn_text(&seq)).unwrap() {
                moves.push(mv);
                if c.dims().layers(mv.axis) > 2 {
                    // Inner slices never move corners.
                    moves.push(LayerTurn {
                        layer: 1,
                        turns: 1,
                        ..mv
                    });
                }
            }
            assert_eq!(
                c.corner_state(&c.scrambled(&moves)),
                Some(cube2.scrambled(&seq)),
                "{dims} {}",
                face_turn_text(&seq)
            );
        }
    }
}

#[test]
fn corner_projection_rejects_impossible_corners() {
    let c = cuboid("3x3x3");
    let urf = c.corners[0];
    // URF's U sticker recoloured D: the remaining colours still name a real piece.
    let mut s = c.solved();
    s[urf[0] as usize] = 3;
    assert_eq!(c.corner_state(&s), None);
    // Two copies of the URF piece.
    let mut s = c.solved();
    for (dst, src) in c.corners[1].iter().zip(urf) {
        s[*dst as usize] = c.solved()[src as usize];
    }
    assert_eq!(c.corner_state(&s), None);
}

#[test]
fn corner_heuristic_is_perfect_on_2x2x2_and_zero_on_rotations() {
    let c = cuboid("2x2x2");
    let h = CornerHeuristic::new(&c);
    let mut rng = StdRng::seed_from_u64(12);
    for len in 0..14 {
        let s = c.scrambled(&c.random_scramble(&mut rng, len));
        let sol = IdaStar::new(CornerHeuristic::new(&c))
            .solve(&c, &s, &SearchLimits::default())
            .unwrap();
        assert_eq!(h.estimate(&s) as usize, sol.moves.len());
    }
    for (dims, rotation) in [
        ("3x3x3", "R 2R L'"),
        ("4x4x4", "U 2U 3U 4U"),
        ("3x3x2", "U 2U"),
        ("2x3x4", "R2 2R2"),
    ] {
        let c = cuboid(dims);
        let s = c.scrambled(&c.parse_sequence(rotation).unwrap());
        assert!(c.is_solved(&s));
        assert_eq!(CornerHeuristic::new(&c).estimate(&s), 0, "{dims}");
    }
}

#[test]
fn corner_heuristic_is_admissible_and_prunes() {
    let mut rng = StdRng::seed_from_u64(13);
    for (dims, max_len) in [
        ("3x3x2", 5),
        ("2x2x3", 5),
        ("2x3x4", 4),
        ("3x3x3", 4),
        ("4x4x4", 3),
    ] {
        let c = cuboid(dims);
        let h = CornerHeuristic::new(&c);
        let (mut blind, mut informed) = (0, 0);
        for len in 0..=max_len {
            let s = c.scrambled(&c.random_scramble(&mut rng, len));
            let limits = SearchLimits::default();
            let a = IdaStar::new(ZeroHeuristic).solve(&c, &s, &limits).unwrap();
            let b = IdaStar::new(CornerHeuristic::new(&c))
                .solve(&c, &s, &limits)
                .unwrap();
            assert_eq!(a.moves.len(), b.moves.len(), "{dims}");
            assert!(h.estimate(&s) as usize <= a.moves.len(), "{dims}");
            assert!(c.is_solved(&c.apply_all(&s, &b.moves)));
            blind += a.stats.expanded;
            informed += b.stats.expanded;
        }
        assert!(informed < blind, "{dims}: {informed} >= {blind}");
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    /// For random shapes and scrambles: every move is undone by its inverse, has order 2
    /// or 4, and a scramble followed by its inverse sequence is solved.
    #[test]
    fn prop_cuboid_algebra(
        x in 2u8..=4, y in 2u8..=4, z in 2u8..=4,
        seed in proptest::prelude::any::<u64>(),
        len in 0usize..12,
    ) {
        let c = Cuboid::new(Dims::new(x, z, y));
        let mut rng = StdRng::seed_from_u64(seed);
        let scramble = c.random_scramble(&mut rng, len);
        let s = c.scrambled(&scramble);
        for &mv in c.moves() {
            proptest::prop_assert_eq!(&c.apply(&c.apply(&s, mv), c.inverse(mv)), &s);
            let twice = c.apply(&c.apply(&s, mv), mv);
            if mv.turns == 2 {
                proptest::prop_assert_eq!(&twice, &s);
            } else {
                proptest::prop_assert_eq!(&c.apply(&c.apply(&twice, mv), mv), &s);
            }
            proptest::prop_assert_eq!(c.parse_move(&mv.to_string()).unwrap(), mv);
        }
        let back: Vec<LayerTurn> = scramble.iter().rev().map(|&m| c.inverse(m)).collect();
        proptest::prop_assert!(c.is_solved(&c.apply_all(&s, &back)));
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    /// The corner heuristic is consistent: one move changes it by at most 1.
    #[test]
    fn prop_corner_heuristic_is_consistent(
        x in 2u8..=4, y in 2u8..=4, z in 2u8..=4,
        seed in proptest::prelude::any::<u64>(),
        len in 0usize..20,
    ) {
        let c = Cuboid::new(Dims::new(x, z, y));
        let h = CornerHeuristic::new(&c);
        let mut rng = StdRng::seed_from_u64(seed);
        let s = c.scrambled(&c.random_scramble(&mut rng, len));
        let hs = h.estimate(&s);
        proptest::prop_assert!(c.corner_state(&s).is_some());
        for &mv in c.moves() {
            proptest::prop_assert!(h.estimate(&c.apply(&s, mv)).abs_diff(hs) <= 1);
        }
    }
}
