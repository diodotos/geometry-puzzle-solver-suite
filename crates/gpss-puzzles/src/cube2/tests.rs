use super::*;
use crate::cube::{invert_sequence, parse_sequence, random_scramble};
use gpss_core::pdb::PatternDatabase;
use gpss_core::search::{AStar, Bfs, IdaStar};
use proptest::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

fn seq(s: &str) -> Vec<FaceTurn> {
    parse_sequence(s).unwrap()
}

#[test]
fn every_face_turn_has_order_four() {
    let p = Cube2::new();
    for m in FaceTurn::all().filter(|m| m.turn == Turn::Cw) {
        let s = p.scrambled(&[m, m, m, m]);
        assert_eq!(s, Cube2State::SOLVED, "{m}^4");
        assert_ne!(p.scrambled(&[m]), Cube2State::SOLVED);
    }
}

#[test]
fn sexy_move_has_order_six() {
    let p = Cube2::new();
    let sexy = seq("R U R' U'");
    let rep = |n: usize| sexy.iter().cycle().take(4 * n).copied().collect::<Vec<_>>();
    assert_eq!(p.scrambled(&rep(6)), Cube2State::SOLVED);
    for n in 1..6 {
        assert_ne!(p.scrambled(&rep(n)), Cube2State::SOLVED);
    }
}

#[test]
fn twist_sum_is_invariant() {
    let p = Cube2::new();
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..100 {
        let s = p.scrambled(&random_scramble(&mut rng, 30));
        assert_eq!(s.co.iter().map(|&c| c as u32).sum::<u32>() % 3, 0);
    }
}

#[test]
fn opposite_faces_are_a_whole_cube_rotation() {
    let p = Cube2::new();
    assert!(p.scrambled(&seq("R L'")).is_solved_up_to_rotation());
    assert!(!p.scrambled(&seq("R L")).is_solved_up_to_rotation());
}

#[test]
fn regrip_fixes_dbl_and_preserves_solvedness() {
    let p = Cube2::new();
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..200 {
        let scr = random_scramble(&mut rng, 25);
        let state = p.scrambled(&scr);
        let (g, regrip) = p.regrip(&state);
        assert_eq!((g.cp[DBL], g.co[DBL]), (DBL as u8, 0));
        // Solving the regripped state with the inverse moves (mapped back) must solve
        // the original cube up to rotation.
        let undo = IdaStar::new(Cube2Heuristic::Dual.build(&p))
            .solve(&p, &g, &SearchLimits::default())
            .unwrap();
        let mapped: Vec<_> = undo.moves.iter().map(|&m| regrip.to_original(m)).collect();
        assert!(state.then_all(&mapped).is_solved_up_to_rotation());
    }
}

#[test]
fn full_state_space_matches_published_distribution() {
    let p = Cube2::new();
    let pdb = PatternDatabase::build(&p, &FullCoord);
    assert_eq!(pdb.histogram(), KNOWN_HTM_DISTRIBUTION.to_vec());
}

#[test]
fn coordinate_tables_are_complete() {
    let p = Cube2::new();
    let ori = PatternDatabase::build(&p, &OrientationCoord);
    let perm = PatternDatabase::build(&p, &PermutationCoord);
    assert_eq!(ori.histogram().iter().sum::<u64>(), 729);
    assert_eq!(perm.histogram().iter().sum::<u64>(), 5040);
    // Sub-problem diameters, cross-checked by an independent Python BFS.
    assert_eq!(ori.histogram().len() - 1, 6);
    assert_eq!(perm.histogram().len() - 1, 7);
}

#[test]
fn heuristics_are_admissible_and_ordered() {
    let p = Cube2::new();
    let perfect = Cube2Heuristic::Perfect.build(&p);
    let dual = Cube2Heuristic::Dual.build(&p);
    let ori = Cube2Heuristic::Orientation.build(&p);
    let perm = Cube2Heuristic::Permutation.build(&p);
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..2000 {
        let (s, _) = p.regrip(&p.scrambled(&random_scramble(&mut rng, 20)));
        let exact = perfect.estimate(&s);
        let d = dual.estimate(&s);
        assert!(d <= exact);
        assert_eq!(d, ori.estimate(&s).max(perm.estimate(&s)));
    }
}

#[test]
fn all_optimal_solvers_agree_with_the_perfect_table() {
    let p = Cube2::new();
    let perfect = Cube2Heuristic::Perfect.build(&p);
    let solvers: Vec<Box<dyn Solver<Cube2>>> = vec![
        Box::new(Bfs),
        Box::new(AStar::new(Cube2Heuristic::Dual.build(&p))),
        Box::new(AStar::new(Cube2Heuristic::Orientation.build(&p))),
        Box::new(IdaStar::new(Cube2Heuristic::Dual.build(&p))),
        Box::new(IdaStar::new(Cube2Heuristic::Perfect.build(&p))),
    ];
    let mut rng = StdRng::seed_from_u64(5);
    for len in [0usize, 1, 2, 4, 6, 8] {
        let scr = random_scramble(&mut rng, len);
        let (start, _) = p.regrip(&p.scrambled(&scr));
        let want = perfect.estimate(&start) as usize;
        for solver in &solvers {
            let sol = solver.solve(&p, &start, &SearchLimits::default()).unwrap();
            assert!(sol.optimal, "{}", solver.name());
            assert_eq!(sol.moves.len(), want, "{} on {:?}", solver.name(), scr);
            assert!(p.is_solved(&start.then_all(&sol.moves)));
        }
    }
}

#[test]
fn weighted_astar_solves_but_is_not_flagged_optimal() {
    let p = Cube2::new();
    let solver = AStar::weighted(Cube2Heuristic::Dual.build(&p), 2.0);
    let scr = seq("R U2 F' R2 U F2 R' U' F R2 U");
    let sol = p
        .solve_scramble(&scr, &solver, &SearchLimits::default())
        .unwrap();
    assert!(!sol.optimal);
    assert!(p
        .scrambled(&scr)
        .then_all(&sol.moves)
        .is_solved_up_to_rotation());
}

#[test]
fn limits_are_enforced() {
    let p = Cube2::new();
    let scr = seq("R U2 F' R2 U F2 R' U' F R2 U");
    let limits = SearchLimits {
        max_expanded: Some(10),
        ..Default::default()
    };
    assert!(matches!(
        p.solve_scramble(&scr, &Bfs, &limits),
        Err(SearchError::LimitReached { .. })
    ));
    let depth = SearchLimits {
        max_depth: Some(3),
        ..Default::default()
    };
    assert!(p
        .solve_scramble(&scr, &IdaStar::new(Cube2Heuristic::Dual.build(&p)), &depth)
        .is_err());
}

proptest! {
    #[test]
    fn scramble_then_inverse_is_identity(len in 0usize..40, seed in any::<u64>()) {
        let p = Cube2::new();
        let scr = random_scramble(&mut StdRng::seed_from_u64(seed), len);
        let s = p.scrambled(&scr).then_all(&invert_sequence(&scr));
        prop_assert_eq!(s, Cube2State::SOLVED);
    }

    #[test]
    fn coordinates_are_in_range(len in 0usize..40, seed in any::<u64>()) {
        let p = Cube2::new();
        let (s, _) = p.regrip(&p.scrambled(&random_scramble(&mut StdRng::seed_from_u64(seed), len)));
        prop_assert!(OrientationCoord.index(&s) < 729);
        prop_assert!(PermutationCoord.index(&s) < 5040);
        prop_assert!(FullCoord.index(&s) < STATE_COUNT);
    }
}

#[test]
fn encoding_is_injective_and_in_range() {
    use gpss_core::Encodable;
    let p = Cube2::new();
    let spec = p.encoding();
    assert_eq!(spec.key(), "cube2/corner-cubie/v1");
    let mut buf = vec![0u8; spec.len];
    p.encode(&Cube2State::SOLVED, &mut buf);
    assert_eq!(buf, [0, 3, 6, 9, 12, 15, 18, 21]);
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen = std::collections::HashMap::new();
    for _ in 0..2000 {
        let s = p.scrambled(&random_scramble(&mut rng, 20));
        p.encode(&s, &mut buf);
        assert!(buf.iter().all(|&c| (c as u16) < spec.classes));
        if let Some(prev) = seen.insert(buf.clone(), s) {
            assert_eq!(prev, s, "two states share an encoding");
        }
    }
}

#[test]
fn bwas_with_admissible_heuristic_is_optimal_and_learned_settings_stay_valid() {
    use gpss_core::search::BatchWeightedAStar;
    let p = Cube2::new();
    let dual = Cube2Heuristic::Dual.build(&p);
    let perfect = PdbHeuristic::new(&p, FullCoord);
    let optimal = BatchWeightedAStar::new(&dual, 1.0, 50);
    let greedy = BatchWeightedAStar::new(&dual, 0.3, 200);
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..40 {
        let scr = random_scramble(&mut rng, 25);
        let (start, _) = p.regrip(&p.scrambled(&scr));
        let d = perfect.estimate(&start) as usize;
        let sol = optimal.solve(&p, &start, &SearchLimits::default()).unwrap();
        assert!(sol.optimal);
        assert_eq!(sol.moves.len(), d);
        assert!(p.apply_all(&start, &sol.moves) == Cube2State::SOLVED);
        let sol = greedy.solve(&p, &start, &SearchLimits::default()).unwrap();
        assert!(!sol.optimal);
        assert!(sol.moves.len() >= d);
        assert!(p.apply_all(&start, &sol.moves) == Cube2State::SOLVED);
    }
}
