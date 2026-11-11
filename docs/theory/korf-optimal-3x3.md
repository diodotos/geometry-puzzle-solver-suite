# Optimal 3x3x3 solving (Korf) - planned

Status: **design**. Source: Korf, *Finding Optimal Solutions to Rubik's Cube Using
Pattern Databases*, AAAI-97 [Korf97]; IDA* from [Korf85].

- Search: `IdaStar` from `gpss-core`, 18 face turns, `FaceTurn::can_follow` pruning
  (same face never twice; opposite faces in canonical order only), which reduces the
  branching factor from 18 to about 13.35.
- Heuristic: `max` of pattern databases - all 8 corners (8! x 3^7 = 88,179,840 entries),
  and two databases of 6 edges each - as in [Korf97]. Values fit in 4 bits.
- Engineering needed beyond the 2x2x2: 4-bit packed PDBs, on-disk caching of built tables
  (building them takes minutes), symmetry reduction, and cubie state with edges.
- Expectation: random 3x3x3 positions are typically 17-18 moves from solved [Rok13]; optimal
  solves take seconds to hours depending on depth and table sizes. The benchmark suite will
  bucket by optimal depth and report expanded nodes, which are machine-independent.
