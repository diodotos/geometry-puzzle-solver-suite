# References

Academic and technical work this project builds on. When code or docs depend on an idea
from one of these, cite it by its key (e.g. `[Korf97]`). Bibliographic details were checked
against Crossref, arXiv, AAAI's archive, or the author's own site.

## Search

- **[HNR68]** P. E. Hart, N. J. Nilsson, B. Raphael. *A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.* IEEE Transactions on Systems Science and
  Cybernetics 4(2):100-107, 1968. doi:10.1109/TSSC.1968.300136.
  A* search and the admissibility/optimality result. Used by `gpss_core::search::AStar`.
- **[Pohl70]** I. Pohl. *Heuristic search viewed as path finding in a graph.* Artificial
  Intelligence 1(3-4):193-204, 1970. doi:10.1016/0004-3702(70)90007-X.
  Weighting the heuristic term (weighted A*), which trades optimality for speed. Used by
  `AStar::weighted` and, later, by batch weighted A* [Ago19].
- **[Korf85]** R. E. Korf. *Depth-first iterative-deepening: An optimal admissible tree
  search.* Artificial Intelligence 27(1):97-109, 1985. doi:10.1016/0004-3702(85)90084-0.
  Introduces IDA*: optimal with an admissible heuristic, memory linear in solution depth.
  Used by `gpss_core::search::IdaStar`.

## Pattern databases and optimal Rubik's Cube solving

- **[CS98]** J. C. Culberson, J. Schaeffer. *Pattern Databases.* Computational
  Intelligence 14(3):318-334, 1998. doi:10.1111/0824-7935.00065.
  Exact distances of an abstraction, stored in a table, as an admissible heuristic. The
  basis of `gpss_core::pdb`.
- **[Korf97]** R. E. Korf. *Finding Optimal Solutions to Rubik's Cube Using Pattern
  Databases.* Proceedings of the Fourteenth National Conference on Artificial Intelligence
  (AAAI-97), 1997. <https://cdn.aaai.org/AAAI/1997/AAAI97-109.pdf>.
  IDA* with a maximum over corner and edge pattern databases: the first optimal solutions
  of random 3x3x3 positions. The planned 3x3x3 optimal solver follows this design.
- **[Rok13]** T. Rokicki, H. Kociemba, M. Davidson, J. Dethridge. *The Diameter of the
  Rubik's Cube Group Is Twenty.* SIAM Journal on Discrete Mathematics 27(2):1082-1105,
  2013. doi:10.1137/120867366. (Reprinted in SIAM Review 56(4):645-670, 2014,
  doi:10.1137/140973499.) God's number for the 3x3x3 in the half-turn metric; used to
  bound the 3x3x3 benchmark range.

## Two-phase (near-optimal) solving

- **[Koc]** H. Kociemba. *The Two-Phase Algorithm* and *Two-Phase Algorithm Details.*
  <http://kociemba.org/math/twophase.htm>, <http://kociemba.org/math/imptwophase.htm>.
  Kociemba's own description of the algorithm he developed in 1991-92: phase 1 reaches
  the subgroup G1 = <U, D, R2, L2, F2, B2>, phase 2 solves within G1, and search continues
  over suboptimal phase-1 solutions to shorten the total. Also the source of the cubie
  and coordinate conventions (corner order `URF UFL ULB UBR DFR DLF DBL DRB`, "replaced-by"
  move tables) used in `gpss_puzzles::cube2`. The planned 3x3x3 fast solver follows this.
- **[KocCE]** H. Kociemba. *Cube Explorer.* <http://kociemba.org/cube.htm>. Reference
  implementation of the two-phase algorithm.

## Learned heuristics and self-play

- **[Ago19]** F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. *Solving the Rubik's cube
  with deep reinforcement learning and search.* Nature Machine Intelligence 1:356-363,
  2019. doi:10.1038/s42256-019-0070-z.
  DeepCubeA: a cost-to-go network trained by deep approximate value iteration (DAVI) on
  states scrambled backwards from the goal, then used as the heuristic of batch weighted
  A* (BWAS). Solves the 3x3x3 and other puzzles with near-optimal solutions. This is the
  blueprint for the planned learned solver; see `docs/theory/deepcubea.md`.
- **[McA18]** S. McAleer, F. Agostinelli, A. Shmakov, P. Baldi. *Solving the Rubik's Cube
  Without Human Knowledge.* arXiv:1805.07470, 2018. The predecessor (DeepCube, autodidactic
  iteration + MCTS).

## The 2x2x2 state space

The exact half-turn-metric distribution of the 2x2x2 (3,674,160 positions, God's number
11) is widely published (e.g. Wikipedia, "Pocket Cube"); this repository does not rely on
that citation: `conformance/generate_cube2.py` and `cube2::tests::full_state_space_matches_published_distribution`
recompute it exhaustively with two independent implementations.

## Software

- [`argparse.h`](https://github.com/p-ranav/argparse) by Pranav Srinivas Kumar, vendored in
  `legacy/cpp/` (MIT).
