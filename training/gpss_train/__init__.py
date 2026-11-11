"""gpss_train: DeepCubeA-style training for the Geometry Puzzle Solver Suite.

Trains a cost-to-go network J(s) with deep approximate value iteration (DAVI) and exports
it to ONNX for the Rust solver (``gpss solve --solver bwas --weights ...``). The method is
from Agostinelli, McAleer, Shmakov & Baldi, "Solving the Rubik's cube with deep
reinforcement learning and search", Nature Machine Intelligence 1:356-363 (2019),
doi:10.1038/s42256-019-0070-z.

Puzzles are never re-implemented here: training data (encoded states, their successors
and goal masks) comes from the Rust binary (``gpss scramble-data``), so the move function
and the state encoding have exactly one definition (ADR 0008).

Modules:
    data    - read the Rust data format; run ``gpss scramble-data``
    model   - the residual MLP
    davi    - the training loop
    export  - ONNX export + weights manifest
    launch  - local / AWS / GCP training launchers (cost estimate + confirmation)
    cli     - ``python -m gpss_train ...``
"""

__version__ = "0.1.0"

AGOSTINELLI_2019 = (
    "F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. Solving the Rubik's cube with deep "
    "reinforcement learning and search. Nature Machine Intelligence 1:356-363, 2019. "
    "doi:10.1038/s42256-019-0070-z"
)
