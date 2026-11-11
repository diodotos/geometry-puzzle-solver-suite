# Training learned heuristics (DeepCubeA-style)

How to train, export, ship and use a cost-to-go network for any puzzle that implements
`Encodable`. Theory: [`theory/deepcubea.md`](theory/deepcubea.md). Decisions:
[ADR 0005](decisions/0005-learned-heuristics-as-heuristics.md) (networks are heuristics),
[ADR 0008](decisions/0008-rust-owns-learned-encodings.md) (encodings and weights format),
[ADR 0009](decisions/0009-training-launchers.md) (cloud launchers).

The method is due to Agostinelli, McAleer, Shmakov and Baldi, *Solving the Rubik's cube
with deep reinforcement learning and search*, Nature Machine Intelligence 2019 [Ago19].
Everything here (DAVI, the residual network shape, BWAS) follows that paper; deviations
are listed at the end.

## TL;DR

```bash
# 0. One-off setup (CPU is fine for the 2x2x2)
pip install -r training/requirements.txt
cargo build --release --features gpss-cli/learn      # `gpss` with ONNX inference

# 1. Use the shipped 2x2x2 network
./target/release/gpss solve --scramble "R U2 F' R2 U F2 R' U' F R2 U" \
    --solver bwas --weights weights/cube2/cube2-davi-v1.json --lambda 0.6 --batch-size 100

# 2. Train your own (resumable; ~2 h on 8 CPU cores for the shipped config)
./target/release/gpss train train --puzzle cube2 --run-dir training/runs/cube2 --updates 400
./target/release/gpss train export --checkpoint training/runs/cube2/checkpoint.pt \
    --out weights/cube2/my-net

# 3. Evaluate exactly against the full 2x2x2 state space
./target/release/gpss evaluate --weights weights/cube2/my-net.json --lambda 0.6 --batch-size 100

# 4. Plan a cloud run (dry run: prints cost estimate + exact commands, launches nothing)
./target/release/gpss train launch aws --bucket s3://my-bucket/gpss --budget-usd 5
```

`gpss train <args>` is a thin wrapper for `python3 -m gpss_train <args>` with
`PYTHONPATH=training` and `GPSS_BIN` set to the running binary; you can call the Python
module directly from `training/` instead.

## Pieces and where they live

| piece | location | language |
|-------|----------|----------|
| state encoding (`Encodable`, `EncodingSpec`) | `crates/gpss-core/src/encode.rs`, per puzzle in `gpss-puzzles` | Rust |
| training data generator (`gpss scramble-data`) | `crates/gpss-learn/src/data.rs` | Rust |
| weights manifest + checksum validation | `crates/gpss-learn/src/manifest.rs` | Rust |
| ONNX inference (`OnnxHeuristic`, feature `onnx`) | `crates/gpss-learn/src/onnx.rs` | Rust |
| batch weighted A* (`BatchWeightedAStar`) | `crates/gpss-core/src/search/bwas.rs` | Rust |
| exact evaluation (`gpss evaluate`) | `crates/gpss-learn/src/eval.rs` | Rust |
| network, DAVI loop, ONNX export | `training/gpss_train/{model,davi,export}.py` | Python |
| launchers (`local`, `aws`, `gcp`) | `training/gpss_train/launch/` | Python |
| shipped weights | `weights/<puzzle>/<name>.{onnx,json}` | data |

## How the encoding stays in sync

Rust is the single source of truth (ADR 0008). Python never implements a puzzle:

1. `Encodable::encode` maps a state to `len` categorical features in `0..classes`
   (2x2x2: 8 corner slots, feature `3 * piece + twist`, 24 classes; key
   `cube2/corner-cubie/v1`).
2. `gpss scramble-data --puzzle cube2 --count N --min-scramble 1 --max-scramble K` writes a
   JSON header line (`format: gpss-davi-data`, the `EncodingSpec`, move names, array
   shapes) followed by raw `u8` arrays: `states [N, len]`, `children [N, moves, len]`,
   `child_is_goal [N, moves]`, `is_goal [N]`, `scramble_depth [N]`.
3. The trainer reads those bytes, stores the encoding spec in its checkpoint, and the
   exporter copies it into the manifest.
4. `OnnxHeuristic::load` compares the manifest's spec with `puzzle.encoding()` field by
   field, and the ONNX file with the manifest's SHA-256, and refuses on any mismatch.

**If you change an encoding, bump `EncodingSpec::version`.** Old weights then fail to load
instead of silently producing garbage.

### Making a new puzzle trainable

1. Implement `gpss_core::Encodable` for it (keep `len * classes` modest; the network
   one-hot-expands internally).
2. Add a `--puzzle` arm to `gpss scramble-data` / `gpss solve` / `gpss evaluate` in
   `gpss-cli` (one line each once the puzzle is in the CLI).
3. Add a preset to `PRESETS` in `training/gpss_train/davi.py`: at least `max_scramble`
   (should exceed the puzzle's diameter or typical optimal length; DeepCubeA used 30 for the
   3x3x3), and larger `hidden1/hidden2/blocks` for big puzzles (the paper's 3x3x3 net is
   5000 / 1000 / 4 blocks).
4. Train, export, evaluate. No other Python change is needed.

## The trainer (`gpss_train train`)

DAVI as in [Ago19]:

- each **update** draws `states_per_update` states scrambled `k ~ U{min..max}` moves from
  the goal (fresh data every update), with all successors;
- targets `y(s) = min_a [1 + J'(A(s,a))]` with `J'(goal) = 0`, `y(goal) = 0`, from the
  frozen target network `J'` (zero before the first sync); negative `J'` outputs are
  clamped to 0;
- `iters_per_update` Adam steps of `batch_size` on MSE;
- if the update's mean loss is below `loss_threshold` (0.05, as in the paper), `J' <- J`;
  as a fallback, also after `max_updates_per_sync` (10) updates without a sync (logged as
  `forced_sync`).

Distances propagate one move per sync, so the target maximum `y_max` in the log climbs by
about one per sync until it reaches the puzzle's diameter (11 for the 2x2x2 in the
half-turn metric). After that, further updates refine accuracy.

Main options (all have defaults; `--help` lists them):

| option | default | meaning |
|--------|---------|---------|
| `--puzzle` | `cube2` | must be supported by `gpss scramble-data` |
| `--run-dir` | `runs/<puzzle>` | checkpoint, `log.jsonl`, `config.json` |
| `--updates` | 400 | number of DAVI updates |
| `--states-per-update` | 100000 | fresh scrambled states per update |
| `--iters-per-update` / `--batch-size` | 250 / 1000 | Adam steps per update and minibatch |
| `--min-scramble` / `--max-scramble` | 1 / 14 (cube2) | scramble depth range |
| `--loss-threshold` | 0.05 | target-network sync threshold |
| `--max-updates-per-sync` | 10 | force a sync after this many updates without one (0 = never) |
| `--hidden1` `--hidden2` `--blocks` | 512 256 3 | network size |
| `--lr` `--lr-decay` | 1e-3, 0.9999993 | Adam LR, per-step exponential decay |
| `--threads` | all | torch intra-op threads |
| `--resume` | off | continue from `run_dir/checkpoint.pt` |

Output: `run_dir/checkpoint.pt` (written atomically after every update),
`run_dir/log.jsonl` (one JSON record per update: loss, syncs, target mean/max, timings).

### Network

`CostToGo` (`model.py`): one-hot input -> `hidden1` -> `hidden2` -> `blocks` residual
blocks of width `hidden2` -> scalar, batch norm + ReLU, i.e. the DeepCubeA shape at a
smaller size. The one-hot first layer is an embedding sum (identical maths, no one-hot
tensor), exported to ONNX as Gather + ReduceSum. ONNX input `state: int64[batch, len]`,
output `cost_to_go: float32[batch]`.

## Export and the weights manifest (`gpss_train export`)

Writes `<out>.onnx` (opset 18, dynamic batch) and `<out>.json`, and checks ONNX Runtime
output against PyTorch on random inputs (max abs diff < 1e-3). Manifest fields:

```json
{
  "format": "gpss-weights", "format_version": 1, "name": "cube2-davi-v1",
  "encoding": {"puzzle": "cube2", "scheme": "corner-cubie", "version": 1, "len": 8, "classes": 24},
  "model": {"file": "cube2-davi-v1.onnx", "sha256": "...", "input": "state", "output": "cost_to_go", "opset": 18},
  "architecture": {"type": "resmlp-onehot", "hidden1": 512, ...},
  "training": {"method": "DAVI ...", "config": {...}, "updates": 400, "target_syncs": ..., ...},
  "evaluation": {...},
  "references": ["F. Agostinelli, ... 2019 ..."]
}
```

`--evaluation report.json` embeds an evaluation summary. If you edit the ONNX file you
must re-export (the checksum will no longer match).

## Inference

- Build with `--features gpss-cli/learn` (or depend on `gpss-learn` with feature `onnx`).
  The default build has no ONNX dependency.
- ONNX Runtime (>= 1.22) is loaded at run time from `$ORT_DYLIB_PATH`, else from the
  Python `onnxruntime` package, else the system library path.
- `gpss solve --solver bwas --weights <manifest.json> [--lambda L] [--batch-size N]
  [--threads T]`. `--weights` also accepts the `.onnx` (the manifest is found next to it).
  Without `--weights`, `bwas` uses the classical `--heuristic` (with `--lambda 1` and an
  admissible heuristic it is optimal; CI checks this on the full corpus).
- In code: `OnnxHeuristic::load(puzzle, path, threads)?` gives a `Heuristic<P>`;
  pass it to `BatchWeightedAStar::new(h, lambda, batch)`. It always reports
  `is_admissible() == false`, so solutions are never claimed optimal.

`lambda` and `batch_size` trade solution length against nodes expanded: `lambda = 1` is
plain A* order on the learned estimate, smaller `lambda` is greedier. Larger batches mean
fewer network calls (better throughput) but more nodes expanded.

## Evaluation (`gpss evaluate`)

For the 2x2x2 the whole fixed-frame state space (3,674,160 states) is enumerated by BFS,
which gives the exact distance of every state. `gpss evaluate` reports, as JSON:

- heuristic error over **all** states, overall and per true depth: mean estimate, mean
  absolute error, RMSE, over-estimation rate and maximum, fraction exact after rounding;
- BWAS over a stratified sample (`--per-depth` states per optimal depth) and a uniform
  random sample (`--random`): % solved within `--max-expanded`, mean length vs optimal,
  % optimal, nodes expanded (mean / median / p95 / max), wall time.

The committed report for the shipped weights is
[`benchmarks/cube2-learned.md`](benchmarks/cube2-learned.md).

## Cloud training (`gpss_train launch`)

```bash
python -m gpss_train launch local                                 # estimate + plan only
python -m gpss_train launch aws --bucket s3://b/gpss --machine g4dn.xlarge \
    --max-hours 6 --budget-usd 5 --updates 2000 --puzzle cube2    # dry run
python -m gpss_train launch gcp --bucket gs://b/gpss --extra project=my-proj \
    --budget-usd 5 --execute                                      # asks "yes" first
```

All backends share one flow (ADR 0009): a cost/time **estimate**, the exact **plan**
(commands and VM bootstrap script), and a launch only with `--execute` plus an interactive
`yes` (or `--yes`). Cloud launches also need `--budget-usd` and are refused when the
estimate exceeds it. Each VM shuts itself down after `--max-hours` and is created so that
shutdown deletes it (on GCP a guest shutdown only *stops* the VM and its disk keeps billing,
so the bootstrap deletes its own instance on exit, and `--max-run-duration` with
`--instance-termination-action=DELETE` is the provider-side backstop); artefacts (checkpoint, log, ONNX + manifest) go to `--bucket`.
Training itself is stopped (`timeout --signal=INT`) 5-15 minutes before that deadline, so
the last checkpoint is still exported and uploaded and the run can be resumed (a
`TIME_LIMIT` file marks this). If training crashes instead, the last checkpoint is still
uploaded, next to a `TRAINING_FAILED` file, and the bootstrap exits with the error. The trainer
uses CUDA automatically when PyTorch sees a GPU.

**Spot / preemptible capacity (`--spot`).** The provider may reclaim (and, with these
settings, delete) the VM at any time, so the bootstrap does not rely on the end-of-run upload:

1. before training it restores `<bucket>/<puzzle>-davi/run/` into the run directory (if
   present; a failed restore of an existing checkpoint aborts instead of starting over), and
   training always starts with `--resume`;
2. during training a background loop mirrors the run directory (atomic `checkpoint.pt`,
   `log.jsonl`, `config.json`) back to that prefix every 10 minutes
   (`gcloud storage rsync` / `aws s3 sync`), plus a final mirror once training stops.

A reclaim therefore loses at most ~10 minutes of training. To continue, re-run the **same**
launch command (same `--bucket` and `--puzzle`); the new VM picks the run up where the
mirror left it. Nothing relaunches automatically; budget each relaunch with `--budget-usd`.

**GPU images (GCP).** GPU machine types (e.g. `g2-standard-4`, 1x L4) boot the Deep Learning
VM image family `common-cu129-ubuntu-2204-nvidia-580`, which ships the NVIDIA driver; it
needs a boot disk of at least 50 GB (the launcher uses 100 GB). CPU machine types keep plain
Ubuntu 22.04. The pip `torch` wheel bundles CUDA, so no toolkit install is needed.

**3x3x3.** `--puzzle cube3` uses the `cubie` v1 encoding (8 corners as `3*piece+twist`,
12 edges as `2*piece+flip`; 20 positions x 24 classes) and a preset matching DeepCubeA's
3x3x3 setup: scrambles 1-30, a (5000, 1000) network with four residual blocks, batch 10,000,
1M states per value-iteration update. It is far too slow for a CPU; see the cloud section.

Prerequisites you set up yourself (nothing here touches credentials):

- **AWS:** `aws` CLI configured (`aws configure` or SSO); an S3 bucket; an instance profile
  allowed to write to it (`--extra instance_profile=NAME`); optionally `--extra key_name=`.
- **GCP:** `gcloud auth login`; a project (`--extra project=ID` or gcloud default); a GCS
  bucket; optionally a VM service account (`--extra service_account=EMAIL`). The VM's
  account needs Storage write and Compute Instance Admin (the VM deletes itself on exit).

Machine tables (prices, relative speeds) are in `launch/aws.py` and `launch/gcp.py`. The
reference speed (1.0) is the 8-vCPU VM that trained the shipped weights; other speeds,
especially GPUs, are extrapolations and the estimate marks them as such. Check current
provider prices before relying on the dollar figure.

Status: the cloud backends are unit-tested with mocks and dry runs only. They have not been
run against a real account yet.

### Measured: 3x3x3 on one L4 (GCP spot pilot)

`cube3` preset (15.4 M parameters), `g2-standard-4` spot in europe-west4, before the
uint8-transfer/TF32 change:

| per update (1 M states) | seconds |
|---|---|
| Rust data generation | 2.5 |
| Bellman targets (18 M child evaluations) | 89 |
| 250 SGD steps, batch 10,000 (~66 k states/s) | 38 |
| **total** | **~128** |

So ~110 updates fit in 4 h. Targets dominate, which is why `bellman_targets` ships the uint8
encoding to the device and keeps results there, and TF32 matmuls are enabled on GPU.
The first pilot also showed that the old `Embedding(...).sum(1)` first layer materialised a
`[batch, 20, 5000]` tensor (24 GB at the target chunk size) and ran out of L4 memory; the
layer now uses `embedding_bag` while training.

## Deviations from the paper

- Smaller network for the 2x2x2 (630k parameters instead of ~14.7M); same shape.
- Scramble range `1..14` instead of `1..30` (the 2x2x2's diameter is 11).
- Fixed number of updates instead of training until a time budget. The sync rule
  (loss < 0.05) is the paper's, plus a forced sync after 10 updates without one: with the
  small 2x2x2 network the loss plateaued around 0.08 once targets reached depth ~8, which
  stalled value propagation (observed in the shipped run, updates 14-52).
- The 2x2x2 searches in a fixed frame with `U R F` only (ADR 0004), so the network never
  sees whole-cube rotations.
- BWAS goal test happens when a node is popped (as in A*), which makes `lambda = 1` with an
  admissible heuristic provably optimal; the paper's implementation stops as soon as a
  goal is generated.
