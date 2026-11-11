# 0009. Training launchers: estimate, plan, confirm; provider CLIs, no stored credentials

**Status:** Accepted

## Context

3x3x3 and larger networks need GPU hours. The repository owner wants to "specify cloud
resources and we invoke those directly", starting with AWS and GCP. Cloud launches cost
real money, can leak resources if a VM outlives its job, and need credentials that must
never end up in the repository or in an agent's context.

## Decision

- One interface, `gpss_train.launch.Launcher`, with backends `local`, `aws`, `gcp`
  (`python -m gpss_train launch <backend> ...`, also `gpss train launch ...`).
- Every launch goes through `confirm_and_launch`: **estimate** (wall time and cost from a
  throughput model calibrated on the machine that trained the shipped weights) ->
  **plan** (the exact commands and the VM bootstrap script, printed) -> nothing happens
  unless `--execute` is given, cloud launches also need `--budget-usd` and are refused when
  the estimate exceeds it, and an interactive `yes` is required unless `--yes`.
- Backends drive the official CLIs (`aws ec2 run-instances`, `gcloud compute instances
  create`) through an injectable runner, not SDKs: no extra Python dependencies, the dry
  run is literally the command list, and tests replace the runner with a recorder.
- Credentials are whatever the provider CLI is already configured with. This code never
  reads, prompts for or stores them.
- Two kill switches per VM: the bootstrap schedules `shutdown -h +<max_hours>` (and powers
  off on exit, success or failure), and the VM is created so that shutdown deletes it
  (AWS `InstanceInitiatedShutdownBehavior=terminate`; GCP `--max-run-duration` +
  `--instance-termination-action=DELETE`, enforced by the provider).
- Artefacts (checkpoint, log, exported ONNX + manifest) are copied to a user-given bucket
  (`s3://` / `gs://`) before shutdown. Training is resumable (`--resume`).

## Consequences

- AWS/GCP backends are unit-tested only with mocks and dry runs; they have not yet been
  run against real accounts (tracked in `docs/roadmap.md`). The first real run needs the
  owner's credentials and an explicit go-ahead.
- Prices and GPU speed-ups in the machine tables are approximate and must be checked
  against provider pricing; the estimate says so when throughput is extrapolated.
- Adding a provider (Azure, Lambda, a Slurm cluster) means one `Launcher` subclass with a
  machine table and a `plan`.
