"""``gcp``: one Compute Engine VM launched with the ``gcloud`` CLI; artefacts go to GCS.

Prerequisites (yours, not ours): ``gcloud auth login``, a project
(``--extra project=<id>`` or the gcloud default), a GCS bucket, and a service account the
VM runs as that can write to it (``--extra service_account=<email>``). The VM is created
with ``--max-run-duration`` and ``--instance-termination-action=DELETE``, a provider-side
limit that holds even if the bootstrap script hangs.
"""

from __future__ import annotations

import math

from .aws import _written
from .base import Launcher, LaunchPlan, MachineType, Resources
from .bootstrap import bootstrap_script
from ..davi import DaviConfig


GPU_IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"


class GcpLauncher(Launcher):
    name = "gcp"
    default_machine = "c3-standard-8"
    machines = {
        "c3-standard-8": MachineType("c3-standard-8", 8, 0.402, speed=1.0),
        "c3-standard-22": MachineType("c3-standard-22", 22, 1.105, speed=2.4),
        "n2-standard-8": MachineType("n2-standard-8", 8, 0.389, speed=0.9),
        "g2-standard-4": MachineType("g2-standard-4", 4, 0.707, speed=14.0, gpu="1x L4"),
    }

    @staticmethod
    def image_args(machine: MachineType) -> list:
        """GPU machines boot a Deep Learning VM image that installs the NVIDIA driver."""
        if machine.gpu:
            return ["--image-family", GPU_IMAGE_FAMILY, "--image-project", "deeplearning-platform-release",
                    "--metadata", "install-nvidia-driver=True"]
        return ["--image-family", "ubuntu-2204-lts", "--image-project", "ubuntu-os-cloud"]

    def plan(self, cfg: DaviConfig, res: Resources) -> LaunchPlan:
        if not res.bucket:
            raise ValueError("gcp: --bucket gs://<bucket>/<prefix> is required for artefacts")
        zone = res.region or "us-central1-a"
        machine = self.machine(res)
        artefacts = f"{res.bucket.rstrip('/')}/{cfg.puzzle}-davi"
        script = bootstrap_script(cfg, repo=res.repo, ref=res.ref, max_hours=res.max_hours,
                                  upload="gcloud storage cp --recursive {src} {dst}", artefacts=artefacts,
                                  exists="gcloud storage ls {path}", sync="gcloud storage rsync --recursive {src} {dst}")
        cmd = [
            "gcloud", "compute", "instances", "create", "gpss-davi",
            "--zone", zone,
            "--machine-type", machine.name,
            *self.image_args(machine),
            "--boot-disk-size", "100GB" if machine.gpu else "40GB",  # DLVM GPU images are 50 GB
            "--max-run-duration", f"{math.ceil(res.max_hours * 3600)}s",
            "--instance-termination-action", "DELETE",
            "--labels", "gpss=train",
            "--metadata-from-file", "startup-script=gpss-bootstrap.sh",
            "--scopes", "storage-rw",
        ]
        if machine.gpu:
            cmd += ["--maintenance-policy", "TERMINATE"]
        if "project" in res.extra:
            cmd += ["--project", res.extra["project"]]
        if "service_account" in res.extra:
            cmd += ["--service-account", res.extra["service_account"]]
        if res.spot:
            cmd += ["--provisioning-model", "SPOT"]
        return LaunchPlan(self.name, self.estimate(cfg, res), [cmd],
                          files={"gpss-bootstrap.sh": script}, artefacts=artefacts)

    def launch(self, plan: LaunchPlan):
        with _written(plan.files):
            return super().launch(plan)
