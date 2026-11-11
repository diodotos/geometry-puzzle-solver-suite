"""``aws``: one EC2 instance launched with the ``aws`` CLI; artefacts go to S3.

Prerequisites (yours, not ours): ``aws configure`` (or SSO) with permission to run
instances, an S3 bucket, and an instance profile that can write to it
(``--extra instance_profile=<name>``). The instance is created with
``InstanceInitiatedShutdownBehavior=terminate`` so the bootstrap's ``shutdown`` deletes it.
"""

from __future__ import annotations

from .base import Launcher, LaunchPlan, MachineType, Resources
from .bootstrap import bootstrap_script
from ..davi import DaviConfig

# Ubuntu 22.04 LTS amd64, resolved by AWS at launch time via SSM (region independent).
UBUNTU_AMI = "resolve:ssm:/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"


class AwsLauncher(Launcher):
    name = "aws"
    default_machine = "c7i.2xlarge"
    machines = {
        "c7i.2xlarge": MachineType("c7i.2xlarge", 8, 0.357, speed=1.0),
        "c7i.4xlarge": MachineType("c7i.4xlarge", 16, 0.714, speed=1.8),
        "c7i.8xlarge": MachineType("c7i.8xlarge", 32, 1.428, speed=3.0),
        "g4dn.xlarge": MachineType("g4dn.xlarge", 4, 0.526, speed=10.0, gpu="1x T4"),
        "g6.xlarge": MachineType("g6.xlarge", 4, 0.805, speed=14.0, gpu="1x L4"),
    }

    def plan(self, cfg: DaviConfig, res: Resources) -> LaunchPlan:
        if not res.bucket:
            raise ValueError("aws: --bucket s3://<bucket>/<prefix> is required for artefacts")
        region = res.region or "us-east-1"
        machine = self.machine(res)
        artefacts = f"{res.bucket.rstrip('/')}/{cfg.puzzle}-davi"
        script = bootstrap_script(cfg, repo=res.repo, ref=res.ref, max_hours=res.max_hours,
                                  upload="aws s3 cp --recursive {src} {dst}", artefacts=artefacts,
                                  exists="aws s3 ls {path}", sync="aws s3 sync {src} {dst}",
                                  fetch="aws s3 cp {src} {dst}")
        cmd = [
            "aws", "ec2", "run-instances",
            "--region", region,
            "--image-id", UBUNTU_AMI,
            "--instance-type", machine.name,
            "--count", "1",
            "--instance-initiated-shutdown-behavior", "terminate",
            "--block-device-mappings", "DeviceName=/dev/sda1,Ebs={VolumeSize=40,DeleteOnTermination=true}",
            "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=gpss-davi},{Key=gpss,Value=train}]",
            "--user-data", "file://gpss-bootstrap.sh",
        ]
        if "instance_profile" in res.extra:
            cmd += ["--iam-instance-profile", f"Name={res.extra['instance_profile']}"]
        if "key_name" in res.extra:
            cmd += ["--key-name", res.extra["key_name"]]
        if res.spot:
            cmd += ["--instance-market-options", "MarketType=spot"]
        return LaunchPlan(self.name, self.estimate(cfg, res), [cmd],
                          files={"gpss-bootstrap.sh": script}, artefacts=artefacts)

    def launch(self, plan: LaunchPlan):
        with _written(plan.files):
            return super().launch(plan)


class _written:
    """Write plan files to the working directory for the duration of the launch."""

    def __init__(self, files):
        self.files = files

    def __enter__(self):
        import pathlib
        clash = [name for name in self.files if pathlib.Path(name).exists()]
        if clash:
            raise FileExistsError(f"refusing to overwrite {', '.join(clash)} in the working directory; move it first")
        for name, body in self.files.items():
            pathlib.Path(name).write_text(body)

    def __exit__(self, *exc):
        import pathlib
        for name in self.files:
            pathlib.Path(name).unlink(missing_ok=True)
