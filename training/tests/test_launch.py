import pytest

from gpss_train.davi import DaviConfig
from gpss_train.launch import Resources, confirm_and_launch, get_launcher
from gpss_train.launch.base import LaunchRefused


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        return "ok"


CFG = DaviConfig(updates=50)


def res(backend, **kw):
    bucket = {"aws": "s3://b/p", "gcp": "gs://b/p"}.get(backend)
    return Resources(backend, bucket=bucket, **kw)


@pytest.mark.parametrize("backend", ["local", "aws", "gcp"])
def test_dry_run_never_executes(backend):
    rec = Recorder()
    out = []
    assert confirm_and_launch(get_launcher(backend, rec), CFG, res(backend), execute=False, assume_yes=True,
                              show=out.append) is None
    assert rec.calls == []
    assert "dry run" in out[-1] and "cost:" in out[0]


@pytest.mark.parametrize("backend", ["aws", "gcp"])
def test_cloud_requires_budget_and_confirmation(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = Recorder()
    launcher = get_launcher(backend, rec)
    quiet = dict(show=lambda s: None)
    with pytest.raises(LaunchRefused, match="budget"):
        confirm_and_launch(launcher, CFG, res(backend), execute=True, assume_yes=True, **quiet)
    with pytest.raises(LaunchRefused, match="exceeds"):
        confirm_and_launch(launcher, CFG, res(backend, budget_usd=0.01), execute=True, assume_yes=True, **quiet)
    with pytest.raises(LaunchRefused, match="not confirmed"):
        confirm_and_launch(launcher, CFG, res(backend, budget_usd=100), execute=True, assume_yes=False,
                           ask=lambda q: "no", **quiet)
    assert rec.calls == []
    confirm_and_launch(launcher, CFG, res(backend, budget_usd=100), execute=True, assume_yes=False,
                       ask=lambda q: "yes", **quiet)
    assert len(rec.calls) == 1
    assert not (tmp_path / "gpss-bootstrap.sh").exists()


def test_aws_plan_has_kill_switches():
    plan = get_launcher("aws", Recorder()).plan(CFG, res("aws", max_hours=2, spot=True,
                                                         extra={"instance_profile": "gpss"}))
    cmd = plan.commands[0]
    assert cmd[:3] == ["aws", "ec2", "run-instances"]
    assert cmd[cmd.index("--instance-initiated-shutdown-behavior") + 1] == "terminate"
    assert "Name=gpss" in cmd and "MarketType=spot" in cmd
    script = plan.files["gpss-bootstrap.sh"]
    assert "shutdown -h +120" in script and "aws s3 cp --recursive /opt/gpss/out s3://b/p/cube2-davi" in script
    assert "--updates 50" in script and "--resume" in script


def test_gcp_plan_has_kill_switches():
    plan = get_launcher("gcp", Recorder()).plan(CFG, res("gcp", max_hours=1.5, machine="g2-standard-4"))
    cmd = plan.commands[0]
    assert cmd[cmd.index("--max-run-duration") + 1] == "5400s"
    assert cmd[cmd.index("--instance-termination-action") + 1] == "DELETE"
    assert "--maintenance-policy" in cmd
    assert "gcloud storage cp --recursive" in plan.files["gpss-bootstrap.sh"]


def test_cloud_requires_bucket_and_known_machine():
    with pytest.raises(ValueError, match="bucket"):
        get_launcher("aws").plan(CFG, Resources("aws"))
    with pytest.raises(ValueError, match="unknown machine"):
        get_launcher("gcp").plan(CFG, res("gcp", machine="nope"))
    with pytest.raises(ValueError, match="unknown backend"):
        get_launcher("azure")


def test_estimate_scales():
    aws = get_launcher("aws")
    small = aws.estimate(CFG, res("aws"))
    big = aws.estimate(DaviConfig(updates=100), res("aws"))
    fast = aws.estimate(DaviConfig(updates=100), res("aws", machine="c7i.8xlarge"))
    assert big.train_hours == pytest.approx(2 * small.train_hours)
    assert fast.train_hours < big.train_hours
    capped = aws.estimate(DaviConfig(updates=100_000), res("aws", max_hours=3))
    assert capped.billed_hours == 3 and any("max_hours" in n for n in capped.notes)


def test_local_plan_runs_train():
    rec = Recorder()
    confirm_and_launch(get_launcher("local", rec), CFG, Resources("local"), execute=True, assume_yes=True,
                       show=lambda s: None)
    assert rec.calls[0][1:4] == ["-m", "gpss_train", "train"]


def test_bootstrap_quotes_bucket_and_stops_training_before_kill_switch():
    plan = get_launcher("aws", Recorder()).plan(CFG, Resources("aws", max_hours=2, bucket="s3://b/p; touch /pwned"))
    script = plan.files["gpss-bootstrap.sh"]
    assert "'s3://b/p; touch /pwned/cube2-davi'" in script
    # Kill switch at 120 min; training is stopped at 105 min so export + upload still run.
    assert "shutdown -h +120" in script and "timeout --signal=INT --kill-after=2m 105m" in script
    assert script.index("timeout") < script.index("gpss_train export") < script.index("aws s3 cp")


def test_launch_refuses_to_overwrite_existing_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "gpss-bootstrap.sh").write_text("mine")
    rec = Recorder()
    with pytest.raises(FileExistsError):
        confirm_and_launch(get_launcher("aws", rec), CFG, res("aws", budget_usd=100), execute=True,
                           assume_yes=True, show=lambda s: None)
    assert rec.calls == [] and (tmp_path / "gpss-bootstrap.sh").read_text() == "mine"
