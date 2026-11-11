"""Training launchers: run ``gpss_train train`` locally or on a managed cloud VM.

One interface (:class:`~.base.Launcher`) with three backends:

* ``local`` - a subprocess on this machine;
* ``aws``   - an EC2 instance via the ``aws`` CLI, artefacts to S3;
* ``gcp``   - a Compute Engine instance via the ``gcloud`` CLI, artefacts to GCS.

Every launch goes through the same three steps (see ``docs/training.md``):

1. **estimate** - wall time and cost from a throughput model of the chosen machine;
2. **plan**     - the exact commands that would run (printed; this is the dry run);
3. **confirm**  - nothing is executed without ``--execute`` *and* an interactive "yes"
   (or ``--yes``), and never if the estimate exceeds ``--budget-usd``.

Cloud VMs carry two independent kill switches: the bootstrap script schedules
``shutdown -h +<max_hours*60>`` before training starts, and the instance is created so
that shutdown terminates/deletes it (AWS ``InstanceInitiatedShutdownBehavior=terminate``,
GCP ``--max-run-duration`` + ``--instance-termination-action=DELETE``).

Credentials are never read or stored here: the provider CLIs use their own configuration
(``aws configure`` / ``gcloud auth login``). All commands go through an injectable
``runner`` so the backends are unit-tested without any cloud access.
"""

from .base import Estimate, LaunchPlan, Launcher, Resources, confirm_and_launch
from .registry import BACKENDS, get_launcher

__all__ = ["BACKENDS", "Estimate", "LaunchPlan", "Launcher", "Resources", "confirm_and_launch", "get_launcher"]
