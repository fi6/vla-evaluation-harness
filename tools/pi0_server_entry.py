#!/usr/bin/env python3
"""Docker entrypoint for the Jetson pi0 PyTorch server.

This wrapper is intentionally tiny: it only supplies a local version module
when the repo is bind-mounted without an installed wheel, then delegates to the
normal vla-eval server runner.
"""

from __future__ import annotations

import datetime
import sys
import types

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

version = types.ModuleType("vla_eval._version")
version.__version__ = "0+jetson-pi0"
version.__version_tuple__ = (0, "jetson-pi0")
sys.modules.setdefault("vla_eval._version", version)

from vla_eval.model_servers.pi0 import Pi0ModelServer
from vla_eval.model_servers.serve import run_server


if __name__ == "__main__":
    run_server(Pi0ModelServer)
