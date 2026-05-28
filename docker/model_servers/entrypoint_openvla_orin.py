#!/usr/bin/env python3
"""Jetson Orin entrypoint for the OpenVLA model server."""

from __future__ import annotations

import datetime
import sys
import types

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

version = types.ModuleType("vla_eval._version")
version.__version__ = "0+jetson-openvla"
version.__version_tuple__ = (0, "jetson-openvla")
sys.modules.setdefault("vla_eval._version", version)

from vla_eval.model_servers.openvla import OpenVLAModelServer
from vla_eval.model_servers.serve import run_server


if __name__ == "__main__":
    run_server(OpenVLAModelServer)
