#!/usr/bin/env python3
"""Send a short synthetic episode to any vla-eval model server."""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import statistics
import time
import sys
import types
from pathlib import Path
from typing import Any

if not hasattr(_dt, "UTC"):
    _dt.UTC = _dt.timezone.utc

version = types.ModuleType("vla_eval._version")
version.__version__ = "0+latency-smoke"
version.__version_tuple__ = (0, "latency-smoke")
sys.modules.setdefault("vla_eval._version", version)

import numpy as np
import websockets

from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p90_ms": round(_percentile(values, 0.90), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


async def run(url: str, steps: int, image_size: int, state_dim: int, task: str) -> dict[str, Any]:
    roundtrip_ms: list[float] = []
    action_shapes: list[list[int]] = []
    async with websockets.connect(url, max_size=None, compression=None, ping_interval=None) as ws:
        await ws.send(pack_message(Message(MessageType.HELLO, make_hello_payload(client="latency-smoke"), seq=1)))
        hello = unpack_message(await ws.recv())

        await ws.send(pack_message(Message(MessageType.EPISODE_START, {"task": task}, seq=2)))
        obs = {
            "images": {
                "base": np.zeros((image_size, image_size, 3), dtype=np.uint8),
                "wrist": np.zeros((image_size, image_size, 3), dtype=np.uint8),
            },
            "states": np.zeros(state_dim, dtype=np.float64),
            "task_description": task,
        }
        for i in range(steps):
            start = time.perf_counter()
            await ws.send(pack_message(Message(MessageType.OBSERVATION, obs, seq=10 + i)))
            reply = unpack_message(await ws.recv())
            roundtrip_ms.append((time.perf_counter() - start) * 1000)
            action_shapes.append(list(np.asarray(reply.payload["actions"]).shape))
            print(f"step={i} type={reply.type.value} roundtrip_ms={roundtrip_ms[-1]:.3f}")
        await ws.send(pack_message(Message(MessageType.EPISODE_END, {"success": True}, seq=99)))

    return {
        "url": url,
        "steps": steps,
        "image_size": image_size,
        "state_dim": state_dim,
        "task": task,
        "server_hello": hello.payload,
        "roundtrip_ms": _summary(roundtrip_ms),
        "action_shapes": action_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = asyncio.run(run(args.url, args.steps, args.image_size, args.state_dim, args.task))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
