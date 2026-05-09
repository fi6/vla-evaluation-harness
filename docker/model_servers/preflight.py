# /// script
# requires-python = "~=3.11"
# dependencies = ["huggingface_hub>=0.23"]
# ///
"""Model weight preflight check and download for VLA model servers.

Checks for required HuggingFace repos in the local HF cache.
- If all repos are present: exits immediately (nothing to do).
- If any are missing: prints a clear message and downloads them with
  retry + built-in integrity verification.

Usage:
    uv run --python 3.11 docker/model_servers/preflight.py ORG/REPO [ORG/REPO ...]

Respects HF_ENDPOINT (e.g. https://hf-mirror.com) for mirror support.
Users may pre-populate ~/.cache/huggingface/hub/ manually — if the model
snapshot already exists there, no download is attempted.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [preflight] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 15.0  # seconds; multiplied by attempt number


def _hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _is_cached(repo_id: str) -> bool:
    """Return True if at least one complete snapshot of *repo_id* exists locally."""
    hub_dir = _hf_cache_root() / "hub"
    repo_dir = hub_dir / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    # A snapshot directory must be non-empty (incomplete downloads leave only .lock files).
    for snap in snapshots_dir.iterdir():
        if snap.is_dir() and any(f for f in snap.iterdir() if not f.name.endswith(".lock")):
            return True
    return False


def _download(repo_id: str) -> None:
    """Download *repo_id* to the HF cache with retry. Raises on final failure."""
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    log.info("Downloading  %s  via  %s", repo_id, endpoint)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            snapshot_download(repo_id)
            log.info("Download complete: %s", repo_id)
            return
        except Exception as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            delay = _RETRY_BASE_DELAY * attempt
            log.warning(
                "Attempt %d/%d failed (%s). Retrying in %.0fs ...",
                attempt,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {repo_id} after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}\n"
        f"Hint: set HF_ENDPOINT=https://hf-mirror.com to use a mirror, "
        f"or pre-download manually and place files in {_hf_cache_root() / 'hub'}."
    )


def main(repos: list[str]) -> None:
    if not repos:
        log.info("No repos specified — nothing to check.")
        return

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    log.info("HF_ENDPOINT = %s", endpoint)
    log.info("Cache root  = %s", _hf_cache_root())

    missing = [r for r in repos if not _is_cached(r)]
    present = [r for r in repos if r not in missing]

    for repo in present:
        log.info("✓ Found in cache: %s", repo)

    if not missing:
        log.info("All models present. Skipping download.")
        return

    log.info("")
    log.info("The following models are not in the local cache and will be downloaded:")
    for repo in missing:
        log.info("  - %s", repo)
    log.info("")

    for repo in missing:
        _download(repo)

    log.info("All models ready.")


if __name__ == "__main__":
    main(sys.argv[1:])
