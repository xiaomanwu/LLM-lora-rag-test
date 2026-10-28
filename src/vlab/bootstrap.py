"""Network bootstrap — import and call this *before* transformers/datasets/peft.

``huggingface_hub`` caches its endpoint in a module-level constant at import time.
Any later ``os.environ["HF_ENDPOINT"] = ...`` is silently ignored, and every model
load then retries against an unreachable host for minutes. The only safe fix is to
set the endpoint before those packages are imported, which is what this
stdlib-only module does.

Usage, as the very first import of every entry point::

    import vlab.bootstrap as bootstrap
    bootstrap.setup_environment()
"""

from __future__ import annotations

import os
import sys
import urllib.request

OFFICIAL = "https://huggingface.co"
MIRRORS = ("https://hf-mirror.com",)
PROBE_PATH = "/api/models?limit=1"
PROBE_TIMEOUT = 3.0

_CONFIGURED = False


def _reachable(base: str) -> bool:
    request = urllib.request.Request(f"{base}{PROBE_PATH}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT):
            return True
    except Exception:  # noqa: BLE001 - any network error counts as unreachable
        return False


def _from_argv() -> str:
    """Honour an explicit ``--hf-endpoint`` flag before auto-detection kicks in."""
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--hf-endpoint" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--hf-endpoint="):
            return arg.split("=", 1)[1]
    return ""


def setup_environment() -> str:
    """Pick a reachable HF host and export it. Returns the endpoint used ("" = official)."""
    global _CONFIGURED
    if _CONFIGURED:
        return os.environ.get("HF_ENDPOINT", "")
    endpoint = (os.environ.get("HF_ENDPOINT", "") or _from_argv()).rstrip("/")
    if not endpoint and not _reachable(OFFICIAL):
        for mirror in MIRRORS:
            if _reachable(mirror):
                endpoint = mirror
                break
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    # Xet transfers fail with 401 on several networks; plain HTTPS is more portable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _CONFIGURED = True
    return endpoint
