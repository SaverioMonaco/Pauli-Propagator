"""Fail native CI rather than silently skip tests against an older extension."""
import os

import pytest


def pytest_sessionstart(session):
    if os.environ.get("PPROP_REQUIRE_NATIVE") != "1":
        return
    try:
        import pprop_rs
    except ImportError as exc:
        raise pytest.UsageError("Build the native extension before running native CI") from exc
    required = ("Evaluator", "ragged_layout")
    missing = [name for name in required if not hasattr(pprop_rs, name)]
    if missing:
        raise pytest.UsageError("Rebuild pprop_rs; missing native APIs: " + ", ".join(missing))
