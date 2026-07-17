from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the installed distribution's version, which comes
    # from `[project].version` in pyproject.toml.
    __version__ = version("cccgpu")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    # Fallback for running from a source checkout that was never installed.
    __version__ = "0.2.4"
