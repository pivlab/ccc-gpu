"""Environment-aware configuration for the GPU test suite.

Every test under ``tests/gpu/`` is automatically tagged with the ``gpu`` marker
so it can be selected/deselected with ``-m gpu`` / ``-m "not gpu"``.

On a machine without ``cupy`` or the compiled ``ccc_cuda_ext`` extension the GPU
modules cannot even be imported (they import cupy at module scope), so we tell
pytest to ignore them during collection. This keeps a plain ``pytest tests/``
green on a CPU-only box instead of erroring out at import time. On a GPU box the
modules collect and run normally.
"""

import importlib.util
from pathlib import Path

import pytest

_HERE = Path(__file__).parent

# The GPU suite needs both cupy (for memory management / array transfer) and the
# compiled CUDA extension. If either is missing the tests are not collected.
_MISSING_GPU_DEPS = [
    name for name in ("cupy", "ccc_cuda_ext") if importlib.util.find_spec(name) is None
]
GPU_AVAILABLE = not _MISSING_GPU_DEPS

# When GPU deps are unavailable, skip collecting the GPU test modules entirely so
# their module-level ``import cupy`` does not raise a collection error.
collect_ignore_glob = ["test_*.py"] if not GPU_AVAILABLE else []


def pytest_collection_modifyitems(items):
    """Auto-apply the ``gpu`` marker to every test collected under tests/gpu/."""
    for item in items:
        item_path = Path(str(item.fspath))
        if item_path.parent == _HERE or _HERE in item_path.parents:
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True)
def clean_gpu_memory():
    """Free the cupy default memory pool after each GPU test.

    Replaces the copy-pasted ``@clean_gpu_memory`` decorator that used to wrap
    individual GPU tests. Applied automatically to every test in this directory.
    """
    yield
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
