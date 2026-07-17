"""Environment capture for benchmark records.

Collects GPU / driver / CUDA / CPU / package metadata so every benchmark record
is self-describing and comparable across machines and commits. Everything here
degrades gracefully: missing GPU or tools yield ``None`` fields rather than
raising.
"""

import importlib.util
import platform
import subprocess


def gpu_available() -> bool:
    """True when both cupy and the compiled CUDA extension are importable."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("cupy", "ccc_cuda_ext")
    )


def _package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("cccgpu")
    except Exception:
        try:
            import ccc

            return getattr(ccc, "__version__", None)
        except Exception:
            return None


def _cpu_model() -> str | None:
    """Best-effort CPU model string (Linux /proc/cpuinfo, else platform)."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _nvidia_smi_field(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _gpu_info() -> dict:
    """GPU name / driver / CUDA runtime via cupy, falling back to nvidia-smi."""
    info = {"gpu_name": None, "gpu_driver": None, "cuda_runtime": None}
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy as cp

            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name")
            info["gpu_name"] = name.decode() if isinstance(name, bytes) else name
            runtime = cp.cuda.runtime.runtimeGetVersion()
            major, minor = divmod(runtime, 1000)
            info["cuda_runtime"] = f"{major}.{minor // 10}"
        except Exception:
            pass
    if info["gpu_name"] is None:
        info["gpu_name"] = _nvidia_smi_field("name")
    if info["gpu_driver"] is None:
        info["gpu_driver"] = _nvidia_smi_field("driver_version")
    return info


def capture_environment() -> dict:
    """Return a metadata dict embedded in every benchmark record."""
    import os

    env = {
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu_present": gpu_available(),
    }
    env.update(_gpu_info())
    return env
