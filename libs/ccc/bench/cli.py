"""``ccc-gpu-bench`` command-line interface (also ``python -m ccc.bench``).

Reproducible, structured performance measurement decoupled from the pytest
suite. Modes:

  coef     end-to-end GPU-vs-CPU coefficient benchmark (grid over
           features x samples x n_jobs), with ``--pvalue-n-perms`` and
           ``--return-parts`` variants.
  ari      kernel-level ARI GPU-vs-CPU micro-benchmark.
  scaling  CPU parallelism (n_jobs) scaling.

Output is JSON Lines (default) or CSV, written incrementally, one record per
case with full config + environment metadata + seed. Only stdlib + already
present deps are used.
"""

import argparse
import sys

from . import runners
from .env import capture_environment, gpu_available
from .output import open_writer
from .presets import get_preset


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--preset",
        choices=["smoke", "paper"],
        default=None,
        help="named grid; explicit grid args override individual axes",
    )
    sub.add_argument(
        "-o",
        "--output",
        default=None,
        help="output path (default: stdout). '-' also means stdout",
    )
    sub.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="output format (default: jsonl)",
    )
    sub.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    sub.add_argument(
        "--repeats", type=int, default=3, help="timed repeats per case (default: 3)"
    )
    sub.add_argument(
        "--warmup", type=int, default=1, help="untimed warmup calls (default: 1)"
    )
    sub.add_argument(
        "--profile",
        action="store_true",
        help="CPU category profile (partitioning/ARI/...) + nsys hint for GPU",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccc-gpu-bench",
        description="Structured GPU/CPU benchmarks for the CCC coefficient.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # coef ------------------------------------------------------------------ #
    coef = subparsers.add_parser("coef", help="end-to-end GPU vs CPU coefficient")
    _add_common(coef)
    coef.add_argument("--features", type=int, nargs="+", help="feature counts")
    coef.add_argument("--samples", type=int, nargs="+", help="sample (object) counts")
    coef.add_argument("--n-jobs", type=int, nargs="+", help="CPU worker counts")
    coef.add_argument("--pvalue-n-perms", type=int, default=None)
    coef.add_argument("--return-parts", action="store_true")
    coef.add_argument("--gpu-only", action="store_true")
    coef.add_argument("--cpu-only", action="store_true")

    # ari ------------------------------------------------------------------- #
    ari = subparsers.add_parser("ari", help="kernel-level ARI GPU vs CPU")
    _add_common(ari)
    ari.add_argument("--n-features", type=int, nargs="+")
    ari.add_argument("--n-parts", type=int, nargs="+")
    ari.add_argument("--n-objs", type=int, nargs="+")
    ari.add_argument("--k", type=int, nargs="+")
    ari.add_argument("--gpu-only", action="store_true")
    ari.add_argument("--cpu-only", action="store_true")

    # scaling --------------------------------------------------------------- #
    scaling = subparsers.add_parser("scaling", help="CPU n_jobs scaling")
    _add_common(scaling)
    scaling.add_argument("--features", type=int, nargs="+")
    scaling.add_argument("--samples", type=int, nargs="+")
    scaling.add_argument("--n-jobs", type=int, nargs="+")
    scaling.add_argument("--pvalue-n-perms", type=int, default=None)

    return parser


def _resolve_grid(mode: str, args: argparse.Namespace, keys: list[str]) -> dict:
    """Resolve grid axes: explicit args > preset > smoke defaults."""
    preset = get_preset(mode, args.preset) if args.preset else {}
    smoke = get_preset(mode, "smoke")
    resolved = {}
    for key in keys:
        attr = key.replace("-", "_")
        val = getattr(args, attr, None)
        if val:
            resolved[key] = val
        elif key in preset:
            resolved[key] = preset[key]
        else:
            resolved[key] = smoke[key]
    return resolved


def _require_gpu(cpu_only: bool) -> None:
    """Fail fast (clear message) when a GPU mode is asked for without a GPU."""
    if not cpu_only and not gpu_available():
        msg = (
            "GPU benchmark requested but cupy / ccc_cuda_ext are unavailable.\n"
            "Install the CUDA extension, or pass --cpu-only to run CPU-side only."
        )
        raise SystemExit(msg)


def _maybe_profile(mode: str, args: argparse.Namespace, grid: dict) -> None:
    if not args.profile:
        return
    import numpy as np

    from . import profiling

    if mode == "ari":
        nf = grid["n_features"][0]
        no = grid["n_objs"][0]
        print("\n" + profiling.nsys_suggestion(nf, no, args.seed), file=sys.stderr)
        return

    nf = grid["features"][0]
    ns = grid["samples"][0]
    cpu_only = getattr(args, "cpu_only", False)
    gpu_only = getattr(args, "gpu_only", False)

    if not gpu_only:
        from ccc.coef.impl import ccc as ccc_cpu

        np.random.seed(args.seed)
        data = np.random.rand(nf, ns)
        kwargs = {}
        if mode == "coef":
            kwargs = {
                "pvalue_n_perms": args.pvalue_n_perms,
                "return_parts": args.return_parts,
            }
        elif mode == "scaling":
            kwargs = {"pvalue_n_perms": args.pvalue_n_perms}
        prof = profiling.profile_cpu(ccc_cpu, data, n_jobs=1, **kwargs)
        profiling.print_cpu_profile(prof)

    if not cpu_only and gpu_available():
        print("\n" + profiling.nsys_suggestion(nf, ns, args.seed), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    mode = args.mode
    env = capture_environment()
    writer = open_writer(args.output, args.format)

    try:
        if mode == "coef":
            if args.gpu_only and args.cpu_only:
                raise SystemExit("--gpu-only and --cpu-only are mutually exclusive")
            grid = _resolve_grid("coef", args, ["features", "samples", "n_jobs"])
            _require_gpu(args.cpu_only)
            print(f"Running coef benchmark: {grid}", file=sys.stderr)
            runners.run_coef(
                writer=writer,
                env=env,
                features=grid["features"],
                samples=grid["samples"],
                n_jobs=grid["n_jobs"],
                pvalue_n_perms=args.pvalue_n_perms,
                return_parts=args.return_parts,
                seed=args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
                gpu_only=args.gpu_only,
                cpu_only=args.cpu_only,
            )
            _maybe_profile("coef", args, grid)

        elif mode == "ari":
            if args.gpu_only and args.cpu_only:
                raise SystemExit("--gpu-only and --cpu-only are mutually exclusive")
            grid = _resolve_grid("ari", args, ["n_features", "n_parts", "n_objs", "k"])
            _require_gpu(args.cpu_only)
            print(f"Running ari benchmark: {grid}", file=sys.stderr)
            runners.run_ari(
                writer=writer,
                env=env,
                n_features=grid["n_features"],
                n_parts=grid["n_parts"],
                n_objs=grid["n_objs"],
                k=grid["k"],
                seed=args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
                gpu_only=args.gpu_only,
                cpu_only=args.cpu_only,
            )
            _maybe_profile("ari", args, grid)

        elif mode == "scaling":
            grid = _resolve_grid("scaling", args, ["features", "samples", "n_jobs"])
            print(f"Running scaling benchmark: {grid}", file=sys.stderr)
            runners.run_scaling(
                writer=writer,
                env=env,
                features=grid["features"],
                samples=grid["samples"],
                n_jobs=grid["n_jobs"],
                pvalue_n_perms=args.pvalue_n_perms,
                seed=args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            _maybe_profile("scaling", args, grid)
    finally:
        writer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
