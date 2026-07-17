"""CPU-only plumbing tests for the ``ccc-gpu-bench`` CLI.

These exercise argument parsing and structured (JSONL/CSV) output only. They do
NOT assert on timings and never require a GPU (they use the CPU-only ``scaling``
mode and ``coef --cpu-only``). Unmarked, so they run in the fast CI subset.
"""

import json

from ccc.bench import cli
from ccc.bench.env import capture_environment
from ccc.bench.presets import get_preset


def test_parser_builds_and_parses_coef():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["coef", "--features", "10", "20", "--samples", "50", "--cpu-only"]
    )
    assert args.mode == "coef"
    assert args.features == [10, 20]
    assert args.samples == [50]
    assert args.cpu_only is True
    assert args.format == "jsonl"


def test_capture_environment_has_expected_keys():
    env = capture_environment()
    for key in (
        "package_version",
        "python_version",
        "platform",
        "cpu_count",
        "gpu_present",
        "gpu_name",
    ):
        assert key in env


def test_presets_available():
    smoke = get_preset("coef", "smoke")
    assert "features" in smoke and "samples" in smoke and "n_jobs" in smoke


def test_scaling_writes_valid_jsonl(tmp_path):
    out = tmp_path / "scaling.jsonl"
    rc = cli.main(
        [
            "scaling",
            "--features",
            "10",
            "--samples",
            "50",
            "--n-jobs",
            "1",
            "2",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2  # one record per n_jobs
    records = [json.loads(line) for line in lines]
    for rec in records:
        assert rec["mode"] == "scaling"
        assert rec["n_features"] == 10
        assert rec["n_samples"] == 50
        assert rec["cpu_time_min_s"] > 0
        assert "package_version" in rec  # env metadata embedded
        assert rec["seed"] == 42


def test_coef_cpu_only_writes_valid_csv(tmp_path):
    out = tmp_path / "coef.csv"
    rc = cli.main(
        [
            "coef",
            "--features",
            "10",
            "--samples",
            "50",
            "--n-jobs",
            "1",
            "--cpu-only",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--format",
            "csv",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text().strip().splitlines()
    assert len(text) == 2  # header + 1 data row
    assert "mode" in text[0]
    assert "n_coefficients" in text[0]
