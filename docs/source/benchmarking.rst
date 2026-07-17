Benchmarking
============

CCC-GPU ships a small benchmarking command, ``ccc-gpu-bench`` (also runnable as
``python -m ccc.bench``), for reproducible, structured performance measurement
decoupled from the pytest suite. It is installed with the package.

Modes
-----

``ccc-gpu-bench`` has three sub-commands:

``coef``
    End-to-end GPU-vs-CPU coefficient benchmark, sweeping a grid over feature
    counts, sample (object) counts, and CPU worker counts (``n_jobs``). Supports
    ``--pvalue-n-perms`` and ``--return-parts`` variants and ``--gpu-only`` /
    ``--cpu-only``.

``ari``
    Kernel-level Adjusted Rand Index GPU-vs-CPU micro-benchmark.

``scaling``
    CPU parallelism (``n_jobs``) scaling for the CPU implementation.

Quick start
-----------

.. code-block:: bash

    # Fast sanity sweep (finishes in seconds); prints JSON Lines to stdout
    ccc-gpu-bench coef --preset smoke

    # Kernel-level ARI smoke benchmark
    ccc-gpu-bench ari --preset smoke

    # CPU n_jobs scaling smoke benchmark
    ccc-gpu-bench scaling --preset smoke

Common options (all modes):

- ``--preset {smoke,paper}`` -- named grid; explicit grid flags override individual axes.
- ``-o, --output PATH`` -- output file (default: stdout; ``-`` also means stdout).
- ``--format {jsonl,csv}`` -- output format (default: ``jsonl``).
- ``--seed INT`` -- random seed (default: 42).
- ``--repeats INT`` -- timed repeats per case (default: 3).
- ``--warmup INT`` -- untimed warmup calls (default: 1).
- ``--profile`` -- add a per-category CPU profile (and an ``nsys`` hint for GPU runs).

Grid flags for ``coef`` / ``scaling``: ``--features``, ``--samples``,
``--n-jobs`` (each accepts multiple values). For ``ari``: ``--n-features``,
``--n-parts``, ``--n-objs``, ``--k``.

Output schema
-------------

Records are written incrementally (one per grid case), so an interrupted sweep
still leaves a parseable file. In ``jsonl`` format each line is one JSON object;
in ``csv`` format the header is taken from the first record and any nested value
is JSON-encoded.

Every record embeds environment metadata from :func:`ccc.bench.env.capture_environment`
(``package_version``, ``python_version``, ``platform``, ``cpu_model``,
``cpu_count``, ``gpu_present``, ``gpu_name``, ``gpu_driver``, ``cuda_runtime``)
so results are self-describing and comparable across machines and commits.

A ``coef`` record additionally contains:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``mode``
     - ``"coef"``
   * - ``timestamp``
     - ISO-8601 time the case finished
   * - ``n_features`` / ``n_samples`` / ``n_jobs``
     - grid point for this case
   * - ``pvalue_n_perms`` / ``return_parts``
     - variant flags (``null`` / ``false`` when unused)
   * - ``seed`` / ``repeats`` / ``warmup``
     - run configuration
   * - ``n_coefficients``
     - number of pairwise coefficients computed (``n*(n-1)/2``)
   * - ``gpu_time_min_s`` / ``gpu_time_mean_s``
     - GPU timing (min and mean over repeats; ``null`` with ``--cpu-only``)
   * - ``cpu_time_min_s`` / ``cpu_time_mean_s``
     - CPU timing (``null`` with ``--gpu-only``)
   * - ``speedup``
     - ``cpu_time_min_s / gpu_time_min_s`` (``null`` if a side is missing)

The ``ari`` and ``scaling`` records follow the same pattern with mode-specific
grid fields.

Reproducing the README performance table
-----------------------------------------

The speedup table in the README / :doc:`introduction` was produced with the
``paper`` preset of the ``coef`` mode (a feature sweep at 1000 fixed samples).
On the reference GPU box:

.. code-block:: bash

    ccc-gpu-bench coef --preset paper --format csv -o coef_paper.csv

The ``paper`` grid runs a large feature sweep and its CPU reference for the
biggest cases takes minutes per point, so run it on the dedicated GPU machine.
The ``speedup`` column of the resulting records corresponds to the
"CCC-GPU vs. CCC" column of the table.
