Optimization Backlog
====================

This page is the authoritative, evidence-ranked backlog of GPU/CPU performance
optimizations for CCC-GPU. It exists so a future contributor can pick up a single
item and scope an implementation change from it *without re-deriving the analysis*:
every entry cites the code it targets (``file:line``), gives an estimated
impact/effort/risk, records whether it is **inspection-justified** or **needs
profiling**, and is assigned to an implementation **tranche**.

.. important::

   This is a *documentation-only* backlog. It makes **zero** changes to production
   code. Each item becomes its own OpenSpec change when implementation begins, and
   each such change is gated on the baseline numbers recorded here (and, for the
   kernel restructures, on golden-parity tests against the CPU reference).

.. contents:: On this page
   :local:
   :depth: 2

Scope and provenance
--------------------

The opportunities below come from the 2026-07-16 architecture review of the two CUDA
translation units (``libs/ccc_cuda_ext/coef.cu`` and ``libs/ccc_cuda_ext/metrics.cu``)
and the CPU partitioning path (``libs/ccc/coef/impl.py`` /
``libs/ccc/coef/impl_gpu.py``). They are grouped by layer:

- **K** — kernel-level (device code: launch shape, memory access, atomics).
- **H** — host-side orchestration (uploads, launches, allocations, synchronization).
- **A** — algorithmic (batch sizing, skipping invalid work).

Line numbers were re-verified against the current tree (they shifted after the
``add-lint-tooling`` reformat and the ``fix-cuda-correctness`` edits). The commit and
hardware the baseline was measured on are recorded in
:ref:`baseline-measurements`.

Legend
~~~~~~

- **Impact** — expected end-to-end speedup contribution: High / Medium / Low.
- **Effort** — implementation size/complexity: Low / Medium / High.
- **Risk** — chance of changing results or destabilizing the kernels: Low / Medium / High.
- **Status** — *inspection-justified* (pathological by reading the code; safe to
  schedule now) or *needs profiling* (rank cannot be confirmed without nsys/ncu
  counters that this environment cannot capture — see
  :ref:`profiling-not-captured`).
- **Tranche** — recommended scheduling group (see :ref:`tranches`).

Relationship to ``fix-cuda-correctness``
---------------------------------------

The merged ``fix-cuda-correctness`` change touched the p-value path for
*correctness*, and in doing so **partially** addressed two items on this backlog.
It did **not** close the performance gap; the remaining headroom is tracked here:

- **Monolithic permutation buffer (part of H2).** The old code allocated a single
  ``n_feature_comp × n_perms`` device buffer for permuted CCC values, which OOMs on
  large inputs. It is now chunked into a memory-bounded buffer
  (``d_perm_ccc_values`` sized to ``chunk_comps × n_perms``,
  ``coef.cu:806``) with per-chunk p-value reduction, and the per-thread contingency
  scratch is now sub-batched to a memory budget (``perm_batch``, ``coef.cu:758-766``).
  **Remaining headroom:** the host still issues **one kernel launch per feature
  comparison** in a serial loop, each followed by a full device synchronization (see
  H2 below) — the memory fix did not remove the launch/sync serialization.

- **Redundant device upload (part of H1/H3).** The p-value path uploads the whole
  ``parts`` array to the device a second time (``coef.cu:742``) in addition to the
  upload the coefficient path already performs inside ``ari_core_device``
  (``metrics.cu:593``). This duplication remains.

.. _baseline-measurements:

Baseline measurements (this environment)
---------------------------------------

Captured with the ``ccc-gpu-bench`` CLI (shipped by the ``extract-benchmarks``
change). Raw JSONL artifacts are intentionally kept **out of git**; only the
summaries below are committed.

Environment
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - Commit measured
     - ``810970c`` (branch ``pr-f-optimization-backlog``)
   * - GPU
     - NVIDIA GeForce RTX 4090 (sm_89, 24 GB)
   * - GPU driver
     - 580.159.03
   * - CUDA runtime (cupy)
     - 12.9 (toolkit 12.5)
   * - CPU
     - AMD Ryzen Threadripper 7960X (24 cores / 48 threads)
   * - OS / Python
     - Linux 7.0.11 / CPython 3.12.13
   * - Package version
     - ccc-gpu 0.2.4
   * - Bench settings
     - ``--repeats`` as noted, ``--warmup 1`` (PTX JIT + allocator warmup absorbed)

GPU coefficient-only scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ccc-gpu-bench coef --features 1000 2000 5000 10000 --samples 1000 --gpu-only
--repeats 3``. Throughput = ``n_coefficients / gpu_time_min``.

.. list-table::
   :header-rows: 1
   :widths: 14 12 18 18 20 18

   * - Features
     - Samples
     - Coefficients
     - GPU min (s)
     - GPU throughput (coef/s)
     - Internal batches [1]_
   * - 1000
     - 1000
     - 499,500
     - 0.671
     - 0.74 M
     - 1
   * - 2000
     - 1000
     - 1,999,000
     - 1.762
     - 1.13 M
     - 1
   * - 5000
     - 1000
     - 12,497,500
     - 7.851
     - 1.59 M
     - 1
   * - 10000
     - 1000
     - 49,995,000
     - 28.107
     - 1.78 M
     - 4

.. [1] Number of ARI batches the coefficient path splits the work into. The batch
   size is a **hardcoded** ``batch_n_features = 5000`` (``coef.cu:586``), so inputs of
   5000 features or fewer run in a single batch; 10000 features run in four. Each
   batch re-uploads the *entire* ``parts`` array (``metrics.cu:593``) — see H1.

**Reading of this data.** Throughput *rises* monotonically with problem size
(0.74 M → 1.78 M coef/s). This is direct evidence that at small sizes the workload is
**overhead-bound** (fixed per-call host orchestration, uploads, and kernel-launch
latency dominate — the target of the tranche-1 host-side items), while at large sizes
it approaches **compute-bound** behavior (the target of the tranche-2 kernel
restructures). Note that 10000 features runs in four batches (four full ``parts``
re-uploads) yet throughput still improves, so the redundant upload is **not** the
dominant cost at ``n_objs = 1000``; its relative cost grows with the number of objects
and the number of batches, and must be profiled to rank precisely.

GPU vs CPU (single-threaded reference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ccc-gpu-bench coef --features 500 1000 --samples 1000 --repeats 2``. The CPU
reference here runs with ``n_jobs = 1`` (the bench default), i.e. **single-threaded**;
the 48-thread machine would narrow the gap substantially, so treat these speedups as
an *upper bound* against an unparallelized CPU baseline, not a like-for-like core
comparison.

.. list-table::
   :header-rows: 1
   :widths: 14 12 18 18 18 18

   * - Features
     - Samples
     - GPU min (s)
     - CPU min (s, 1 thread)
     - Speedup
     - Coefficients
   * - 500
     - 1000
     - 0.296
     - 35.94
     - 121x
     - 124,750
   * - 1000
     - 1000
     - 0.674
     - 142.06
     - 211x
     - 499,500

P-value path
~~~~~~~~~~~~

``ccc-gpu-bench coef --features 1000 --samples 1000 --pvalue-n-perms 100 --gpu-only
--repeats 2`` and a smaller CPU-comparison case at 200 features.

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 18 18 22

   * - Features
     - Samples
     - Perms
     - GPU min (s)
     - CPU min (s, 1 thread)
     - Speedup
   * - 50
     - 1000
     - 100
     - 18.80
     - 37.43
     - 2.0×
   * - 200
     - 1000
     - 100
     - > 300 (timed out)
     - n/a (GPU-only)
     - n/a
   * - 200
     - 1000
     - 0 (coef-only)
     - 0.11
     - n/a
     - n/a

At 200 features the same input takes **0.11 s** without p-values but **exceeds
300 s** (the measurement timeout) with only 100 permutations — a **> 2,700×**
blow-up — and even the tiny 50-feature case spends **18.8 s** on just 1,225
comparisons. This cost lives almost entirely in the **serial per-comparison
launch loop** (H2), not in raw arithmetic: the null distribution is
``n_feature_comp × n_perms`` small ARI computations, but they are issued as one
kernel launch per comparison with a full device sync each (the
``computePermutationCCC`` launch at ``coef.cu:865`` is guarded by
``CUDA_CHECK_KERNEL`` at ``coef.cu:869``, which calls ``cudaDeviceSynchronize`` —
``utils.cuh:112``). Wall time therefore scales with the *number of comparisons*
rather than with useful work, which is the single strongest argument for
tranche-1 item **H2** (fold the comparison index into the grid / batch the
launches).

CPU category profile (partitioning vs ARI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ccc-gpu-bench coef --cpu-only --profile --features 200 --samples 1000``. This
attributes CPU time to categories via ``cProfile`` (see
``libs/ccc/bench/profiling.py``). It matters for the GPU path too, because the GPU
path still computes all partitions **on the CPU** (``impl_gpu.py:626`` calls
``get_feature_parts`` before ``ccc_cuda_ext.compute_coef`` at ``impl_gpu.py:636``).

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Category
     - Time (s)
     - Percent
     - Notes
   * - ARI
     - 5.57
     - 83.1%
     - the coefficient inner loop — dominant, and why the kernel work moved to the GPU
   * - Coordination
     - 0.87
     - 12.9%
     - executor / chunking overhead
   * - NumPy / Numba
     - 0.17
     - 2.5%
     - array plumbing
   * - Partitioning
     - 0.08
     - 1.2%
     - quantile clustering (``get_parts``)
   * - Other
     - 0.02
     - 0.2%
     - remaining CCC + misc (total 6.69 s at 200 features, 1000 samples, 1 thread)

.. _profiling-not-captured:

What could NOT be captured, and why
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel-level profiling — the counter data needed to *confirm* the ranks of the
"needs profiling" items — could **not** be captured in this environment:

- **Nsight Systems (``nsys``) is not installed** (no system or conda-env binary).
  Timeline share of H2D/D2H vs kernels vs launch-gap could not be measured.
- **Nsight Compute (``ncu``) is installed** (version 2024.2.1.0 in the
  ``ccc-gpu-dev`` env) **but is blocked by driver policy.** The kernel
  ``RmProfilingAdminOnly`` flag is ``1`` (confirmed via
  ``/proc/driver/nvidia/params``), so a non-root user cannot read GPU performance
  counters. Any ``ncu`` run fails with::

     ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
     NVIDIA GPU Performance Counters on the target device 0.

  Fixing this requires either running as root or setting
  ``NVreg_RestrictProfilingToAdminUsers=0`` in the ``nvidia`` kernel module and
  rebooting (see https://developer.nvidia.com/ERR_NVGPUCTRPERM).

- **The categorical-heavy workload for A2 was not measured**, and cannot be with
  the CLI as shipped: ``ccc-gpu-bench coef`` generates continuous random data and
  exposes no ``--n-parts`` / categorical-fraction knob, so it cannot hold
  ``n_parts`` fixed or vary the invalid-partition (categorical/singleton) fraction
  that A2's payoff depends on. Ranking A2 therefore requires **both** a small
  benchmark enhancement (a categorical/mixed data generator with a controllable
  invalid-partition fraction) **and** the kernel counters above. Until then A2
  stays in *needs profiling*.

**No kernel-level numbers are fabricated in this document.** Every profile-dependent
item stays flagged *needs profiling* with the exact command to run once tools and
permissions are available (see :ref:`profiling-procedure`). The wall-clock
baselines above (coefficient scaling, GPU-vs-CPU speedups, the CPU category split,
and the p-value blow-up) are captured and confirm the *inspection-justified*
tranche-1 items; they do not substitute for the counter data the tranche-2 and
"needs profiling" items still require.

.. _profiling-procedure:

Profiling procedure (run when tools/permissions are available)
--------------------------------------------------------------

Reproducible workloads
~~~~~~~~~~~~~~~~~~~~~~~

Drive everything through ``ccc-gpu-bench`` so numbers are comparable and carry
environment metadata. Recommended baseline grid:

.. code-block:: bash

   # (a) coefficient-only scaling
   ccc-gpu-bench coef --features 1000 2000 5000 10000 --samples 1000 \
       --gpu-only --repeats 3 --format jsonl -o coef_scaling.jsonl

   # (b) p-value path (exercises the serial per-comparison launch loop)
   ccc-gpu-bench coef --features 1000 --samples 1000 --pvalue-n-perms 100 \
       --gpu-only --repeats 3 --format jsonl -o coef_pvalue.jsonl

   # (c) large representative workload from the review (needs a big-memory GPU)
   ccc-gpu-bench coef --features 20000 --samples 1000 --gpu-only \
       --repeats 3 --format jsonl -o coef_20k.jsonl

   # (d) CPU category split (partitioning vs ARI) — also informs H4
   ccc-gpu-bench coef --cpu-only --profile --features 1000 --samples 1000

Nsight Systems (timeline / launch-latency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bench prints the exact command on a GPU ``--profile`` run
(``libs/ccc/bench/profiling.py:nsys_suggestion``). Equivalent invocation:

.. code-block:: bash

   nsys profile -o ccc_gpu_f20000_n1000 \
     python -c "import numpy as np; from ccc.coef.impl_gpu import ccc; \
       np.random.seed(42); ccc(np.random.rand(20000, 1000))"

   # p-value variant
   nsys profile -o ccc_gpu_pvalue \
     python -c "import numpy as np; from ccc.coef.impl_gpu import ccc; \
       np.random.seed(42); ccc(np.random.rand(1000, 1000), pvalue_n_perms=100)"

**Metrics to record from the timeline:**

- Wall-clock share of H2D copies, D2H copies, kernels, and inter-kernel *gaps*
  (gaps = host launch latency + per-launch ``cudaDeviceSynchronize``). Confirms H1,
  H2, K8.
- Per-kernel total time and call count for ``ari_kernel``, ``findMaxAriKernel``,
  and ``computePermutationCCC``. Confirms K1/K2 (``d_aris`` traffic) and K6.
- Number of ``computePermutationCCC`` launches vs the number of feature comparisons.
  Directly quantifies H2.

Nsight Compute (kernel counters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Full section set on the hottest kernel; --launch-count keeps it quick.
   ncu --set full --launch-count 20 -k ari_kernel -o ari_kernel_report \
     python -c "import numpy as np; from ccc.coef.impl_gpu import ccc; \
       np.random.seed(42); ccc(np.random.rand(2000, 1000))"

**Metrics to record per kernel, and which backlog item each confirms:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric (ncu section / counter)
     - What it tells you
     - Confirms
   * - ``sm__throughput`` / ``gpu__compute_memory_throughput`` (roofline)
     - compute- vs memory-bound
     - K1, K7
   * - ``l1tex__data_bank_conflicts`` and shared-atomic replay
     - contingency-histogram atomic serialization
     - K3
   * - ``dram__bytes`` / ``sm__sass_data_bytes`` (achieved vs peak BW)
     - int16 load efficiency vs vectorized loads
     - K7
   * - ``sm__warps_active`` / achieved occupancy
     - block-size / shared-mem occupancy limits
     - K1/K2 sizing
   * - Global-memory traffic per batch
     - size of the ``d_aris`` intermediate (~8 GB/batch estimated)
     - K1/K2

Archive raw ``.nsys-rep`` / ``.ncu-rep`` files **outside** the repository and record
their path + a hash here; commit only the summarized tables.

.. _tranches:

Ranked backlog
--------------

Tranche 1 — schedule now (high impact, low/medium effort, inspection-justified)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are pathological by inspection, do not change numerical results, and do not
require kernel restructuring. They can be scheduled without waiting for the profiler.

.. list-table::
   :header-rows: 1
   :widths: 8 30 10 10 10 22

   * - ID
     - Opportunity
     - Impact
     - Effort
     - Risk
     - Code reference
   * - H1
     - Hoist the per-batch full ``parts`` H2D upload; pass ``k`` from Python instead of
       recomputing it on-device each batch
     - Medium
     - Low
     - Low
     - ``metrics.cu:593`` (upload), ``metrics.cu:603`` (``thrust::reduce`` for ``k``),
       ``coef.cu:742`` / ``coef.cu:748`` (duplicate upload + reduce in p-value path)
   * - H2
     - Remove the serial per-comparison launch + ``cudaDeviceSynchronize`` in the
       p-value loop (batch comparisons into one launch, or use CUDA streams)
     - High
     - Medium
     - Low
     - ``coef.cu:811-871`` (host loop), ``coef.cu:865`` (launch), ``coef.cu:869`` +
       ``utils.cuh:112`` (per-launch sync)
   * - H3
     - Reuse/pool device allocations instead of re-allocating ``d_parts``, ``d_out``,
       and the global contingency scratch on every call/batch
     - Medium
     - Low
     - Low
     - ``metrics.cu:593-594`` (per-call ``d_parts``/``d_out``), ``metrics.cu:648``
       (per-call global scratch)
   * - A3
     - Replace the hardcoded ``batch_n_features = 5000`` with memory-aware batch sizing
       derived from ``cudaMemGetInfo`` and the actual per-ARI footprint
     - Medium
     - Low
     - Low
     - ``coef.cu:586-589``

**H1 — redundant H2D upload + per-batch ``k`` reduction.** Every batch,
``ari_core_device`` rebuilds a device vector from the *entire* ``parts`` array
(``metrics.cu:593``) and re-runs a full ``thrust::reduce`` to find the max cluster id
``k`` (``metrics.cu:603``), even though ``parts`` is unchanged across batches and ``k``
is already known in Python (it is ``max(range_n_clusters)``). The p-value path uploads
``parts`` a *second* time (``coef.cu:742``) and reduces ``k`` again (``coef.cu:748``).
Fix: upload ``parts`` once, reuse the device buffer across the coefficient and p-value
phases, and pass ``k`` down as a parameter. Inspection-justified; the *magnitude* at
large ``n_objs`` / many batches should be confirmed with the nsys timeline.

**H2 — p-value launch/sync serialization.** The p-value section loops over every
feature comparison on the host (``coef.cu:815``), and for each one launches
``computePermutationCCC`` (``coef.cu:865``) followed by a full device synchronization
(``CUDA_CHECK_KERNEL`` → ``cudaDeviceSynchronize``). For 1000 features that is ~500k
serialized launch+sync round-trips. The ``fix-cuda-correctness`` change made the buffer
memory-bounded but left this serialization in place. Fix: process many comparisons per
launch (one grid over comparisons × permutations) and/or overlap independent
comparisons on multiple streams, syncing once per chunk rather than per comparison.
This is the single largest headroom item for the p-value path.

**H3 — allocation churn.** ``d_parts`` and ``d_out`` are freshly allocated with
``std::make_unique<thrust::device_vector<...>>`` on every ``ari_core_device`` call
(``metrics.cu:593-594``), and the global-memory kernel allocates its scratch per call
(``metrics.cu:648``). The batch loop already reuses ``d_cm_values`` / ``d_max_parts``
(``coef.cu:647-648``) — extend the same reuse to the per-batch buffers, or adopt a
caching device allocator. Pairs naturally with H1.

**A3 — memory-aware batch sizing.** ``batch_n_features`` is a magic ``5000``
(``coef.cu:586``). On a 24 GB RTX 4090 this leaves memory idle for small ``n_objs`` and
risks being wrong for large ``n_objs``; it also fixes the batch count regardless of the
device. Derive the batch size from free memory (``cudaMemGetInfo``) and the measured
per-ARI footprint so the code adapts to both the GPU and the workload.

Tranche 2 — kernel restructures (gate on golden-parity tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These change kernel structure and therefore the floating-point reduction order, so each
must land behind a golden-parity test that checks GPU output against the CPU reference
within tolerance before and after.

.. list-table::
   :header-rows: 1
   :widths: 8 30 10 10 10 22

   * - ID
     - Opportunity
     - Impact
     - Effort
     - Risk
     - Code reference
   * - K1+K2
     - Fuse ARI computation and the argmax reduction into one block-per-feature-pair
       kernel: drop the ``d_aris`` intermediate (~8 GB traffic/batch, estimated) and the
       separate ``findMaxAriKernel``; cache partition rows in shared memory
     - High
     - High
     - Medium
     - ``coef.cu:678`` (``d_aris`` intermediate), ``metrics.cu:392`` (``ari_kernel``),
       ``coef.cu:45`` + ``coef.cu:687`` (``findMaxAriKernel``)
   * - K6
     - Rewrite ``computePermutationCCC`` block-parallel, removing the duplicate
       per-thread ARI implementation and its global/local scratch spills
     - High
     - High
     - Medium
     - ``coef.cu:345`` (kernel, thread-per-permutation), ``coef.cu:236``
       (``computePermutedARI`` duplicate ARI impl)

**K1+K2 — fused ARI + argmax.** Today ``ari_kernel`` writes every partition-pair ARI
to a large global ``d_aris`` buffer (``metrics.cu:498`` → ``coef.cu:678``), then
``findMaxAriKernel`` (``coef.cu:45``, launched at ``coef.cu:687``) reads it all back to
take the per-comparison max. For a feature pair with ``n_parts^2`` partition pairs this
is a full round-trip through global memory (~8 GB/batch by the review's estimate). A
block-per-feature-pair kernel that computes all ``n_parts^2`` ARIs and reduces to the
max in registers/shared memory eliminates both the intermediate buffer and the second
kernel. Shared-memory caching of the partition rows (K2) additionally cuts redundant
global loads. Gate on parity; the argmax tie-breaking currently relies on CUB's behavior
(``coef.cu:134``) and must be preserved.

**K6 — permutation kernel rewrite.** ``computePermutationCCC`` (``coef.cu:345``) is
thread-per-permutation: each thread walks all ``n_parts^2`` partition pairs and calls
``computePermutedARI`` (``coef.cu:236``), a *second, independent* ARI implementation
that builds the contingency matrix serially in per-thread global scratch
(``coef.cu:359``). This duplicates ``ari_kernel``'s logic (a maintenance hazard — the
two must be kept numerically consistent, which is exactly what
``fix-cuda-correctness`` had to repair) and spills to global memory. A block-parallel
rewrite that reuses the *same* contingency/ARI device code as the observed statistic
would remove the duplication and the scratch traffic. High effort, high payoff for the
p-value path, and must be co-designed with H2.

Needs profiling — rank before committing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plausible wins whose *rank* depends on counters this environment cannot read. Run the
:ref:`profiling-procedure` first; promote to a tranche once the numbers justify it.

.. list-table::
   :header-rows: 1
   :widths: 8 34 10 10 10 18

   * - ID
     - Opportunity
     - Impact
     - Effort
     - Risk
     - Code reference
   * - K3
     - Privatized / warp-aggregated contingency histograms to cut shared-memory atomic
       serialization
     - ?
     - Medium
     - Medium
     - ``metrics.cu:143`` (shared ``atomicAdd``), ``metrics.cu:184`` (global)
   * - K7
     - Vectorized ``int16`` loads (e.g. ``int4``/``short2``) for the partition reads
     - ?
     - Medium
     - Medium
     - ``metrics.cu:439`` (``Todo: Use int4*?``), ``metrics.cu:134-135`` (element loads)
   * - K8
     - Pinned host memory + CUDA streams to overlap H2D/compute/D2H
     - ?
     - Medium
     - Medium
     - synchronous ``thrust`` copies at ``metrics.cu:593`` and ``coef.cu:706-710``
   * - A2
     - Skip invalid partition pairs on the host so no block is launched for them (instead
       of classifying inside every kernel block)
     - ?
     - Medium
     - Medium
     - ``metrics.cu:450`` and ``coef.cu:373`` (in-kernel ``classify_partition_pair``)
   * - K-misc
     - Warp-level reductions in ``get_pair_confusion_matrix`` (currently serial on
       ``tid == 0``); avoid per-batch ``cudaMemGetInfo`` in hot paths
     - ?
     - Low
     - Low
     - ``metrics.cu:227-264`` (serial reduction), ``coef.cu:671`` / ``coef.cu:725``
   * - H4
     - CPU partitioning: hoist the per-``k`` ``argsort``/``rank`` out of the cluster loop
       (both are independent of ``k``); make parallel partitioning the default
     - ?
     - Low
     - Low
     - ``impl_gpu.py:55-56`` / ``impl.py:56-57`` (argsort/rank), ``impl_gpu.py:139-141``
       (per-``k`` loop)

**K3** — all 128 threads in a block ``atomicAdd`` into the same ``k×k`` shared
contingency matrix (``metrics.cu:143``); with skewed cluster distributions this
serializes on a few bins. Privatized sub-histograms or warp-aggregated atomics may help,
but the win depends on the measured replay rate — *needs profiling*.

**K7** — partition elements are read one ``int16`` at a time
(``metrics.cu:134-135``); the code even carries a ``Todo: Use int4*?`` note
(``metrics.cu:439``). Whether vectorized loads help depends on whether ``ari_kernel`` is
memory-bound, which requires a roofline measurement.

**K8** — every H2D/D2H is a synchronous ``thrust`` copy with pageable host memory. Pinned
buffers + streams could overlap transfer and compute, but the benefit depends on the
timeline transfer share (*needs profiling*; the coefficient-scaling data above suggests
transfer is a minority of time at ``n_objs = 1000``, so this may rank low there).

**A2** — the kernels launch a block for every partition pair and classify validity inside
the block (``metrics.cu:450``); skipping invalid pairs on the host avoids launching those
blocks, but the payoff scales with the fraction of invalid pairs in real data, which must
be measured.

**H4** — inside ``run_quantile_clustering`` the ``argsort`` and ``rank`` of the feature
data (``impl_gpu.py:55-56``) are recomputed for every ``k`` in the cluster-range loop
(``impl_gpu.py:139-141``), though both are independent of ``k``. Because the GPU path
partitions on the CPU, this shows up in the GPU end-to-end time; the CPU category profile
above quantifies partitioning's share.

How to use this backlog
-----------------------

1. Pick an item. Start with **tranche 1** (host-side, low-risk, no parity gate needed).
2. Open an OpenSpec change scoped to that single item, citing this page and the code
   references.
3. For **tranche 2** and any promoted "needs profiling" item, first run the
   :ref:`profiling-procedure`, add a golden-parity test, then implement.
4. Update the **Status**/tranche of the item here (and re-measure the baseline) as part
   of that change, so this document tracks reality. Record the commit each measurement
   was taken at.
