# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- **BREAKING (wheel contents):** the published `cccgpu` wheel no longer ships the
  research/analysis-only modules `ccc.plots`, `ccc.methods`, `ccc.giant`, and
  `ccc.corr`. These modules import dependencies that are not declared as runtime
  requirements (IPython, minepy, requests) and were only used by the code under
  `analysis/`. They remain available in a source checkout but are excluded from
  the PyPI package. Code that did `from ccc.plots import ...` (or
  `methods`/`giant`/`corr`) against an installed wheel must now install the extra
  dependencies and run from a source checkout instead.

### Added

- Declared `pyyaml` as a runtime dependency so that every module shipped in the
  wheel (notably `ccc.log`, used by `ccc.utils.curl`) imports cleanly with the
  declared dependencies only.

### Changed

- Cleaned up dead code across the library and repository: removed the unreachable
  CPU compute stack duplicated in `ccc.coef.impl_gpu`, the abandoned
  `ccc.sklearn.metrics_gpu` prototype, the empty `ccc.numpy` subpackage, demo/dead
  CUDA code (`cub.cu`, `example_return_optional_vectors`, `ari_reduced`), an
  orphaned CMake project and broken test stubs, committed benchmark/test `.log`
  files, and upstream Docker/environment leftovers. The public `ccc()` GPU entry
  point and its behavior are unchanged.
