# Tasks — upgrade-dependencies

## 1. Environment refresh

- [ ] 1.1 Write the real source env file (py3.12, numpy 2.x, numba ≥0.61, scipy/pandas current, CUDA 12.8/12.9 toolchain); regenerate `conda-lock.yml`
- [ ] 1.2 Full CPU + GPU suite in the new env on GPU hardware; fix any numba/numpy fallout
- [ ] 1.3 Consolidate env files: fold/delete `environment-benchmark.yaml` and `environment-toolchain.yaml`; update env docs

## 2. Build configuration

- [ ] 2.1 pyproject build-system: drop setuptools/wheel; add `scikit-build-core>=0.10`; validate pybind11 3.x then pin `>=2.13,<4`
- [ ] 2.2 Set `CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90`; measure wheel size; single CMake floor across pyproject + CMakeLists (refresh `...3.26` policy ceiling)
- [ ] 2.3 cibuildwheel: bump toolkit install to 12.8/12.9 (fix the mismatched comment); gate googletest fetch already handled in restructure-tests — verify no network fetch in wheel builds

## 3. Runtime deps and extras

- [ ] 3.1 Add `[project.optional-dependencies]`: `plots`, `research`, `test` extras (coordinate with cleanup-dead-code wheel scoping); replace the private `seaborn.distributions._freedman_diaconis_bins` import in `plots.py`
- [ ] 3.2 Audit declared vs imported deps for shipped modules (import-walk check from cleanup track reused)

## 4. Support matrix validation

- [ ] 4.1 Build cp310–cp314 wheels via cibuildwheel; run CPU suite per wheel; note cp313/cp314 status
- [ ] 4.2 Align Python-range and CUDA/CC claims in README/classifiers/docs with what actually built (with improve-docs)

## 5. Verification

- [ ] 5.1 Fresh `pip install` from built wheel on a clean machine: verification one-liner + GPU smoke; changelog entries for floors/arch changes
