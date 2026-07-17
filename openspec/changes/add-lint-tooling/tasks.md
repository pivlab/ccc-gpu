# Tasks — add-lint-tooling

## 1. Python tooling consolidation

- [ ] 1.1 Move/tune `[tool.ruff]` into root `pyproject.toml` (rule set from the scaffold minus `required-imports`; per-file-ignores for `tests/` and `analysis/`)
- [ ] 1.2 Update `.pre-commit-config.yaml`: drop black/black-jupyter, bump ruff hook pin to the current version, keep hygiene hooks
- [ ] 1.3 Verify ruff config no longer resolves from `libs/ccc/pyproject.toml` (deletion itself happens in cleanup-dead-code)

## 2. C++/CUDA tooling

- [ ] 2.1 Add `.clang-format` (Microsoft base, IndentWidth 4, ColumnLimit 120, PointerAlignment Right, sorted includes)
- [ ] 2.2 Add `.clang-tidy` with the curated check list and exclusions
- [ ] 2.3 Add clang-format pre-commit hook with `types_or: [c++, c, cuda]`
- [ ] 2.4 Set `CMAKE_EXPORT_COMPILE_COMMANDS ON` in root `CMakeLists.txt`; document the clang-tidy invocation (`--cuda-host-only`) in the dev docs

## 3. Apply formatting

- [ ] 3.1 Run `pre-commit run --all-files`; fix any non-mechanical fallout (ruff lint errors needing judgment)
- [ ] 3.2 Land formatting as a single isolated commit; add `.git-blame-ignore-revs` and document it

## 4. CI workflow

- [ ] 4.1 Add `.github/workflows/ci.yml` with the lint job (pre-commit run --all-files)
- [ ] 4.2 Add wheel-build smoke job (CUDA toolkit via container or apt, compile-only, cached)
- [ ] 4.3 Add CPU-test job (initially `--ignore=tests/gpu`; switch to `-m` marker expression once restructure-tests lands)
- [ ] 4.4 Add clang-tidy report-only CI step; file follow-up issues for existing findings

## 5. Verification

- [ ] 5.1 Fresh-clone check: `pre-commit run --all-files` passes; CI green on a no-op PR
