# Workflows

## Current Development Workflow
1. Add or update implementation under `src/`.
2. Keep reusable helpers in `src.utils`.
3. Add or update tests under `tests/`.
4. Use `scripts/` only as legacy reference material, not as the primary execution path.

## Dask Lazy I/O Workflow
Use this workflow for large TIFF frame sets:

1. Resolve input frames through `imread_lazy` or `imread_lazy_with_paths`
2. Operate on slices or per-frame indexing to preserve lazy execution
3. Materialize data with `.compute()` only at explicit processing boundaries
4. Keep downstream processing code compatible with `(frame, height, width)` stacks

## Testing Workflow
- Run `pytest tests/test_dask_io.py` during Dask I/O changes
- Run `pytest -m dask` for the fast lazy-loading suite
- Run `make test` before merging broader changes

## Migration Workflow
When moving functionality from legacy code:

1. identify the reusable logic in `scripts/`
2. reimplement or port it under the correct `src` subpackage
3. add tests in `tests/`
4. update docs once `src` becomes the source of truth

Do not add new production behavior directly into `scripts/`.
