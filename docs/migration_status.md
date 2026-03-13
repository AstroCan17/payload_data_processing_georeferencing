# Migration Status

## Active Decisions
- `src/` is the only active development surface for new code.
- `src.*` remains the external import style in this phase.
- Dask-backed lazy TIFF loading is the first stabilized subsystem.
- `scripts/` is preserved as legacy reference code and is not an active runtime path.

## Stable in This Iteration
- `src.utils.image.dask_io`
- `src.utils` exports for lazy loading
- Dask-focused tests under `tests/test_dask_io.py`

## Transitional Areas
- `src.algorithms.pipeline` defines the target orchestration shape but step implementations are placeholders.
- Segmentation, registration, and georeferencing modules may contain duplicated or legacy-adapted code while the migration is ongoing.
- Packaging metadata still needs a later cleanup pass once the module structure settles.

## Legacy Inventory
Legacy files currently retained under `scripts/`:

- `statement_1.py`
- `statement_2.py`
- `statement_2_new.py`
- `install_dependencies.sh`
- `format_code.sh`
- `auto_dependencies.sh`

These files may be mined for reusable logic later, but no new feature work should target them directly.

## Next Refactor Targets
1. Remove or consolidate duplicated algorithm modules once the new `src` implementation is selected as the source of truth.
2. Replace placeholder pipeline steps with real `src.algorithms` integrations.
3. Clean up packaging metadata after the namespace and module layout stabilizes.
