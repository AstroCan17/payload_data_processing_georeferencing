# Architecture

## Guiding Structure
The project is moving from one-off scripts to a modular `src` layout:

- `src.utils`: shared utilities and stable helpers
- `src.algorithms`: domain algorithms and future pipeline integrations
- `src.validation`: validation hooks, checks, and metrics

## Current Source of Truth
In the current revision, the most reliable implementation is the lazy TIFF loading layer in `src.utils.image.dask_io`.

It provides:
- deterministic path resolution
- lazy Dask-backed frame loading
- stack-level metadata validation for shape and dtype consistency
- optional path return for downstream metadata alignment

## Target Processing Flow
The intended end-state processing flow is:

1. load source frames lazily
2. segment bands
3. register consecutive frames
4. generate mosaics
5. georeference outputs

## Implementation Reality
- Pipeline orchestration classes exist in `src.algorithms.pipeline`
- Step implementations are placeholders and should be treated as interface stubs
- Some algorithm domains currently exist in duplicated or transitional module layouts
- Legacy `scripts/` files are not part of the active architecture

## Near-Term Rule
If functionality exists in both `src/` and `scripts/`, `src/` is the reference location for all new development and tests.
