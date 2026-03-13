# Documentation

This directory tracks the migration from legacy scripts to the `src`-based project structure.

## Available Documents
- `architecture.md`: target architecture and what is already implemented
- `workflows.md`: current recommended development workflow
- `migration_status.md`: active migration decisions, constraints, and next steps

## Current State
- `src/` is the active development surface
- Dask lazy TIFF reading is the most stable implemented feature
- `scripts/` is legacy reference code and should not be used as the primary entrypoint
- algorithm pipeline modules remain transitional

## Building Documentation
There is no separate generated documentation pipeline in this iteration. Keep the Markdown files in `docs/` as the source of truth until a dedicated docs stack is introduced.
