# Legacy Scripts

This directory is kept only as reference during the migration to the `src`-based project layout.

## Status
- Do not add new features here
- Do not treat these files as the primary runtime entrypoint
- Port any reusable logic into `src/` before extending it

## Current Contents
- historical image-processing experiments
- old dependency and formatting helpers
- transitional georeferencing and statement scripts

When there is a conflict between `scripts/` and `src/`, prefer `src/`.
