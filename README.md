# Satellite Image Processing Project

## Current Status
This repository is in an active migration to a `src`-centered layout.

- Active development happens under `src/`
- The stable area in this iteration is Dask-backed lazy TIFF reading
- `scripts/` contains legacy code from earlier experiments and is not the active entrypoint
- Pipeline, segmentation, registration, and georeferencing modules remain under construction

See [docs/migration_status.md](docs/migration_status.md) for the current migration notes.

## Active Focus: Dask Lazy Image Reading
The reference implementation for lazy frame loading lives in `src.utils.image.dask_io`.

Public entrypoints:

- `src.utils.imread_lazy`
- `src.utils.imread_lazy_with_paths`

Current behavior contract:

- accepts a directory, glob pattern, single file path, or explicit list of paths
- raises `FileNotFoundError` when no TIFF files are resolved
- returns a Dask array shaped `(n_frames, height, width)`
- uses one chunk per frame
- sorts files deterministically by default
- validates that all frames share the same shape and dtype

Example:

```python
from src.utils import imread_lazy, imread_lazy_with_paths

stack = imread_lazy("data/input_frames/AerialData_GeoModule/ImageFrames/")
frame_0 = stack[0].compute()
batch = stack[10:20].compute()

stack_with_paths, paths = imread_lazy_with_paths(
    "data/input_frames/AerialData_GeoModule/ImageFrames/"
)
```

## Repository Layout
```text
.
├── data/                     # Input, metadata, and processed outputs
├── docs/                     # Project and migration documentation
├── requirements/            # Dependency inputs and compiled lock file
├── scripts/                 # Legacy scripts kept for reference only
├── src/
│   ├── algorithms/          # Target home for the new processing pipeline
│   ├── utils/               # Shared utilities, including lazy Dask I/O
│   └── validation/          # Validation hooks and metrics
└── tests/                   # Automated tests
```

## Target Architecture vs Current Reality
### Target architecture
The intended end state is a modular processing pipeline with:

1. band segmentation
2. inter-frame registration
3. mosaic generation
4. georeferencing

### Current implementation status

- `src.utils.image.dask_io` is implemented and tested
- `src.algorithms.pipeline` defines the intended orchestration surface
- pipeline step classes in `src.algorithms.pipeline.steps.image_processing` are placeholders
- algorithm modules may contain duplicated or transitional code while the migration continues

The pipeline example below is illustrative of the target API, not a production-ready workflow in the current revision.

```python
from src.algorithms.pipeline import (
    Pipeline,
    BandSegmentationStep,
    InterFrameRegistrationStep,
    MosaicGenerationStep,
)

pipeline = Pipeline("image_processing")
pipeline.add_step(BandSegmentationStep())
pipeline.add_step(InterFrameRegistrationStep())
pipeline.add_step(MosaicGenerationStep())

pipeline.set_config(
    {
        "band_count": 5,
        "registration_method": "feature_based",
        "mosaic_overlap": 0.3,
    }
)

result = pipeline.run(input_data)
```

At the moment the step implementations are placeholders and should not be treated as complete processing features.

## Development
Docker remains the expected local environment.

```bash
docker compose build
docker compose run --rm app bash
```

Useful commands inside the environment:

```bash
pytest tests/test_dask_io.py
pytest -m dask
make test
```

## Legacy Scripts
Files under `scripts/` are retained for historical reference during the migration. They may contain useful ideas or code to port later, but they are not the source of truth for the current architecture.

## Documentation
- [docs/README.md](docs/README.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/workflows.md](docs/workflows.md)
- [docs/migration_status.md](docs/migration_status.md)

## License
[MIT License](LICENSE)
