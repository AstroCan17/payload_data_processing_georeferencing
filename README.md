# Satellite Image Processing Project

## Overview
This project processes multispectral aerial imagery datasets for:
- Band segmentation
- Inter-frame registration
- Mosaic generation
- False-Color Composite (FCC) creation
- Georeferencing
- Generation of multiband mosaics for subsets
- Pansharpening

## Installation (Docker + Dask)

The environment runs in Docker; distributed processing uses Dask. Only Docker is required on the host.

```bash
git clone https://github.com/AstroCan17/payload_data_processing_georeferencing.git
cd payload_data_processing_georeferencing
docker compose build
docker compose run --rm app bash
```

Inside the container the project and dependencies are ready. Run tests with `make test` or `pytest tests/`.

**Dask:** For large data, the pipeline uses `use_dask=True` and `n_workers`; the default scheduler is processes. A dedicated Dask scheduler/worker service can be added later.

## Project Structure
```
.
├── data/                    # Data directory
│   ├── input_frames/        # Raw image frames
│   ├── metadata/           # IMU and other metadata
│   └── processed/          # Processed outputs
├── docs/                    # Documentation
│   ├── api/                # API documentation
│   ├── user_guide/         # User guides
│   └── development/        # Development guides
├── requirements/            # Dependency management
│   ├── requirements.in     # Primary requirements
│   └── requirements.txt    # Locked dependencies
├── scripts/                 # Utility scripts
│   ├── format_code.sh      # Code formatting
│   ├── auto_dependencies.sh # Dependency management
│   └── install_dependencies.sh # Installation
├── src/                    # Source code
│   ├── algorithms/         # Core algorithms
│   │   └── pipeline/      # Processing pipeline
│   │       ├── base.py    # Base pipeline classes
│   │       └── steps/     # Pipeline processing steps
│   ├── utils/             # Utility functions
│   └── validation/        # Validation code
└── tests/                  # Test suite
```

## Pipeline Architecture

The project uses a modular pipeline architecture for image processing:

### Base Classes
- `Pipeline`: Main pipeline manager
- `PipelineStep`: Abstract base class for pipeline steps

### Processing Steps
1. **Band Segmentation**
   - Separates multi-band images into individual spectral bands
   - Handles: Red, Green, Blue, NIR, Red Edge bands

2. **Inter-Frame Registration**
   - Aligns consecutive frames
   - Compensates for aircraft movement
   - Sub-pixel precision alignment

3. **Mosaic Generation**
   - Combines registered frames
   - Handles frame stitching
   - Maintains spectral integrity

## Usage Example

```python
from src.algorithms.pipeline import (
    Pipeline,
    BandSegmentationStep,
    InterFrameRegistrationStep,
    MosaicGenerationStep
)

# Create pipeline
pipeline = Pipeline("image_processing")

# Add processing steps
pipeline.add_step(BandSegmentationStep())
pipeline.add_step(InterFrameRegistrationStep())
pipeline.add_step(MosaicGenerationStep())

# Configure pipeline
pipeline.set_config({
    "band_count": 5,
    "registration_method": "feature_based",
    "mosaic_overlap": 0.3
})

# Process data
result = pipeline.run(input_data)
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Running Tests
```bash
# Run all tests
make test

# Run specific test
pytest tests/
```

### Code Formatting
```bash
# Format all code
make format
```

## License
[MIT License](LICENSE)

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for version history.
