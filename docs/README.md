# Documentation

This directory contains the project documentation.

## Structure

- `api/`: API documentation and reference
- `user_guide/`: End-user documentation and tutorials
- `development/`: Developer documentation and guidelines

## Building Documentation

To build the documentation locally:

1. Install dependencies:
```bash
pip install -r requirements/requirements-dev.txt
```

2. Build documentation:
```bash
cd docs
make html
```

Documentation will be available in `_build/html/`. 