# Contributing Guidelines

## Development Setup
1. Clone the repository
2. Install dependencies: `bash scripts/install_dependencies.sh`
3. Set up pre-commit hooks: `pre-commit install`

## Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Run `bash scripts/format_code.sh` before committing

## Testing
- Write tests for new features
- Run tests with `pytest tests/`
- Ensure all tests pass before submitting PR

## Pull Request Process
1. Create a feature branch
2. Update documentation as needed
3. Update CHANGELOG.md
4. Submit PR with clear description 