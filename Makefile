.PHONY: install format lint test clean

install:
	bash scripts/install_dependencies.sh

format:
	bash scripts/format_code.sh

lint:
	pylint src/ tests/

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 