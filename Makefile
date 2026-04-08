# Makefile for TensorBoardX

.PHONY: help init test coverage lint format clean docs

PROTO_SRC = $(wildcard tensorboardX/proto/*.proto)
PROTO_OBJ = $(PROTO_SRC:.proto=_pb2.py)

# Default target
help:
	@echo "TensorBoardX Development Tasks:"
	@echo "  init        Setup development environment (venv and dependencies)"
	@echo "  compile     Regenerate protobuf files (only if changed)"
	@echo "  test        Run all tests"
	@echo "  coverage    Run tests and show coverage report"
	@echo "  lint        Run ruff linter"
	@echo "  format      Run ruff formatter"
	@echo "  docs        Build documentation"
	@echo "  clean       Remove build artifacts and temporary files"

# Environment Setup
init:
	uv venv --python 3.13
	uv pip install -e ".[dev]"
	uv pip install "setuptools==81.0.0"
	@echo "Environment initialized. Run 'source .venv/bin/activate' to use it."

# Protobuf Compilation (Dependency-aware)
compile: $(PROTO_OBJ)

%_pb2.py: %.proto
	./compile.sh

# Testing
test:
	uv run pytest

coverage:
	uv run pytest --cov=tensorboardX --cov-report=term-missing

# Quality Control
lint:
	uv run ruff check tensorboardX/

format:
	uv run ruff format tensorboardX/ tests/

# Documentation
docs:
	$(MAKE) -C docs html

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f debug_output.gif debug_video.py
