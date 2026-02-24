.PHONY: install check

install:
	uv sync --extra dev

check:
	uv run ruff check src scripts
	uv run ruff format --check src scripts
