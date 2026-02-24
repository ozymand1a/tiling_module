.PHONY: install check

install:
	uv sync --extra dev --extra large-images --extra scripts

check:
	uv run ruff check src scripts
	uv run ruff format --check src scripts
