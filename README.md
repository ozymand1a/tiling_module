# Tiling Module

Image tiling, slicing, and object detection postprocessing (NMS, NMM, WBF) with YOLO/ONNX support.

## Quickstart (uv)

```bash
# Create venv and install project (uv will read pyproject.toml)
uv sync

# Optional: include dev tools (ruff) and script extras
make install

# Run the module (example)
PYTHONPATH=. uv run python -m scripts.check_tiling --image data/1.jpg --output out.jpg
```

Install with optional dependencies (e.g. large image support, SAHI for scripts):

```bash
uv sync --extra large-images --extra scripts --extra dev
```

## Linting & formatting (Ruff)

```bash
# Check only (lint + format check)
uv run ruff check src scripts
uv run ruff format --check src scripts

# Fix and format (modify files)
uv run ruff check --fix src scripts
uv run ruff format src scripts
```

## Example

```bash
PYTHONPATH="." python3 run.py -m weights/yolo26m.onnx -i data/small-vehicles.jpeg -o output.jpeg
```
