# Current Status

Inspected on 2026-07-07.

## Git State

Current branch: `shiven`. Remote: `origin` at `https://github.com/shivenshiwach/agri-vision-ai.git`.

Tracked branch state:

- `shiven` and `origin/shiven` point to `036b939 Make full PlantVillage classifier the main training path`.
- `main` and `origin/main` point to `ab12fef Add Maharashtra taxonomy and dataset mapping`.

Current workspace is dirty before these documentation updates. Existing non-doc changes include:

- Modified: `README.md`, `configs/download_sources.yaml`, `docs/DATASET_INSPECTION.md`, `docs/DATA_ACQUISITION_WORKFLOW.md`, `docs/REAL_DATASETS.md`, `scripts/download_sources.py`.
- Untracked: disease registry config, disease alias CSV, disease prep scripts, disease prep docs, and `AGENTS.md`.

These files were inspected and preserved.

## Implemented

- FastAPI app with `/health` and `/predict`.
- Pydantic response schemas for recommendations, boxes, and predictions.
- Mock inference contract in `api/inference.py`.
- Taxonomy and dataset planning metadata.
- Guarded PlantVillage and IP102 download tooling.
- Raw dataset inspection for PlantVillage and IP102.
- IP102 safe extraction script.
- Full PlantVillage ImageFolder dataset builder.
- EfficientNet-B0 training script for ImageFolder train/val datasets.
- EfficientNet-B0 evaluation script with confusion matrix output.
- Multi-dataset disease label inspection and normalized staging.
- YOLO class planning and basic YOLO dataset validation.
- Pytest coverage for `/health`, mock `/predict`, safe dry-runs, and YOLO path helpers.

## Verified Locally

Using default `python`:

- `python -m pytest` fails because FastAPI is not installed in the default environment.

Using `venv/bin/python`:

- `venv/bin/python -m pytest` passes: 9 tests passed.
- `venv/bin/python scripts/check_env.py` reports Python 3.12.4, Torch 2.12.1, CUDA unavailable, and Ultralytics 8.4.83.
- Dataset dry-runs work and do not create files.

## Not Present

- No local `datasets/` directory.
- No local `models/` directory.
- No trained checkpoints or exported model artifacts.
- No generated evaluation outputs.
- No production model inference integration.
- No CI workflow.
- No formatter or linter configuration such as `pyproject.toml`, `ruff.toml`, `setup.cfg`, or `.flake8`.
- No package lock file.

## Technical Debt

- `api/inference.py` is still deterministic mock logic.
- The active classifier uses original PlantVillage folder names, not the project taxonomy.
- Multi-dataset disease staging does not create train/val/test splits.
- YOLO conversion from real annotations is not implemented.
- `requirements.txt` includes `httpx2`; this was verified as correct for the installed Starlette TestClient, which imports `httpx2 as httpx`.
- Test coverage is still limited and does not cover training settings, disease label normalization, checkpoint loading, or evaluation failure paths.
