# Testing Guide

## Current Test Stack

Tests use `pytest` and FastAPI `TestClient`. The only current test is `tests/test_api_health.py`, which checks `GET /health`.

## Known Environment Requirement

The default shell Python used during inspection did not have FastAPI installed, so this failed:

```bash
python -m pytest
```

The repository virtual environment passed:

```bash
venv/bin/python -m pytest
```

Result on 2026-07-07 after Milestone 14 stabilization tests: 9 tests passed.

## Core Validation Commands

Use these before committing Python or workflow changes:

```bash
venv/bin/python -m pytest
venv/bin/python scripts/check_env.py
venv/bin/python scripts/train_image_classifier.py --help
venv/bin/python scripts/evaluate_image_classifier.py --help
venv/bin/python scripts/inspect_disease_datasets.py --help
venv/bin/python scripts/build_multi_disease_classifier_dataset.py --help
git diff --check
```

Use dry-runs for dataset logic:

```bash
venv/bin/python scripts/download_sources.py --source plantvillage --dry-run
venv/bin/python scripts/inspect_datasets.py
venv/bin/python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
venv/bin/python scripts/inspect_disease_datasets.py --dry-run
venv/bin/python scripts/build_multi_disease_classifier_dataset.py --dry-run
```

## Current Validation Observations

- PlantVillage and IP102 are missing locally, so dataset builders report zero images.
- Disease inspection reports all enabled disease source roots as missing locally.
- YOLO validation currently fails because no YOLO dataset exists locally. Relative YAML `path` values now resolve from the current working directory, so the default path points to repo-level `datasets/yolo_detector`.

## Test Coverage Gaps

- Pydantic schema constraints.
- Config parsing defaults and invalid values.
- Disease label normalization and duplicate detection.
- Checkpoint loading and evaluation failure paths.

Avoid tests that require real datasets, checkpoints, internet access, or GPU availability unless they are explicitly marked and isolated.
