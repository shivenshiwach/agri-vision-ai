# Project Context

Inspected on 2026-07-06 from branch `shiven`.

## Purpose

Agri Vision AI is a Python image-vision module for farm images. The intended product direction is to identify crops, pests, diseases, and animal-related farm problems, then return structured predictions, severity, confidence, detection boxes, and recommendations.

## Architecture

The repository is organized around a lightweight API plus offline dataset and training utilities.

```text
api/                  FastAPI app, schemas, and inference entry point
scripts/              Dataset, inspection, conversion, training, and evaluation utilities
configs/              Taxonomy, source, YOLO, and classifier training configuration
data/taxonomy/        Tracked CSV metadata only
docs/                 Project, dataset, training, and workflow documentation
tests/                Pytest coverage for API behavior
training/             Placeholder for future training workflows
```

Heavy artifacts are intentionally outside Git. The ignored locations are `datasets/`, `models/`, `runs/`, `outputs/`, raw data under `data/*`, checkpoints, ONNX/TensorRT exports, `.env`, and virtual environments.

## Current Runtime Flow

`api/main.py` exposes `GET /health` and `POST /predict`. `POST /predict` validates that the upload is an image and that bytes are present, then calls `api/inference.py`. The current inference function returns a deterministic mock `PredictionResult`; it does not load trained models.

## Dataset Strategy

The taxonomy in `configs/classes.yaml` contains 30 crops, 53 pests, 48 diseases, and 13 animal-related classes. `data/taxonomy/*.csv` tracks lightweight planning metadata and dataset coverage estimates.

Implemented dataset support:

- PlantVillage and IP102 guarded download commands exist.
- PlantVillage raw/color can be converted into an ImageFolder train/val classifier dataset.
- IP102 tar extraction is implemented with path traversal checks.
- Multi-dataset disease inspection and normalized staging are implemented for class-folder datasets.
- YOLO support is currently planning and validation only.

No local `datasets/` directory exists in this workspace.

## Model Strategy

The active training path is an EfficientNet-B0 image classifier using ImageFolder data at `datasets/processed/full_plantvillage_classifier`. Training writes `best.pt`, `last.pt`, `classes.json`, and `history.json` to `models/full_plantvillage_classifier`. Evaluation loads a checkpoint, validates class folders, reports accuracy, and writes `confusion_matrix.png`.

No trained model files exist in this workspace. `configs/model_config.yaml` contains placeholder paths for future disease, pest, animal, and crop models.

## Dependencies

Runtime and ML dependencies are listed in `requirements.txt`: FastAPI, Uvicorn, PyTorch, TorchVision, Ultralytics, Pillow, OpenCV, NumPy, Pandas, PyYAML, pytest, requests, and others. The active shell Python lacks FastAPI, but `venv/bin/python` has the needed test/runtime dependencies.

