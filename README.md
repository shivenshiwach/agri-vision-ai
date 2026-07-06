# Agri Vision AI

AI vision module scaffold for detecting crop, pest, disease, and animal problems from uploaded farm images.

This setup does not include datasets, trained models, checkpoints, or generated artifacts. The current API returns deterministic mock predictions so the project structure and integration contract can be developed safely. The active training path is the full PlantVillage image classifier.

## Project Structure

- `api/`: FastAPI app, response schemas, and placeholder inference function.
- `configs/`: Taxonomy, dataset, and training configuration.
- `data/taxonomy/`: Lightweight CSV taxonomy and dataset mapping metadata.
- `docs/`: Project plan, dataset research notes, and training guides.
- `scripts/`: Dataset, inspection, training, and evaluation utilities.
- `training/`: Placeholder for future training workflows.
- `tests/`: API tests.

## Current Training Path

The main classifier path uses all available PlantVillage `raw/color` class folders as a 38-class ImageFolder dataset.

1. Download or place PlantVillage under `datasets/raw/plantvillage`.
2. Inspect raw datasets without modifying files.
3. Build `datasets/processed/full_plantvillage_classifier`.
4. Train EfficientNet-B0 with `scripts/train_image_classifier.py`.
5. Evaluate the validation split with `scripts/evaluate_image_classifier.py`.

Raw datasets, processed datasets, checkpoints, and generated metrics should stay out of Git.

## Dataset Acquisition And Inspection

Dry-run source inspection:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
```

Confirmed PlantVillage download after source review:

```bash
python scripts/download_sources.py --source plantvillage --confirm
```

Inspect local raw datasets:

```bash
python scripts/inspect_datasets.py
```

PlantVillage downloads write to `datasets/raw/plantvillage`, which remains ignored by Git.

## Full PlantVillage Dataset

- Dataset builder: `scripts/build_full_plantvillage_classifier_dataset.py`
- Dataset config: `configs/full_plantvillage_classifier.yaml`
- Dataset guide: `docs/FULL_PLANTVILLAGE_CLASSIFIER.md`

Preview the processed dataset plan:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
```

Create the processed dataset after review:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --confirm
```

Default output:

```text
datasets/processed/full_plantvillage_classifier
```

## Full PlantVillage Training

- Training config: `configs/full_plantvillage_training.yaml`
- Training script: `scripts/train_image_classifier.py`
- Evaluation script: `scripts/evaluate_image_classifier.py`
- Training guide: `docs/FULL_PLANTVILLAGE_TRAINING.md`

Show training options:

```bash
python scripts/train_image_classifier.py --help
```

Train manually:

```bash
python scripts/train_image_classifier.py --config configs/full_plantvillage_training.yaml
```

Show evaluation options:

```bash
python scripts/evaluate_image_classifier.py --help
```

Evaluate the validation split:

```bash
python scripts/evaluate_image_classifier.py --config configs/full_plantvillage_training.yaml
```

Default model output:

```text
models/full_plantvillage_classifier/
  best.pt
  last.pt
  classes.json
  history.json
  confusion_matrix.png
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
pytest
```

## Development Checks

```bash
python scripts/check_env.py
pytest
python scripts/train_image_classifier.py --help
python scripts/evaluate_image_classifier.py --help
python -m compileall scripts
git diff --check
```

## API

Start the API:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/farm-image.jpg"
```
