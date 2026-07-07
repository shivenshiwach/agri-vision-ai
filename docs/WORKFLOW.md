# Workflow

## Local Setup

Use the repository virtual environment when available:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

If the environment is not activated, prefer explicit commands:

```bash
venv/bin/python -m pytest
venv/bin/python scripts/check_env.py
```

Run the API:

```bash
uvicorn api.main:app --reload
```

## Daily Development Loop

1. Check status with `git status --short`.
2. Read relevant docs before changing scripts or model behavior.
3. Run the smallest applicable validation command.
4. Keep generated artifacts under ignored directories.
5. Update docs when workflow, commands, or model assumptions change.

## Dataset Acquisition

Always start with dry-run:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
python scripts/download_sources.py --source ip102 --dry-run
```

Use `--confirm` only after license, storage, class mapping, and dataset version have been reviewed. PlantVillage and IP102 are the only sources with real downloader code. Other source entries are planning records.

## Dataset Inspection

Raw dataset inspection:

```bash
python scripts/inspect_datasets.py
python scripts/dataset_inventory.py
```

Disease source inspection:

```bash
python scripts/inspect_disease_datasets.py --dry-run
python scripts/inspect_disease_datasets.py --report-csv outputs/disease_label_report.csv
```

## PlantVillage Classifier Path

Build the processed dataset:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
python scripts/build_full_plantvillage_classifier_dataset.py --confirm
```

Train:

```bash
python scripts/train_image_classifier.py --config configs/full_plantvillage_training.yaml
```

Evaluate:

```bash
python scripts/evaluate_image_classifier.py --config configs/full_plantvillage_training.yaml
```

## Multi-Dataset Disease Path

This path prepares a normalized staging dataset only. It does not train models.

```bash
python scripts/build_multi_disease_classifier_dataset.py --dry-run
python scripts/build_multi_disease_classifier_dataset.py --confirm
```

Review `manifest.csv`, `classes.csv`, unmapped labels, duplicates, and class balance before defining splits or training.

## Inference API Path

Current inference is mock-only. The API should not be connected to model files until a validated checkpoint exists and preprocessing matches training/evaluation.

