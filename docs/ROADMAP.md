# Roadmap

## Guiding Rules

- Keep datasets, checkpoints, runs, and exported models out of Git.
- Prefer dry-run commands before any write-heavy operation.
- Do not wire a model into API inference until a checkpoint has documented validation results.
- Preserve the FastAPI response contract unless an API version change is intentional.

## Recommended Next Milestone

Milestone 14 should be repository stabilization before more model work.

Scope:

1. Decide whether the current untracked disease-preparation files belong in the next commit.
2. Keep environment reproducibility explicit: use `venv/bin/python`; `httpx2` is intentional for the installed Starlette TestClient.
3. Keep YOLO dataset paths resolving to repo-level `datasets/yolo_detector`.
4. Maintain `/predict`, safe dry-run, and YOLO path tests.
5. Keep exact dataset version and license checks documented before any confirmed downloads.

This milestone should not train models or add datasets.

## Short-Term Milestones

### Milestone 15: PlantVillage Baseline Run

- Download or mount PlantVillage on an approved machine.
- Run raw inspection and record class counts.
- Build `datasets/processed/full_plantvillage_classifier`.
- Train EfficientNet-B0 using `configs/full_plantvillage_training.yaml`.
- Evaluate validation accuracy and save a human-readable run report.

### Milestone 16: Inference Integration Prototype

- Add model loading behind configuration.
- Convert uploaded image bytes into the same preprocessing used during evaluation.
- Return classifier predictions through the existing `PredictionResult` schema.
- Keep mock inference available for development if no checkpoint exists.

### Milestone 17: Disease Dataset Expansion

- Review approved disease sources and licenses.
- Run `inspect_disease_datasets.py` with real source folders.
- Resolve unmapped labels and duplicate normalized labels.
- Define train/val/test split policy for normalized disease data.

### Milestone 18: Detector Pipeline

- Decide first detector classes from `configs/yolo_dataset_template.yaml`.
- Implement conversion from approved source annotations to YOLO labels.
- Validate labels with `validate_yolo_dataset.py`.
- Train and evaluate an Ultralytics baseline only after data quality review.

## Longer-Term Work

- Crop classifier and routing model.
- Pest classifier or detector from IP102 and field data.
- Animal intrusion detector using field-camera data.
- Severity estimation using counts, affected area, crop stage, and feedback.
- API feedback capture and prediction monitoring.
