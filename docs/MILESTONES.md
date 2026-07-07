# Milestones

## Completed Repository Milestones

### Initial Scaffold

- FastAPI app with `/health` and `/predict`.
- Pydantic response schema.
- Deterministic mock inference.
- Basic pytest health test.

### Taxonomy And Dataset Mapping

- `configs/classes.yaml` defines crop, pest, disease, and animal-related labels.
- `data/taxonomy/*.csv` stores lightweight metadata and dataset coverage estimates.
- Heavy data remains out of Git.

### Dataset Planning And YOLO Validation

- `configs/yolo_dataset_template.yaml` defines a 22-class detector plan.
- `scripts/build_yolo_dataset.py` prints planned classes and layout.
- `scripts/validate_yolo_dataset.py` validates YOLO image/label folders when data exists.

### Guarded Dataset Acquisition

- `scripts/download_sources.py` centralizes source dry-runs.
- PlantVillage and IP102 have confirmed downloader implementations.
- Other sources remain planning-only entries.

### Dataset Inspection

- `scripts/inspect_datasets.py` reports PlantVillage and IP102 local state.
- `scripts/extract_ip102.py` safely extracts approved IP102 tar files.
- `scripts/dataset_inventory.py` summarizes local raw dataset folders.

### Full PlantVillage Classifier Path

- `scripts/build_full_plantvillage_classifier_dataset.py` creates a train/val ImageFolder dataset.
- `scripts/train_image_classifier.py` trains EfficientNet-B0.
- `scripts/evaluate_image_classifier.py` reports accuracy and writes a confusion matrix.
- Defaults live in `configs/full_plantvillage_classifier.yaml` and `configs/full_plantvillage_training.yaml`.

### Multi-Dataset Disease Preparation

- `configs/disease_dataset_registry.yaml` defines disease source roots and discovery rules.
- `data/taxonomy/disease_label_aliases.csv` maps dataset labels to canonical disease labels.
- `scripts/inspect_disease_datasets.py` reports mapping coverage.
- `scripts/build_multi_disease_classifier_dataset.py` stages normalized images with a manifest.
- This path does not train models.

## Current Milestone: Documentation Consolidation

This documentation pass records architecture, current status, missing pieces, workflows, testing, git practices, and recommended next work without modifying source code or configs.

## Recommended Next Milestone: Stabilization

Before training or API model integration:

- Commit or intentionally drop current disease-preparation workspace changes.
- Keep dependency and environment reproducibility checks current.
- Keep YOLO path resolution covered by tests.
- Keep `/predict` and script dry-runs covered by tests.
- Record dataset license/version requirements for confirmed downloads.
