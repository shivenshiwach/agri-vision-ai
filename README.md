# Agri Vision AI

AI vision module scaffold for detecting crop, pest, disease, and animal problems from uploaded farm images.

This setup does not include datasets, trained models, checkpoints, or generated artifacts. The current API returns deterministic mock predictions so the project structure and integration contract can be developed safely.

## Project Structure

- `api/`: FastAPI app, response schemas, and placeholder inference function.
- `configs/`: Taxonomy and future model configuration.
- `data/taxonomy/`: Lightweight CSV taxonomy and dataset mapping metadata.
- `docs/`: Project plan, dataset research notes, and taxonomy explanation.
- `scripts/`: Development utility scripts.
- `training/`: Placeholder for future training workflows.
- `tests/`: API tests.

## AI Roadmap

Milestone 1 defines the Maharashtra agriculture taxonomy and dataset mapping. It covers crop, pest, disease, animal, bird, rodent, and reptile classes from the project document.

Planned model path:

1. Build a crop classifier to route uploaded images by crop.
2. Train focused pest detectors for high-priority visible insects and damage.
3. Train disease detectors or classifiers for common leaf, fruit, and storage symptoms.
4. Train animal detectors for wildlife, birds, rodents, and reptiles in field or storage settings.
5. Add severity estimation using detected object counts, affected area, crop stage, and economic thresholds.

Recommended first training targets:

- Crops: Cotton, Soybean, Sugarcane, Paddy (Rice), Wheat, Grapes, Pomegranate, Banana, Tomato, and Onion.
- Pests: Pink Bollworm, Whitefly, Thrips, Aphids, Fruit Borer, Stem Borer, and Fall Armyworm.
- Diseases: Rust, Powdery Mildew, Downy Mildew, Early Blight, Late Blight, Bacterial Blight, Leaf Curl Virus, and Blast Disease.
- Animals: Wild Boar, Monkeys, Birds, Parrots, Sparrows, Rats, and Snakes.

Before training, review `docs/DATASET_GAPS.md` and confirm dataset licenses, class mappings, and annotation formats. Keep raw data under ignored dataset directories.

## Milestone 2: Dataset Preparation

Milestone 2 adds dataset planning and validation utilities without downloading datasets or training models.

- MVP class config: `configs/mvp_classes.yaml`
- YOLO dataset template: `configs/yolo_dataset_template.yaml`
- Dataset source summary: `python scripts/prepare_datasets.py`
- YOLO layout validation: `python scripts/validate_yolo_dataset.py --dataset-yaml configs/yolo_dataset_template.yaml`

The MVP focuses on 32 first-phase classes: 10 crops, 7 pests, 8 diseases, and 7 animal-related classes. Crop recognition starts as a classifier, while pest, disease, and animal-related problems are prepared for YOLO-style detection.

Read `docs/MVP_DATASET_PLAN.md` and `docs/TRAINING_PIPELINE.md` before collecting or annotating data.

## Milestone 3: MVP Dataset Manifest

Milestone 3 adds a dataset manifest for the 32 MVP classes without downloading datasets, training models, or adding heavy files.

- MVP dataset manifest: `configs/mvp_dataset_manifest.yaml`
- Manifest documentation: `docs/MVP_DATASET_MANIFEST.md`
- Manifest validation: `python scripts/validate_manifest.py`

Every MVP class is marked for license review and custom data planning. The manifest is a planning checklist only; it does not make the project training-ready by itself.

## Milestone 4: Dataset Acquisition Tooling

Milestone 4 adds safe dataset acquisition and preparation tooling without downloading datasets, training models, or adding heavy files.

- Download source config: `configs/download_sources.yaml`
- Dry-run source inspection: `python scripts/download_sources.py --source plantvillage --dry-run`
- YOLO build planning: `python scripts/build_yolo_dataset.py`
- Data storage notes: `data/README.md`
- Acquisition workflow: `docs/DATA_ACQUISITION_WORKFLOW.md`

Real downloads remain disabled until source licenses, URLs, class mappings, and annotation formats are approved.

## Milestone 5: Approved Dataset Downloads

Milestone 5 adds real download support for approved sources only: PlantVillage and IP102. Default behavior remains dry-run, and real downloads require `--confirm`.

- PlantVillage downloader: `scripts/download_plantvillage.py`
- IP102 downloader: `scripts/download_ip102.py`
- Unified downloader: `scripts/download_sources.py`
- Dataset inventory: `python scripts/dataset_inventory.py`
- Real dataset notes: `docs/REAL_DATASETS.md`

Dry-run examples:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
python scripts/download_sources.py --source ip102 --dry-run
```

Confirmed downloads write to `datasets/raw/plantvillage` and `datasets/raw/ip102`, which remain ignored by Git.

## Milestone 6: Dataset Extraction and Inspection

Milestone 6 adds safe raw dataset inspection and IP102 extraction tooling without training models or committing datasets.

- IP102 extraction dry-run: `python scripts/extract_ip102.py --dry-run`
- IP102 extraction after approval: `python scripts/extract_ip102.py --confirm`
- Raw dataset inspection: `python scripts/inspect_datasets.py`
- Inspection workflow: `docs/DATASET_INSPECTION.md`

PlantVillage is treated as a classification dataset. IP102 is inspected for classification labels and VOC2007 detection images/XML annotations. The next step is label mapping and conversion into project training formats.

## Milestone 7: Approved Label Mapping

Milestone 7 records approved source-label to MVP-class mappings without training models, moving dataset files, or changing API behavior.

- Approved label mapping: `configs/label_mapping.yaml`
- Label mapping review: `docs/LABEL_MAPPING_REVIEW.md`
- Mapping validation: `python scripts/validate_label_mapping.py`

Only safe PlantVillage disease and IP102 pest mappings are approved. Weak mappings such as plant hopper to Whitefly, bacterial spot to Bacterial Blight, and non-exact Downy Mildew matches remain rejected.

## Milestone 8: Disease Classifier Dataset

Milestone 8 adds a dry-run-first PlantVillage dataset builder for the MVP disease classifier. It does not train models, modify API files, or commit dataset artifacts.

- Disease classifier config: `configs/disease_classifier.yaml`
- Dataset builder dry-run: `python scripts/build_disease_classifier_dataset.py --dry-run`
- Dataset builder after approval: `python scripts/build_disease_classifier_dataset.py --confirm`
- Dataset notes: `docs/DISEASE_CLASSIFIER_DATASET.md`

The builder reads approved PlantVillage mappings from `configs/label_mapping.yaml`, scans `datasets/raw/plantvillage/raw/color`, and writes a `train/` and `val/` image-classification layout under `datasets/processed/disease_classifier` only when `--confirm` is used.

## Milestone 9: Disease Classifier Training

Milestone 9 adds PyTorch and torchvision training/evaluation scripts for an EfficientNet-B0 disease classifier. It does not train automatically or modify API files.

- Training config: `configs/training.yaml`
- Training script help: `python scripts/train_disease_classifier.py --help`
- Evaluation script help: `python scripts/evaluate_disease_classifier.py --help`
- Training guide: `docs/TRAINING_GUIDE.md`

Training uses `datasets/processed/disease_classifier`, auto-detects class folders, and writes checkpoints and metrics under `models/disease_classifier/` when run manually.

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
