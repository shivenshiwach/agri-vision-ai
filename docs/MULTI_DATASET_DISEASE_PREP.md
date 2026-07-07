# Multi-Dataset Disease Classification Preparation

Milestone 13 adds a preparation path for combining disease labels across multiple datasets. It does not train models, modify API code, or change the existing full PlantVillage classifier pipeline.

## Scope

The multi-dataset disease path is driven by:

```text
configs/disease_dataset_registry.yaml
data/taxonomy/disease_label_aliases.csv
scripts/inspect_disease_datasets.py
scripts/build_multi_disease_classifier_dataset.py
```

The registry enables multiple disease sources together, including PlantVillage, PlantDoc, PlantDoc++, a disease-labeled PlantNet export, Kaggle field disease exports, and custom field disease collections.

All raw and processed image folders remain ignored by Git.

## Registry

Each source in `configs/disease_dataset_registry.yaml` defines:

- `enabled`: whether the source is included by default.
- `root`: local raw dataset root.
- `discovery.class_roots`: one or more folders whose direct children are class-label folders.
- source notes and license-review requirements.

For split-style exports, keep class folders under roots such as:

```text
datasets/raw/plantdoc/train/<class-label>/
datasets/raw/plantdoc/val/<class-label>/
datasets/raw/plantdoc/test/<class-label>/
```

For flat ImageFolder exports, use:

```text
datasets/raw/kaggle_field_diseases/<class-label>/
```

## Label Normalization

Label aliases live in:

```text
data/taxonomy/disease_label_aliases.csv
```

The inspector maps dataset-specific labels such as `Tomato___Early_blight`, `early blight`, and `Potato___Early_blight` to one canonical class, `Early Blight`.

Canonical disease classes come from:

```text
configs/classes.yaml
```

Unmapped labels stay visible in reports instead of being silently renamed. Healthy, normal, background, and unknown labels are treated as non-disease labels and excluded from normalized copy plans.

## Inspect Datasets

Run the read-only inspection:

```bash
python scripts/inspect_disease_datasets.py --dry-run
```

The report includes:

- source dataset
- original label
- normalized label
- image count
- duplicate source labels
- duplicate normalized labels
- unmapped or non-disease labels
- canonical disease labels missing from all enabled datasets

Write a CSV report when needed:

```bash
python scripts/inspect_disease_datasets.py --report-csv outputs/disease_label_report.csv
```

Inspect one or more sources:

```bash
python scripts/inspect_disease_datasets.py --dataset plantvillage --dataset plantdoc
```

## Prepare Normalized Staging Dataset

Preview the normalized copy plan:

```bash
python scripts/build_multi_disease_classifier_dataset.py --dry-run
```

Confirmed copy requires an explicit flag:

```bash
python scripts/build_multi_disease_classifier_dataset.py --confirm
```

The confirmed output layout is:

```text
datasets/processed/multi_disease_classifier/
  all/
    <normalized-class-dir>/
      <source>__<original-label>__<hash>__<filename>
  classes.csv
  manifest.csv
```

The builder does not create train/val/test splits and does not train a model. It creates only a traceable staging dataset for later review.

If the output root already exists, the builder exits unless `--overwrite` is used with `--confirm`.

## Safety Rules

- Dry-run is the default.
- No images are copied unless `--confirm` is supplied.
- The full PlantVillage classifier builder remains unchanged.
- The FastAPI app remains unchanged.
- Do not train from this staging dataset until source licenses, duplicates, unmapped labels, class balance, and split strategy are reviewed.

## Verification

Use:

```bash
python -m compileall scripts
python scripts/inspect_disease_datasets.py --help
python scripts/build_multi_disease_classifier_dataset.py --help
python scripts/inspect_disease_datasets.py --dry-run
python scripts/build_multi_disease_classifier_dataset.py --dry-run
git diff -- api
```
