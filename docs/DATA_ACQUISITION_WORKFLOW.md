# Data Acquisition Workflow

Milestone 4 adds safe dataset acquisition tooling only. It does not download datasets, train models, or add heavy files.

## 1. Local Planning

Local development keeps only scripts, configs, taxonomy files, and documentation in Git.

- Review full taxonomy classes in `configs/classes.yaml`.
- Review source plans in `configs/download_sources.yaml`.
- Run dry-run checks with `scripts/download_sources.py`.
- Review detector class order with `scripts/build_yolo_dataset.py`.

Example:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
python scripts/build_yolo_dataset.py
```

## 2. GitHub Sync

Commit only lightweight files:

- source code
- configs
- docs
- taxonomy CSV files
- validation and preparation scripts

Do not commit raw datasets, annotation exports, training runs, checkpoints, or model artifacts.

## 3. Server Pull

On the training server:

1. Pull the GitHub repo.
2. Create or mount ignored dataset storage such as `datasets/raw`.
3. Confirm source licenses, URLs, class mappings, and annotation formats.
4. Enable real download code only after approval.

The current `scripts/download_sources.py` intentionally refuses real downloads.

## 4. Server Dataset Download

After approval, datasets should be downloaded on the server, not into the Git repo history.

Expected raw layout:

```text
datasets/
  raw/
    plantvillage/
    ip102/
    plantdoc/
    roboflow/
    inaturalist/
    open_images/
```

Each downloaded source should have recorded metadata:

- source URL
- dataset version or export date
- license
- class mapping notes
- annotation format
- known quality issues

## 5. YOLO Preparation

After source review and label mapping, convert detector data into the YOLO layout:

```text
datasets/
  yolo_detector/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Class IDs must match the detector class order printed by:

```bash
python scripts/build_yolo_dataset.py
```

## 6. Validation Before Training

Before training any detector, run:

```bash
python scripts/validate_yolo_dataset.py --dataset-yaml configs/yolo_dataset_template.yaml
```

Fix missing labels, invalid class IDs, duplicate leakage, and split problems before training.

## 7. Train After Approval

Training starts only after:

- source licenses are approved
- raw data is stored outside Git
- annotations are converted and validated
- train, validation, and test splits are reviewed
- baseline model goals are agreed

Until then, the API remains a mock inference scaffold.
