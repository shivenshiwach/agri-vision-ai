# Real Datasets

Milestone 5 enables guarded download tooling for approved dataset sources only. Real downloads still require an explicit `--confirm` flag and should be run on a server with enough storage.

## PlantVillage

| Field | Value |
| --- | --- |
| Dataset name | PlantVillage |
| Official source | `https://github.com/spMohanty/PlantVillage-Dataset.git` |
| Reference source | `https://huggingface.co/datasets/mohanty/PlantVillage` |
| License | CC BY-SA 3.0, based on the Hugging Face dataset card. Verify before redistribution or commercial use. |
| Classes covered | Healthy and diseased plant leaf images across 14 crop species and 26 diseases. |
| Intended use | Full PlantVillage classifier training and crop disease classification baselines. |
| Download path | `datasets/raw/plantvillage` |
| Download command | `python scripts/download_sources.py --source plantvillage --confirm` |

Limitations:

- Images are mostly controlled leaf images and may not represent Maharashtra field conditions.
- It is not a pest, animal, or field-intrusion dataset.
- Crop and disease labels must be mapped carefully before use.
- Public data should be combined with local field data before production use.

## IP102

| Field | Value |
| --- | --- |
| Dataset name | IP102 |
| Official source | `https://github.com/xpwu95/IP102` |
| Official download URL | `https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing` |
| License | Free for academic usage according to the official repository. Other uses require contacting the dataset author. |
| Classes covered | 102 insect pest categories with more than 75,000 images; about 19,000 images include bounding boxes. |
| Intended use | Pest classifier baseline, pest detector pretraining, and pest label research against the full taxonomy. |
| Download path | `datasets/raw/ip102` |
| Download command | `python scripts/download_sources.py --source ip102 --confirm` |

Limitations:

- IP102 class names are not guaranteed to match local taxonomy pest labels exactly.
- Many classes have long-tailed distribution and may need rebalancing.
- Bounding boxes cover only part of the dataset.
- Farm damage symptoms may still require custom Maharashtra field annotations.
- The downloader requires `gdown` on the training server for the official Google Drive folder.

## Disease Dataset Expansion

Additional disease-classification sources are configured for dry-run planning and local inspection:

| Source key | Intended local path | Status |
| --- | --- | --- |
| `plantdoc` | `datasets/raw/plantdoc` | Dry-run source entry and registry support. Real download not implemented. |
| `plantdoc_plus` | `datasets/raw/plantdoc_plus` | Dry-run source entry and registry support. Real download not implemented. |
| `plantnet` | `datasets/raw/plantnet_disease` | Registry support for approved disease-labeled exports only. |
| `kaggle_field_diseases` | `datasets/raw/kaggle_field_diseases` | Registry support for approved per-dataset Kaggle exports. |
| `custom_field_diseases` | `datasets/raw/custom_field_diseases` | Local staging support for project-owned field images. |

Use the milestone 13 inspector before any training decision:

```bash
python scripts/inspect_disease_datasets.py --dry-run
```

The inspector reports original labels, normalized labels, image counts, duplicate labels, unmapped labels, and missing canonical disease labels. The optional normalized-copy builder is dry-run by default and copies images only with `--confirm`.

## Safety Rules

- Dry-run first:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
python scripts/download_sources.py --source ip102 --dry-run
python scripts/download_sources.py --source plantdoc --dry-run
```

- Download only on an approved server after license and storage checks.
- Keep `datasets/` out of Git.
- If a destination already exists, the downloader asks before overwriting it.
- Run `python scripts/dataset_inventory.py` after downloads to summarize local dataset folders.

## Required Source Metadata Before Confirmed Download

Before any `--confirm` download or local staging action, record the source review in docs or a tracked manifest. Minimum required fields:

| Field | Required detail |
| --- | --- |
| Source name | Human-readable dataset name and source key. |
| Official URL | Original download or repository URL. |
| Reference URL | Dataset card, paper, repository, or documentation link. |
| Version | Git commit, release tag, export date, archive checksum, or access date when no version exists. |
| License | License name, usage restrictions, redistribution limits, and commercial-use status. |
| Approved use | Classification, detection, validation-only, pretraining, or research-only. |
| Local path | Ignored path under `datasets/raw/...` or another approved external mount. |
| Label mapping | Mapping notes from source labels to project taxonomy or YOLO class IDs. |
| Annotation format | Class folders, YOLO txt, VOC XML, COCO JSON, segmentation masks, or mixed format. |
| Known risks | Duplicates, class imbalance, lab-only images, weak field realism, missing labels, or license uncertainty. |

If any required field is unknown, keep the source in dry-run or local-inspection mode only.
