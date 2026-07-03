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
| Intended use | Crop disease classification baseline and label-mapping research for MVP disease classes. |
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
| Intended use | Pest classifier baseline, pest detector pretraining, and MVP pest label research. |
| Download path | `datasets/raw/ip102` |
| Download command | `python scripts/download_sources.py --source ip102 --confirm` |

Limitations:

- IP102 class names are not guaranteed to match local MVP pest labels exactly.
- Many classes have long-tailed distribution and may need rebalancing.
- Bounding boxes cover only part of the dataset.
- Farm damage symptoms may still require custom Maharashtra field annotations.
- The downloader requires `gdown` on the training server for the official Google Drive folder.

## Safety Rules

- Dry-run first:

```bash
python scripts/download_sources.py --source plantvillage --dry-run
python scripts/download_sources.py --source ip102 --dry-run
```

- Download only on an approved server after license and storage checks.
- Keep `datasets/` out of Git.
- If a destination already exists, the downloader asks before overwriting it.
- Run `python scripts/dataset_inventory.py` after downloads to summarize local dataset folders.
