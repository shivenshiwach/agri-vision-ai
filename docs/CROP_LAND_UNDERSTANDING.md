# Crop/Land Understanding

This module is an early scaffold for image-level crop and land scene understanding. It does not download datasets, train models, or change API behavior.

The first classifier target is:

- `crop_closeup`
- `crop_field`
- `bare_soil_empty_land`
- `weed_dominant_field`
- `harvested_field`
- `waterlogged_land`
- `non_agricultural_land`
- `unknown_or_unclear`

Crop-specific prediction is a later step and should reuse the existing crop taxonomy in `data/taxonomy/crops.csv`.

## Repo Files

- `data/taxonomy/crop_land_scenes.csv` defines the scene labels.
- `configs/crop_land_dataset_registry.yaml` lists approved and planned source roots plus label mappings.
- `configs/crop_land_training.yaml` points the existing EfficientNet-B0 ImageFolder trainer at the crop/land processed dataset.
- `scripts/inspect_crop_land_datasets.py` inspects source class folders and reports mapping coverage.
- `scripts/build_crop_land_classifier_dataset.py` stages a train/val ImageFolder dataset only when `--confirm` is passed.

## Dataset Strategy

Use a ground-level scene classifier first. Public aerial and satellite datasets are useful research proxies, but they do not match farmer phone images closely enough to be the primary training source. Bare soil, harvested field, waterlogged field, and unknown/unclear behavior require project-owned or consented ground images.

## Dataset Source Review

| Dataset | Official source | License | Size | Useful labels | Usefulness | Limitations | Decision | Server prep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Custom Farmer Crop/Land Collection | Local project-owned collection | Project-owned or consented images only | TBD | All target labels | Best match for farmer-upload behavior | Requires manual collection, review, and labeling | Use now | Stage class folders under `datasets/raw/custom_crop_land/<label>/`, then run inspector and builder dry-runs. |
| PlantDoc | `https://github.com/pratikkayal/PlantDoc-Dataset` | CC BY 4.0 | 2,598 images, 13 plant species, up to 17 disease classes | Crop-visible closeups | Useful for `crop_closeup` and future crop-label references | Disease/crop closeups only; not land-scene coverage | Use later as auxiliary after source/version capture | Clone or download official repo on server; either stage accepted images under `datasets/raw/plantdoc/crop_closeup/` or keep disabled until a dedicated crop-label converter is approved. |
| DeepWeeds | `https://github.com/AlexOlsen/DeepWeeds` | Images and annotations CC BY 4.0; repo code Apache-2.0 | 17,509 images, 8 weed species plus negative | Weed presence/species | Useful auxiliary data for weed-heavy vegetation | Australia-specific weed closeups; not Indian field-level scenes | Later | Download official `images.zip` only after approval; manually stage reviewed weed images under `datasets/raw/deepweeds_crop_land/weed_dominant_field/`. |
| Open Images V7 | `https://storage.googleapis.com/openimages/web/index.html` | Annotations CC BY 4.0; listed images CC BY 2.0 with per-image verification required | About 9M images; image labels, boxes, masks, relationships, narratives | Generic crop, soil, road, building, water negatives | Useful for negative mining and generic crop/land samples | Label noise, per-image license checks, not agriculture-specific | Later, small curated sample only | Use official CSV metadata; sample human-verified labels only; verify each image license; stage reviewed images into canonical class folders. |
| Agriculture-Vision | `https://www.agriculture-vision.com/agriculture-vision-2020/dataset` | Restricted challenge terms; internal non-commercial academic use only | 21,061 aerial farmland images in the 2020 challenge subset; 4.4 GB archive | Standing water, waterway, weed cluster, planter skip | Good aerial proxy for waterlogged and weed-cluster patterns | Aerial/NIR farmland tiles, restricted terms, not phone images | Postpone unless research-only approval is explicit | Request access and accept terms; convert masks to class-folder tiles outside Git; run `--include-disabled --dataset agriculture_vision --dry-run`. |
| EuroSAT | `https://github.com/phelber/eurosat` | MIT; Sentinel terms also apply | 27,000 Sentinel-2 images, 10 land-cover classes | AnnualCrop, PermanentCrop, Pasture, Residential, Industrial, Highway, River, SeaLake, Forest | Useful sanity check for crop vs non-crop remote-sensing labels | Satellite domain mismatch; not farmer phone images | Later | Download official Zenodo RGB export; keep disabled or run as a separate remote-sensing experiment. |
| LoveDA | `https://github.com/Junjue-Wang/LoveDA` | Academic use only; commercial use prohibited | 5,987 high-resolution remote-sensing images | Agriculture, barren, water, building, road, forest | Useful proxy for agriculture/barren/water/non-agri segmentation | License limits and aerial domain mismatch | Postpone | Download from official Zenodo/Baidu after approval; convert masks to class-folder tiles before inspection. |
| LandCover.ai | `https://landcover.ai.linuxpolska.com/` | CC BY-NC-SA 4.0 | 41 orthophotos covering 216.27 sq km | Building, woodland, water, road | Useful non-agri and water proxy | Non-commercial, aerial, no crop class | Postpone | Download official v1; split to tiles; convert dominant masks to class folders if approved. |
| FloodNet | `https://github.com/BinaLab/FloodNet-Supervised_v1.0` | CDLA-Permissive | 2,343 UAV flood images | Water, flooded road/building, grass, tree | Auxiliary water/flood visual features | Flood disaster imagery, not crop-specific waterlogging | Postpone | Download official archive; convert selected water/flood labels to reviewed class folders. |
| iNatAg via AgML | `https://github.com/Project-AgML/AgML` | AgML code Apache-2.0; underlying image licenses require per-export review | More than 4M crop/weed images through AgML/iNatAg | Crop and weed species | Strong candidate for future crop-label prediction | Per-image licensing and species mapping must be documented | Later | Install AgML on server only after approval; export reviewed species samples into future crop-label folders, not directly into this scene classifier. |
| CropHarvest | `https://github.com/nasaharvest/cropharvest` | CC BY-SA 4.0 | 95,186 datapoints; crop/non-crop and multiclass labels | Crop vs non-crop from remote sensing | Useful for geospatial crop/non-crop research | Time-series remote-sensing data, not upload-image classification | Postpone | Keep out of the first image classifier; use only for a separate geospatial workflow. |

No public dataset above is sufficient by itself for `bare_soil_empty_land`, `harvested_field`, `waterlogged_land`, or `unknown_or_unclear` in real farmer images.

## First Server Experiment

Use `custom_farmer_crop_land` only for the first real experiment.

Expected source layout:

```text
datasets/raw/custom_crop_land/
  crop_closeup/
  crop_field/
  bare_soil_empty_land/
  weed_dominant_field/
  harvested_field/
  waterlogged_land/
  non_agricultural_land/
  unknown_or_unclear/
```

Class folder names may use aliases in `configs/crop_land_dataset_registry.yaml`, but canonical names are preferred.

## Server Workflow

After the scaffold is pushed and pulled on the training server:

```bash
venv/bin/python -m pytest
venv/bin/python scripts/inspect_crop_land_datasets.py --dry-run
venv/bin/python scripts/build_crop_land_classifier_dataset.py --dry-run
```

After source review and image labeling are complete:

```bash
venv/bin/python scripts/inspect_crop_land_datasets.py --dry-run
venv/bin/python scripts/build_crop_land_classifier_dataset.py --dry-run
venv/bin/python scripts/build_crop_land_classifier_dataset.py --confirm
venv/bin/python scripts/train_image_classifier.py --config configs/crop_land_training.yaml
venv/bin/python scripts/evaluate_image_classifier.py --config configs/crop_land_training.yaml
```

Use `--overwrite` with the builder only when intentionally rebuilding the processed dataset.

## Manual Approval Required

Before any public source is downloaded or enabled:

- confirm official URL and dataset version or export date
- confirm license and commercial-use limits
- confirm local ignored storage path under `datasets/raw/...`
- document label mapping from source labels to canonical crop/land labels
- review whether the data is ground-level farmer-like, aerial, satellite, or generic web imagery
- keep raw images, generated splits, checkpoints, and metrics out of Git

## Known Limitations

- The first model is image-level classification only; it does not localize crop rows, weeds, or standing water.
- Aerial/satellite sources should not be mixed with farmer phone images without an explicit experiment note.
- `unknown_or_unclear` requires deliberately collected bad, ambiguous, and out-of-domain images.
- Crop-label prediction is not implemented in this scaffold.
- API integration is intentionally not part of this module yet.
