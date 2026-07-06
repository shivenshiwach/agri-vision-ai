# Full PlantVillage Classifier Dataset

The full PlantVillage builder creates the main image-classification dataset for the project. It does not train models, modify API files, commit dataset artifacts, or change taxonomy files.

## Scope

This dataset uses every non-empty class folder available under:

```text
datasets/raw/plantvillage/raw/color
```

It keeps the original PlantVillage class folder names and includes all available PlantVillage labels. A complete PlantVillage color download is expected to produce 38 classes.

Use this dataset for the main PlantVillage baseline and for reusable image-classification experiments.

## Output Layout

Confirmed builds write:

```text
datasets/processed/full_plantvillage_classifier/
  train/
    <PlantVillage class folder>/
  val/
    <PlantVillage class folder>/
```

The default validation split is `0.2`, random seed is `42`, and all images are used by default. Empty source folders are skipped.

## Dataset Configuration

Dataset defaults for this builder are recorded in:

```text
configs/full_plantvillage_classifier.yaml
```

```yaml
source_root: datasets/raw/plantvillage/raw/color
output_root: datasets/processed/full_plantvillage_classifier
val_ratio: 0.2
seed: 42
class_labels: original_plantvillage_folder_names
```

Training defaults live separately in:

```text
configs/full_plantvillage_training.yaml
```

## Commands

Preview the plan without copying files:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
```

Create the processed dataset only after review:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --confirm
```

If the output directory already exists, the build exits unless `--overwrite` is passed with `--confirm`.

After building the dataset, see `docs/FULL_PLANTVILLAGE_TRAINING.md` for training and evaluation commands.
