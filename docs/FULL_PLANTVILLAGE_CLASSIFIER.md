# Full PlantVillage Classifier Dataset

Milestone 11 adds a dry-run-first builder for a full PlantVillage image-classification dataset. It does not train models, modify API files, commit dataset artifacts, or change taxonomy files.

## Scope

This dataset uses every non-empty class folder available under:

```text
datasets/raw/plantvillage/raw/color
```

It is broader than the MVP disease classifier dataset. The MVP builder uses only approved PlantVillage labels mapped to project disease classes from the PDF/taxonomy workflow. This full PlantVillage builder keeps the original PlantVillage class folders and includes all available labels, including labels that are not currently mapped to MVP disease classes.

Use this dataset for broader pretraining, baseline experiments, and representation learning before narrowing back to the project disease labels.

PDF/MVP label mapping will be applied later only where PlantVillage labels match approved project labels. Unmatched PlantVillage labels should not be treated as validated project classes.

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

## Configuration

Training defaults for this dataset are recorded in:

```text
configs/full_plantvillage_classifier.yaml
```

```yaml
dataset_path: datasets/processed/full_plantvillage_classifier
image_size: 224
batch_size: 64
epochs: 20
model_name: efficientnet_b0
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
