# Full PlantVillage Training

Milestone 12 makes the full PlantVillage ImageFolder classifier the main training path. The scripts are reusable for any dataset with `train/` and `val/` folders, but the defaults target the processed full PlantVillage dataset.

No training runs automatically. Checkpoints and generated metrics stay under ignored model output paths.

## Dataset

Default dataset root:

```text
datasets/processed/full_plantvillage_classifier
```

Expected layout:

```text
datasets/processed/full_plantvillage_classifier/
  train/
    <class-name>/
      <image-files>
  val/
    <class-name>/
      <image-files>
```

Build the dataset with the full PlantVillage builder:

```bash
python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
python scripts/build_full_plantvillage_classifier_dataset.py --confirm
```

## Defaults

Training defaults live in:

```text
configs/full_plantvillage_training.yaml
```

```yaml
dataset_path: datasets/processed/full_plantvillage_classifier
output_dir: models/full_plantvillage_classifier
model_name: efficientnet_b0
image_size: 224
batch_size: 64
epochs: 20
learning_rate: 0.001
optimizer: adam
loss: cross_entropy
num_workers: 4
seed: 42
pretrained: true
```

The training script auto-detects classes from the `train/` folder and verifies that the `val/` class folders match exactly. Device selection uses CUDA first, then Apple MPS when available, then CPU.

## Training

Show options:

```bash
python scripts/train_image_classifier.py --help
```

Run training manually:

```bash
python scripts/train_image_classifier.py --config configs/full_plantvillage_training.yaml
```

Useful overrides:

```bash
python scripts/train_image_classifier.py --epochs 30 --batch-size 32
python scripts/train_image_classifier.py --dataset-path datasets/processed/full_plantvillage_classifier --output-dir models/full_plantvillage_classifier
python scripts/train_image_classifier.py --no-pretrained
```

Training writes:

```text
models/full_plantvillage_classifier/
  best.pt
  last.pt
  classes.json
  history.json
```

## Evaluation

Show options:

```bash
python scripts/evaluate_image_classifier.py --help
```

Evaluate the default validation split:

```bash
python scripts/evaluate_image_classifier.py --config configs/full_plantvillage_training.yaml
```

Evaluate a specific checkpoint:

```bash
python scripts/evaluate_image_classifier.py --checkpoint models/full_plantvillage_classifier/last.pt
```

The evaluation script prints overall accuracy, per-class accuracy, and writes:

```text
models/full_plantvillage_classifier/confusion_matrix.png
```
