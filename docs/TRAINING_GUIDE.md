# Training Guide

Milestone 9 adds PyTorch training and evaluation scripts for the MVP disease classifier. The scripts do not modify API files and should be run only after `datasets/processed/disease_classifier` has been built and reviewed.

## Disease Classifier Inputs

The expected dataset layout is the image-classification folder structure produced by `scripts/build_disease_classifier_dataset.py`:

```text
datasets/processed/disease_classifier/
  train/
    <class name>/
      image files
  val/
    <class name>/
      image files
```

Class names are auto-detected from the `train/` folders. The `val/` folders must match the same class names.

## Configuration

Default training settings live in `configs/training.yaml`:

```yaml
dataset_path: datasets/processed/disease_classifier
output_dir: models/disease_classifier
model_name: efficientnet_b0
image_size: 224
batch_size: 64
epochs: 20
learning_rate: 0.001
optimizer: adam
loss: cross_entropy
num_workers: 4
seed: 42
```

The scripts use EfficientNet-B0 from `torchvision`. Training starts from ImageNet pretrained weights by default, uses `CrossEntropyLoss`, and runs on CUDA or Apple MPS when available. Otherwise it falls back to CPU.

## Training

Review the dataset first:

```bash
python scripts/build_disease_classifier_dataset.py --dry-run
```

Create the processed dataset only after review:

```bash
python scripts/build_disease_classifier_dataset.py --confirm
```

Train the classifier:

```bash
python scripts/train_disease_classifier.py
```

Useful overrides:

```bash
python scripts/train_disease_classifier.py --epochs 30 --batch-size 32
python scripts/train_disease_classifier.py --dataset-path datasets/processed/disease_classifier --output-dir models/disease_classifier
```

Training writes:

```text
models/disease_classifier/
  best.pt
  last.pt
  classes.json
  history.json
```

`best.pt` is overwritten only when validation accuracy improves. `last.pt` is overwritten every epoch and includes optimizer state for the latest checkpoint. `history.json` records epoch metrics, and `classes.json` stores the class order.

## Evaluation

Evaluate the best checkpoint on the validation split:

```bash
python scripts/evaluate_disease_classifier.py
```

Evaluation reports overall accuracy, per-class accuracy, and writes:

```text
models/disease_classifier/confusion_matrix.png
```

Use a different checkpoint or split when needed:

```bash
python scripts/evaluate_disease_classifier.py --checkpoint models/disease_classifier/last.pt
python scripts/evaluate_disease_classifier.py --split test
```

## Notes

- Do not commit datasets, model checkpoints, or generated confusion matrices.
- Keep `models/disease_classifier/classes.json` with the checkpoint when moving artifacts between machines.
- The disease classifier predicts disease labels only. It does not replace crop routing, object detection, or field severity estimation.
