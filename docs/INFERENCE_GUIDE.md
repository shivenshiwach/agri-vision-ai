# Inference Guide

Milestone 10 adds disease classifier inference for one local image at a time. It does not modify API files, retrain models, change datasets, or update taxonomy files.

## Required Files

Prediction expects the artifacts created by disease classifier training:

```text
models/disease_classifier/
  best.pt
  classes.json
```

`best.pt` contains the EfficientNet-B0 checkpoint. `classes.json` contains the class order used by the classifier head.

## Run Prediction

Use a positional image path:

```bash
python scripts/predict_disease.py image.jpg
```

Or use the explicit `--image` argument:

```bash
python scripts/predict_disease.py --image image.jpg
```

Change the number of displayed predictions:

```bash
python scripts/predict_disease.py --image image.jpg --top-k 5
```

Example output:

```text
Prediction: Early Blight
Confidence: 98.4%

Top predictions:
1. Early Blight 98.4%
2. Late Blight 1.2%
3. Rust 0.4%
```

## Preprocessing

The inference script uses the same evaluation-style preprocessing as the disease classifier:

- resize to `224 x 224`
- center crop to `224`
- convert to tensor
- normalize with ImageNet mean and standard deviation

Images are opened with Pillow and converted to RGB before preprocessing.

## Error Handling

The script exits with a clear error when:

- the image path is missing or does not exist
- `models/disease_classifier/best.pt` is missing
- `models/disease_classifier/classes.json` is missing or malformed
- `--top-k` is not a positive integer
