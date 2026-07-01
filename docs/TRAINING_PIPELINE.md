# Training Pipeline

Milestone 2 defines the training path but does not run training.

## Local Development

- Keep taxonomy and dataset planning files in Git.
- Keep raw images, labels, model checkpoints, and experiment outputs out of Git.
- Use `scripts/prepare_datasets.py` to review planned sources and MVP classes.
- Use `scripts/validate_yolo_dataset.py` before training any YOLO detector.

## GitHub Sync

- Commit source code, configs, docs, and small taxonomy CSV files.
- Do not commit `datasets/`, `data/` raw assets, `models/`, `runs/`, or exported weights.
- Sync dataset manifests and annotation rules through Git so server-side training uses the same class order.

## Server Training

- Provision a GPU server with Python, PyTorch, Ultralytics, and storage for datasets.
- Pull the GitHub repo on the server.
- Copy or mount datasets into ignored folders such as `datasets/mvp_yolo`.
- Validate folder layout and labels before running training.

## Dataset Folder Layout

Recommended YOLO layout:

```text
datasets/
  mvp_yolo/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Each image should have a matching `.txt` label file with the same relative path under `labels/`.

## YOLO Training Flow

1. Finalize MVP detector classes and class IDs in `configs/yolo_dataset_template.yaml`.
2. Collect or export images and labels into the YOLO folder layout.
3. Run `python scripts/validate_yolo_dataset.py --dataset-yaml configs/yolo_dataset_template.yaml`.
4. Train a baseline detector with Ultralytics on the server.
5. Review precision, recall, confusion, per-class performance, and failure cases.
6. Add missing field data and rebalance classes before scaling.

Example future command:

```bash
yolo detect train data=configs/yolo_dataset_template.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Do not run this until datasets and annotation quality are ready.

## Model Export

After a model passes validation:

- Save checkpoints in ignored `runs/` or `models/` folders.
- Export deployment artifacts such as `.pt`, `.onnx`, or TensorRT engines only on the training server.
- Keep exported artifacts out of Git and register their paths in `configs/model_config.yaml`.

## API Integration

- Load model paths and thresholds from `configs/model_config.yaml`.
- Keep the FastAPI response schema stable.
- Add preprocessing for image size, color format, and confidence thresholds.
- Return detections, severity estimates, confidence scores, and farmer-facing recommendations.
- Preserve the current mock API until a validated model artifact is available.
