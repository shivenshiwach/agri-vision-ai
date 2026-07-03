# Data Directory

This directory keeps lightweight project data only. Raw datasets, downloaded images, annotation exports, model checkpoints, and experiment outputs must not be committed to Git.

The current tracked data under `data/taxonomy/` contains small taxonomy CSV files. Heavy dataset assets should stay on a training server or external storage and remain under ignored paths.

## Expected Dataset Storage

Raw source datasets should use ignored storage such as:

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

Prepared YOLO data should use:

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

Each YOLO image should have a matching `.txt` label file with the same relative path under `labels/`. Dataset download and conversion should happen only after source licenses, URLs, class mappings, and annotation formats are approved.
