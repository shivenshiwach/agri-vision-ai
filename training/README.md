# Training

This directory is reserved for future model training workflows.

No datasets, checkpoints, experiment runs, or generated model artifacts should be committed here. Keep training code small and reproducible, and store heavy files under ignored directories such as `datasets/`, `runs/`, `models/`, or `outputs/`.

Planned training work:

1. Finalize crop, pest, disease, and animal taxonomy.
2. Collect and normalize dataset metadata.
3. Prepare dataset conversion scripts for YOLO or classification formats.
4. Train separate baseline models for crop, pest, disease, and animal recognition.
5. Export lightweight inference artifacts for the API layer.
