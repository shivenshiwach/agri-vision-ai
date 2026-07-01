# Dataset Research

This document tracks planned dataset sources. Do not download datasets until licenses, storage paths, and class mappings are reviewed.

## Planned Sources

- PlantVillage: useful for crop disease classification baselines, especially leaf disease imagery.
- IP102: useful for insect pest classification research and pest label expansion.
- Roboflow: useful for exploring public object detection datasets and annotation formats.
- Wildlife datasets: useful for future animal intrusion detection, depending on species coverage and field camera relevance.

## Research Criteria

- License and redistribution restrictions.
- Supported classes and overlap with `configs/classes.yaml`.
- Annotation type: classification labels, bounding boxes, segmentation masks, or keypoints.
- Image quality, field realism, crop region, and lighting variation.
- Dataset size, class balance, and duplicate risk.
- Required conversion work for YOLO, classification, or custom inference formats.

## Notes

Dataset metadata should be documented before any files are downloaded. Heavy assets must stay out of Git and should be stored under ignored paths such as `datasets/` or external storage.
