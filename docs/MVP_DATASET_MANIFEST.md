# MVP Dataset Manifest

Milestone 3 adds a planning manifest for the 32 MVP classes in `configs/mvp_dataset_manifest.yaml`. This milestone does not download datasets, train models, or add heavy artifacts.

The manifest is a review checklist before dataset collection starts. It records the intended source direction, license-review requirement, model target, and collection priority for each MVP class.

## Manifest Fields

Each class entry contains:

- `class_name`: Class label from `configs/mvp_classes.yaml`.
- `category`: Class category used by the dataset plan.
- `target_model`: Planned model family for the class.
- `dataset_source`: Public sources and custom collection route to review.
- `dataset_status`: One of `planned`, `public`, or `custom_needed`.
- `license_review_required`: Must remain `true` until source licenses and redistribution rules are reviewed.
- `notes`: Collection and mapping guidance for the class.
- `priority`: One of `high`, `medium`, or `low`.

For this manifest, every MVP class is marked `custom_needed`. Public datasets may still be useful for baselines, but every class needs local Maharashtra field or storage data before production use.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `planned` | Source candidates are identified, but no usable dataset has been selected. |
| `public` | A public dataset is expected to be usable after license and quality review. |
| `custom_needed` | Public data alone is insufficient, and local custom data is required before training readiness. |

## Priority Definitions

| Priority | Meaning |
| --- | --- |
| `high` | Should be included in the first serious data collection and baseline review. |
| `medium` | Important for the MVP, but can follow the highest-risk classes. |
| `low` | Keep in the MVP manifest, but do not block the first baseline if data is weak. |

## MVP Class Summary

| Group | Classes | Target Model |
| --- | ---: | --- |
| Crops | 10 | `crop_classifier` |
| Pests | 7 | `pest_detector` |
| Diseases | 8 | `disease_detector` |
| Animal-related | 7 | `animal_detector` |
| Total | 32 | Multiple |

## Review Checklist

Before downloading or collecting any data:

1. Confirm that each public source license allows the intended use.
2. Record exact dataset URLs, versions, class mappings, and annotation formats.
3. Decide whether each class starts as classification, detection, or both.
4. Keep raw images and labels outside Git under ignored paths such as `datasets/`.
5. Run `python scripts/validate_manifest.py` after any manifest change.

## Current Limitations

- No dataset has been downloaded.
- No license has been approved.
- No source has been converted to YOLO or classifier format.
- No model can be trained from this manifest alone.
- The current API still returns mock predictions.
