# Disease Classifier Dataset

Milestone 8 adds a dry-run-first builder for a PlantVillage disease classifier dataset. It does not train models, modify API files, or commit dataset artifacts.

## Source Mappings

The builder reads `configs/label_mapping.yaml` and uses only approved PlantVillage disease mappings:

| PlantVillage source label | MVP disease class |
| --- | --- |
| `Potato___Early_blight` | Early Blight |
| `Tomato___Early_blight` | Early Blight |
| `Potato___Late_blight` | Late Blight |
| `Tomato___Late_blight` | Late Blight |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Leaf Curl Virus |
| `Corn_(maize)___Common_rust_` | Rust |
| `Apple___Cedar_apple_rust` | Rust |
| `Squash___Powdery_mildew` | Powdery Mildew |
| `Cherry_(including_sour)___Powdery_mildew` | Powdery Mildew |

The expected source root is:

```text
datasets/raw/plantvillage/raw/color
```

Missing source folders are skipped with warnings so a partial local PlantVillage download can still be inspected safely.

## Output Format

Confirmed builds write an image-classification folder layout:

```text
datasets/processed/disease_classifier/
  train/
    Early Blight/
    Late Blight/
    Leaf Curl Virus/
    Rust/
    Powdery Mildew/
  val/
    Early Blight/
    Late Blight/
    Leaf Curl Virus/
    Rust/
    Powdery Mildew/
```

By default, every available image from each approved source class is used. The builder does not downsample unless `--max-per-source-class` is explicitly provided. Train/validation splitting is deterministic with random seed 42 and defaults to `--val-ratio 0.2`.

Run a dry run:

```bash
python scripts/build_disease_classifier_dataset.py --dry-run
```

Copy files only after review:

```bash
python scripts/build_disease_classifier_dataset.py --confirm
```

If the output directory already exists, the build exits unless `--overwrite` is passed.

## Classifier Scope

This dataset is for image classification, not object detection. Images are placed under class folders, and no bounding boxes or segmentation masks are produced.

PlantVillage images are leaf-focused and often crop-specific. For MVP v1, crop should be user-provided or selected before disease classification instead of inferred from this classifier. The classifier predicts the disease class among approved MVP disease labels; it should not be treated as a crop router or field-ready detector.
