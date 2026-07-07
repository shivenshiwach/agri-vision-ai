# Dataset Inspection

Milestone 6 adds tools for inspecting downloaded raw datasets and safely extracting IP102 archives. These tools do not train models and should not be used to commit dataset files.

## Inspect Raw Datasets

Run:

```bash
python scripts/inspect_datasets.py
```

The inspection script is read-only. It reports whether expected raw dataset folders exist, prints class counts where available, and highlights possible matches to the full taxonomy class names.

## PlantVillage

PlantVillage is used as a classification dataset. After download, the expected class folder layout is:

```text
datasets/
  raw/
    plantvillage/
      raw/
        color/
          <class-name>/
            <image-files>
```

The inspector lists `raw/color` class folders, counts images per class, and prints possible matches against disease names from `configs/classes.yaml`.

PlantVillage is useful for disease classification baselines, but it is not a YOLO detection dataset. It also does not replace local field data.

## IP102

IP102 includes insect pest classification data and a VOC2007-style detection subset. Expected paths after download are:

```text
datasets/
  raw/
    ip102/
      classes.txt
      Classification/
        ip102_v1.1.tar
      Detection/
        VOC2007/
          Annotations.tar
          JPEGImages.tar
```

Use the extraction script in dry-run mode first:

```bash
python scripts/extract_ip102.py --dry-run
```

Real extraction requires:

```bash
python scripts/extract_ip102.py --confirm
```

The extraction script:

- extracts `Detection/VOC2007/Annotations.tar` to `Detection/VOC2007/Annotations`
- extracts `Detection/VOC2007/JPEGImages.tar` to `Detection/VOC2007/JPEGImages`
- extracts `Classification/ip102_v1.1.tar` to `Classification/ip102_v1.1`
- skips existing destinations unless `--overwrite` is passed
- prints source tar sizes and target paths before acting
- blocks unsafe tar member paths

After extraction, rerun:

```bash
python scripts/inspect_datasets.py
```

The inspector checks `classes.txt`, reports the presence of VOC2007 tar files and extracted folders, counts extracted images and XML annotation files, and prints possible matches against pest names from `configs/classes.yaml`.

## Multi-Dataset Disease Inspection

Milestone 13 adds a separate disease-classification inspector:

```bash
python scripts/inspect_disease_datasets.py --dry-run
```

This inspector reads `configs/disease_dataset_registry.yaml` and `data/taxonomy/disease_label_aliases.csv`. It reports source dataset, original label, normalized label, image count, duplicate source labels, duplicate normalized labels, unmapped labels, non-disease labels, and canonical disease labels missing from the enabled sources.

The script is read-only and does not copy images or train models. Use a CSV report only when needed:

```bash
python scripts/inspect_disease_datasets.py --report-csv outputs/disease_label_report.csv
```

The optional normalized-copy planner is also dry-run by default:

```bash
python scripts/build_multi_disease_classifier_dataset.py --dry-run
```

Images are copied only when `--confirm` is supplied.

## Next Step

Possible future detector work should define label mapping and conversion rules:

- convert approved IP102 VOC annotations into the project YOLO class order
- keep all generated datasets under ignored `datasets/` paths
