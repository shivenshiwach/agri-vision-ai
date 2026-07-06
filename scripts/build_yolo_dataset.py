import argparse
import sys
from pathlib import Path

import yaml


DEFAULT_DATASET_YAML = Path("configs/yolo_dataset_template.yaml")
DEFAULT_OUTPUT_DIR = Path("datasets/yolo_detector")


def load_dataset_yaml(dataset_yaml: Path) -> dict:
    with dataset_yaml.open("r", encoding="utf-8") as dataset_file:
        config = yaml.safe_load(dataset_file)

    if not isinstance(config, dict):
        raise ValueError(f"{dataset_yaml} must contain a YAML mapping.")
    if not isinstance(config.get("names"), dict):
        raise ValueError(f"{dataset_yaml} must contain a names mapping.")

    return config


def detector_classes(dataset_config: dict) -> list[str]:
    names = {int(class_id): class_name for class_id, class_name in dataset_config["names"].items()}
    class_ids = sorted(names)
    expected_ids = list(range(len(class_ids)))
    if class_ids != expected_ids:
        raise ValueError(f"YOLO class ids must be contiguous from 0, got {class_ids}.")
    return [str(names[class_id]) for class_id in class_ids]


def print_expected_layout(output_dir: Path) -> None:
    print("\nExpected future YOLO output layout:")
    for relative_path in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        print(f"  - {output_dir / relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print planned YOLO detector classes. This placeholder does not move or copy files."
    )
    parser.add_argument("--dataset-yaml", type=Path, default=DEFAULT_DATASET_YAML, help="Path to YOLO dataset YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Planned YOLO dataset directory.")
    args = parser.parse_args()

    try:
        dataset_config = load_dataset_yaml(args.dataset_yaml)
        planned_classes = detector_classes(dataset_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("YOLO dataset build plan")
    print("=======================")
    print("Mode: planning only")
    print("No images or labels were moved, copied, converted, or created.")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Planned output root: {args.output_dir}")
    print(f"Detector class count: {len(planned_classes)}")

    print("\nPlanned YOLO detector classes:")
    for class_id, class_name in enumerate(planned_classes):
        print(f"  {class_id}: {class_name}")

    print_expected_layout(args.output_dir)
    print("\nNext step after license approval: map approved source labels to these class IDs before building files.")


if __name__ == "__main__":
    main()
