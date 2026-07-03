import argparse
import sys
from pathlib import Path

import yaml


DEFAULT_MVP_PATH = Path("configs/mvp_classes.yaml")
DEFAULT_OUTPUT_DIR = Path("datasets/mvp_yolo")
DETECTOR_GROUPS = ("pests", "diseases", "animals")


def load_mvp_classes(mvp_path: Path) -> dict:
    with mvp_path.open("r", encoding="utf-8") as mvp_file:
        classes = yaml.safe_load(mvp_file)

    if not isinstance(classes, dict):
        raise ValueError(f"{mvp_path} must contain a YAML mapping.")

    for group in DETECTOR_GROUPS:
        if not isinstance(classes.get(group), list):
            raise ValueError(f"{mvp_path} must contain a list for {group}.")

    return classes


def detector_classes(mvp_classes: dict) -> list[str]:
    planned_classes: list[str] = []
    for group in DETECTOR_GROUPS:
        planned_classes.extend(mvp_classes[group])
    return planned_classes


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
        description="Print planned MVP YOLO detector classes. This placeholder does not move or copy files."
    )
    parser.add_argument("--mvp-config", type=Path, default=DEFAULT_MVP_PATH, help="Path to MVP class config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Planned YOLO dataset directory.")
    args = parser.parse_args()

    try:
        mvp_classes = load_mvp_classes(args.mvp_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    planned_classes = detector_classes(mvp_classes)

    print("MVP YOLO dataset build plan")
    print("===========================")
    print("Mode: planning only")
    print("No images or labels were moved, copied, converted, or created.")
    print(f"MVP config: {args.mvp_config}")
    print(f"Planned output root: {args.output_dir}")
    print(f"Detector class count: {len(planned_classes)}")

    print("\nPlanned YOLO detector classes:")
    for class_id, class_name in enumerate(planned_classes):
        print(f"  {class_id}: {class_name}")

    print_expected_layout(args.output_dir)
    print("\nNext step after license approval: map approved source labels to these class IDs before building files.")


if __name__ == "__main__":
    main()
