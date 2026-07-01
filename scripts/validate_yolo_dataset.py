import argparse
import sys
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def load_dataset_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def count_classes(names) -> int:
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    return 0


def resolve_split_path(config_path: Path, config: dict, split: str) -> Path | None:
    split_value = config.get(split)
    if not split_value:
        return None
    if isinstance(split_value, list):
        raise ValueError(f"{split} must be a directory path, not a list")

    root = Path(config.get("path", config_path.parent))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return (root / split_path).resolve()


def labels_dir_for_images_dir(images_dir: Path) -> Path:
    path_text = str(images_dir)
    image_segment = f"{Path('images')}"
    label_segment = f"{Path('labels')}"
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)

    if images_dir.name in SPLITS:
        return images_dir.parent.parent / "labels" / images_dir.name

    return Path(path_text.replace(image_segment, label_segment, 1))


def image_files(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_label_file(label_path: Path, class_count: int, errors: list[str], warnings: list[str]) -> None:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        warnings.append(f"Empty label file: {label_path}")
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        values = line.split()
        if len(values) < 5:
            errors.append(f"Malformed label line in {label_path}:{line_number}")
            continue

        try:
            class_id = int(values[0])
        except ValueError:
            errors.append(f"Invalid class ID in {label_path}:{line_number}: {values[0]}")
            continue

        if class_id < 0 or class_id >= class_count:
            errors.append(f"Class ID out of range in {label_path}:{line_number}: {class_id}")


def validate_split(split: str, images_dir: Path, class_count: int, errors: list[str], warnings: list[str]) -> tuple[int, int]:
    if not images_dir.exists():
        errors.append(f"{split}: images folder does not exist: {images_dir}")
        return 0, 0
    if not images_dir.is_dir():
        errors.append(f"{split}: images path is not a directory: {images_dir}")
        return 0, 0

    labels_dir = labels_dir_for_images_dir(images_dir)
    if not labels_dir.exists():
        errors.append(f"{split}: labels folder does not exist: {labels_dir}")
        return len(image_files(images_dir)), 0
    if not labels_dir.is_dir():
        errors.append(f"{split}: labels path is not a directory: {labels_dir}")
        return len(image_files(images_dir)), 0

    images = image_files(images_dir)
    labels_checked = 0

    if not images:
        warnings.append(f"{split}: no images found in {images_dir}")

    for image_path in images:
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        if not label_path.exists():
            errors.append(f"{split}: missing label file for image: {image_path}")
            continue

        labels_checked += 1
        validate_label_file(label_path, class_count, errors, warnings)

    image_stems = {path.relative_to(images_dir).with_suffix("") for path in images}
    for label_path in sorted(labels_dir.rglob("*.txt")):
        relative_label = label_path.relative_to(labels_dir).with_suffix("")
        if relative_label not in image_stems:
            warnings.append(f"{split}: label has no matching image: {label_path}")

    return len(images), labels_checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local YOLO dataset folder layout and label files.")
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("configs/yolo_dataset_template.yaml"),
        help="Path to YOLO dataset YAML.",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.dataset_yaml)
    class_count = count_classes(config.get("names"))
    if class_count == 0:
        print("ERROR: dataset YAML must define names as a list or id-to-name mapping.")
        sys.exit(1)

    errors: list[str] = []
    warnings: list[str] = []
    total_images = 0
    total_labels = 0

    for split in SPLITS:
        try:
            images_dir = resolve_split_path(args.dataset_yaml, config, split)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if images_dir is None:
            warnings.append(f"{split}: split not configured")
            continue

        image_count, label_count = validate_split(split, images_dir, class_count, errors, warnings)
        total_images += image_count
        total_labels += label_count

    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Classes: {class_count}")
    print(f"Images found: {total_images}")
    print(f"Label files checked: {total_labels}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nYOLO dataset validation passed.")


if __name__ == "__main__":
    main()
