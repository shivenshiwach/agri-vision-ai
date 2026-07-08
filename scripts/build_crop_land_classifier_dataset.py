import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from crop_land_dataset_registry import (
    DEFAULT_REGISTRY_PATH,
    ReportRow,
    iter_row_images,
    load_yaml_mapping,
    output_image_name,
    run_inspection,
    safe_slug,
)


@dataclass
class PlannedImage:
    row: ReportRow
    class_dir: Path
    image_path: Path
    split: str


def mapped_rows(rows: list[ReportRow]) -> list[ReportRow]:
    return [row for row in rows if row.status == "mapped" and row.normalized_label and row.image_count > 0]


def output_root_from_registry(registry_path: Path) -> Path:
    registry = load_yaml_mapping(registry_path)
    return Path(str(registry.get("output_root", "datasets/processed/crop_land_classifier")))


def val_ratio_from_registry(registry_path: Path) -> float:
    registry = load_yaml_mapping(registry_path)
    return float(registry.get("val_ratio", 0.2))


def seed_from_registry(registry_path: Path) -> int:
    registry = load_yaml_mapping(registry_path)
    return int(registry.get("seed", 42))


def class_dir_name(canonical_label: str) -> str:
    return safe_slug(canonical_label)


def validation_count(total_count: int, val_ratio: float) -> int:
    if total_count <= 1 or val_ratio <= 0:
        return 0

    count = round(total_count * val_ratio)
    return min(max(count, 1), total_count - 1)


def build_copy_plan(rows: list[ReportRow], image_extensions: set[str], val_ratio: float, seed: int) -> list[PlannedImage]:
    grouped: dict[str, list[tuple[ReportRow, Path, Path]]] = {}
    for row in rows:
        assert row.normalized_label is not None
        grouped.setdefault(row.normalized_label, [])
        for class_dir, image_path in iter_row_images(row, image_extensions):
            grouped[row.normalized_label].append((row, class_dir, image_path))

    rng = random.Random(seed)
    planned_images: list[PlannedImage] = []
    for label, images in sorted(grouped.items()):
        shuffled_images = list(images)
        rng.shuffle(shuffled_images)
        split_count = validation_count(len(shuffled_images), val_ratio)
        val_images = shuffled_images[:split_count]
        train_images = shuffled_images[split_count:]

        for row, class_dir, image_path in train_images:
            planned_images.append(PlannedImage(row=row, class_dir=class_dir, image_path=image_path, split="train"))
        for row, class_dir, image_path in val_images:
            planned_images.append(PlannedImage(row=row, class_dir=class_dir, image_path=image_path, split="val"))

        if not images:
            continue
        if not train_images:
            raise ValueError(f"No training images would remain for label {label!r}. Add more data or lower --val-ratio.")

    return planned_images


def count_by_source(rows: list[ReportRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source_key] = counts.get(row.source_key, 0) + row.image_count
    return counts


def count_by_label(rows: list[ReportRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        assert row.normalized_label is not None
        counts[row.normalized_label] = counts.get(row.normalized_label, 0) + row.image_count
    return counts


def count_plan_by_label_and_split(planned_images: list[PlannedImage]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for planned_image in planned_images:
        assert planned_image.row.normalized_label is not None
        label_counts = counts.setdefault(planned_image.row.normalized_label, {"train": 0, "val": 0})
        label_counts[planned_image.split] += 1
    return counts


def print_plan(
    output_root: Path,
    rows: list[ReportRow],
    planned_images: list[PlannedImage],
    confirm: bool,
    overwrite: bool,
    val_ratio: float,
    seed: int,
) -> None:
    dataset_counts = count_by_source(rows)
    class_counts = count_by_label(rows)
    split_counts = count_plan_by_label_and_split(planned_images)

    print("Crop/land classifier dataset preparation plan")
    print("=============================================")
    print(f"Mode: {'copy files' if confirm else 'dry-run'}")
    print(f"Output root: {output_root}")
    print(f"Validation ratio: {val_ratio}")
    print(f"Random seed: {seed}")
    print(f"Overwrite output: {overwrite}")
    print("No model training is performed by this script.")
    print()

    print("Images by source dataset")
    print("------------------------")
    if not dataset_counts:
        print("  - none")
    else:
        for dataset_key, count in sorted(dataset_counts.items()):
            print(f"  - {dataset_key}: {count}")
    print()

    print("Images by normalized class")
    print("--------------------------")
    if not class_counts:
        print("  - none")
    else:
        for label, count in sorted(class_counts.items()):
            split = split_counts.get(label, {"train": 0, "val": 0})
            print(f"  - {label}: {count} total, {split['train']} train, {split['val']} val")
    print()
    print(f"Total normalized crop/land classes: {len(class_counts)}")
    print(f"Total images planned: {sum(class_counts.values())}")

    if not confirm:
        print()
        print("No files were copied. Re-run with --confirm to materialize the crop/land classifier dataset.")


def prepare_output(output_root: Path, planned_images: list[PlannedImage], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite with --confirm to rebuild it.")
        shutil.rmtree(output_root)

    labels = sorted({planned_image.row.normalized_label for planned_image in planned_images if planned_image.row.normalized_label})
    for split_name in ("train", "val"):
        for label in labels:
            (output_root / split_name / class_dir_name(label)).mkdir(parents=True, exist_ok=True)


def write_classes_csv(output_root: Path, planned_images: list[PlannedImage]) -> None:
    labels = sorted({planned_image.row.normalized_label for planned_image in planned_images if planned_image.row.normalized_label})
    with (output_root / "classes.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["class_dir", "canonical_label"])
        writer.writeheader()
        for label in labels:
            writer.writerow({"class_dir": class_dir_name(label), "canonical_label": label})


def copy_images(output_root: Path, planned_images: list[PlannedImage]) -> int:
    copied_count = 0
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "source_dataset",
                "source_key",
                "original_label",
                "normalized_label",
                "source_path",
                "destination_path",
            ],
        )
        writer.writeheader()

        for planned_image in planned_images:
            row = planned_image.row
            assert row.normalized_label is not None
            destination_dir = output_root / planned_image.split / class_dir_name(row.normalized_label)
            destination = destination_dir / output_image_name(row.source_key, row.original_label, planned_image.class_dir, planned_image.image_path)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
            shutil.copy2(planned_image.image_path, destination)
            copied_count += 1
            writer.writerow(
                {
                    "split": planned_image.split,
                    "source_dataset": row.source_dataset,
                    "source_key": row.source_key,
                    "original_label": row.original_label,
                    "normalized_label": row.normalized_label,
                    "source_path": planned_image.image_path,
                    "destination_path": destination,
                }
            )

    write_classes_csv(output_root, planned_images)
    return copied_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a normalized crop/land ImageFolder train/val dataset. Dry-run is the default."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Crop/land dataset registry YAML.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset key to include. May be supplied more than once. Defaults to all enabled datasets.",
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Override output root from the registry.")
    parser.add_argument("--dry-run", action="store_true", help="Print the copy plan without copying files. This is the default.")
    parser.add_argument("--confirm", action="store_true", help="Actually copy normalized images into the output root.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output root when used with --confirm.")
    parser.add_argument("--include-disabled", action="store_true", help="Allow explicitly selected disabled registry entries.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio. Defaults to registry val_ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)

    try:
        val_ratio = args.val_ratio if args.val_ratio is not None else val_ratio_from_registry(args.registry)
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("--val-ratio must be greater than or equal to 0 and less than 1.")

        seed = seed_from_registry(args.registry)
        result = run_inspection(
            registry_path=args.registry,
            dataset_keys=args.dataset,
            include_disabled=args.include_disabled,
            mode="copy" if args.confirm else "dry-run",
        )
        rows = mapped_rows(result.rows)
        output_root = args.output_root or output_root_from_registry(args.registry)
        planned_images = build_copy_plan(rows, result.image_extensions, val_ratio, seed)

        print_plan(
            output_root=output_root,
            rows=rows,
            planned_images=planned_images,
            confirm=args.confirm,
            overwrite=args.overwrite,
            val_ratio=val_ratio,
            seed=seed,
        )

        if not args.confirm:
            return
        if not planned_images:
            raise ValueError("No mapped crop/land images were found to copy.")

        prepare_output(output_root, planned_images, args.overwrite)
        copied_count = copy_images(output_root, planned_images)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print()
    print(f"Copied images: {copied_count}")
    print(f"Manifest: {output_root / 'manifest.csv'}")
    print("Crop/land classifier dataset preparation complete.")


if __name__ == "__main__":
    main()
