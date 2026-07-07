import argparse
import csv
import shutil
import sys
from pathlib import Path

from disease_dataset_registry import (
    DEFAULT_REGISTRY_PATH,
    ReportRow,
    iter_row_images,
    load_yaml_mapping,
    output_image_name,
    run_inspection,
    safe_slug,
)


def mapped_rows(rows: list[ReportRow]) -> list[ReportRow]:
    return [row for row in rows if row.status == "mapped" and row.normalized_label and row.image_count > 0]


def output_root_from_registry(registry_path: Path) -> Path:
    registry = load_yaml_mapping(registry_path)
    return Path(str(registry.get("output_root", "datasets/processed/multi_disease_classifier")))


def class_dir_name(canonical_label: str) -> str:
    return safe_slug(canonical_label)


def print_plan(output_root: Path, rows: list[ReportRow], confirm: bool, overwrite: bool) -> None:
    class_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for row in rows:
        assert row.normalized_label is not None
        class_counts[row.normalized_label] = class_counts.get(row.normalized_label, 0) + row.image_count
        dataset_counts[row.source_key] = dataset_counts.get(row.source_key, 0) + row.image_count

    print("Multi-dataset disease classifier preparation plan")
    print("=================================================")
    print(f"Mode: {'copy files' if confirm else 'dry-run'}")
    print(f"Output root: {output_root}")
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
            print(f"  - {label}: {count}")
    print()
    print(f"Total normalized disease classes: {len(class_counts)}")
    print(f"Total images planned: {sum(class_counts.values())}")

    if not confirm:
        print()
        print("No files were copied. Re-run with --confirm to materialize the normalized dataset.")


def prepare_output(output_root: Path, rows: list[ReportRow], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite with --confirm to rebuild it.")
        shutil.rmtree(output_root)

    labels = sorted({row.normalized_label for row in rows if row.normalized_label})
    for label in labels:
        (output_root / "all" / class_dir_name(label)).mkdir(parents=True, exist_ok=True)


def write_classes_csv(output_root: Path, rows: list[ReportRow]) -> None:
    labels = sorted({row.normalized_label for row in rows if row.normalized_label})
    with (output_root / "classes.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["class_dir", "canonical_label"])
        writer.writeheader()
        for label in labels:
            writer.writerow({"class_dir": class_dir_name(label), "canonical_label": label})


def copy_images(output_root: Path, rows: list[ReportRow], image_extensions: set[str]) -> int:
    copied_count = 0
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "source_dataset",
                "source_key",
                "original_label",
                "normalized_label",
                "source_path",
                "destination_path",
            ],
        )
        writer.writeheader()

        for row in rows:
            assert row.normalized_label is not None
            destination_dir = output_root / "all" / class_dir_name(row.normalized_label)
            for class_dir, image_path in iter_row_images(row, image_extensions):
                destination = destination_dir / output_image_name(row.source_key, row.original_label, class_dir, image_path)
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
                shutil.copy2(image_path, destination)
                copied_count += 1
                writer.writerow(
                    {
                        "source_dataset": row.source_dataset,
                        "source_key": row.source_key,
                        "original_label": row.original_label,
                        "normalized_label": row.normalized_label,
                        "source_path": image_path,
                        "destination_path": destination,
                    }
                )

    write_classes_csv(output_root, rows)
    return copied_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a normalized multi-dataset disease ImageFolder-style staging dataset. Dry-run is the default."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Disease dataset registry YAML.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)

    try:
        result = run_inspection(registry_path=args.registry, dataset_keys=args.dataset, mode="copy" if args.confirm else "dry-run")
        rows = mapped_rows(result.rows)
        output_root = args.output_root or output_root_from_registry(args.registry)

        print_plan(output_root, rows, confirm=args.confirm, overwrite=args.overwrite)

        if not args.confirm:
            return
        if not rows:
            raise ValueError("No mapped disease images were found to copy.")

        prepare_output(output_root, rows, args.overwrite)
        copied_count = copy_images(output_root, rows, result.image_extensions)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print()
    print(f"Copied images: {copied_count}")
    print(f"Manifest: {output_root / 'manifest.csv'}")
    print("Multi-dataset disease classifier preparation complete.")


if __name__ == "__main__":
    main()
