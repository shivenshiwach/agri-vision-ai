import argparse
import csv
import sys
from pathlib import Path

from disease_dataset_registry import DEFAULT_REGISTRY_PATH, ReportRow, run_inspection


def format_label(value: str | None) -> str:
    return value if value else "UNMAPPED"


def print_table(rows: list[ReportRow]) -> None:
    if not rows:
        print("Detailed label report: none")
        return

    print("Detailed label report")
    print("---------------------")
    print(f"{'source_dataset':<30} {'original_label':<48} {'normalized_label':<34} {'images':>8} status")
    print("-" * 136)
    for row in sorted(rows, key=lambda item: (item.source_dataset.lower(), item.original_label.lower())):
        source = row.source_dataset[:29]
        original = row.original_label[:47]
        normalized = format_label(row.normalized_label)[:33]
        print(f"{source:<30} {original:<48} {normalized:<34} {row.image_count:>8} {row.status}")


def print_duplicate_groups(title: str, groups: list[tuple[str, str, list[str]]]) -> None:
    print(title)
    print("-" * len(title))
    if not groups:
        print("  - none")
        return

    for source_key, normalized_label, original_labels in groups:
        print(f"  - {source_key}: {normalized_label} <- {', '.join(original_labels)}")


def write_report_csv(path: Path, rows: list[ReportRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["source_dataset", "source_key", "original_label", "normalized_label", "image_count", "status"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.source_key, item.original_label.lower())):
            writer.writerow(
                {
                    "source_dataset": row.source_dataset,
                    "source_key": row.source_key,
                    "original_label": row.original_label,
                    "normalized_label": row.normalized_label or "",
                    "image_count": row.image_count,
                    "status": row.status,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect enabled disease classification datasets and report normalized labels. Dry-run is the default."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Disease dataset registry YAML.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset key to inspect. May be supplied more than once. Defaults to all enabled datasets.",
    )
    parser.add_argument("--include-disabled", action="store_true", help="Inspect disabled registry entries too.")
    parser.add_argument("--report-csv", type=Path, default=None, help="Optional path for the detailed CSV report.")
    parser.add_argument("--dry-run", action="store_true", help="Read-only inspection mode. This is the default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = run_inspection(
            registry_path=args.registry,
            dataset_keys=args.dataset,
            include_disabled=args.include_disabled,
            mode="dry-run",
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("Multi-dataset disease inspection")
    print("================================")
    print("Mode: dry-run")
    print(f"Registry: {result.registry_path}")
    print(f"Taxonomy: {result.taxonomy_path}")
    print(f"Label aliases: {result.alias_path}")
    print("No images were copied. No training was started.")
    print()

    print("Dataset summary")
    print("---------------")
    print(f"{'key':<24} {'status':<22} {'labels':>8} {'mapped':>8} {'unmapped':>9} {'images':>10} root")
    print("-" * 118)
    for inspection in result.datasets:
        mapped = sum(1 for row in inspection.rows if row.status == "mapped")
        unmapped = sum(1 for row in inspection.rows if row.status in {"unmapped", "non_disease"})
        images = sum(row.image_count for row in inspection.rows)
        print(
            f"{inspection.source_key:<24} {inspection.status:<22} {len(inspection.rows):>8} "
            f"{mapped:>8} {unmapped:>9} {images:>10} {inspection.root}"
        )
        if inspection.missing_class_roots:
            missing_roots = ", ".join(str(path) for path in inspection.missing_class_roots[:5])
            extra = len(inspection.missing_class_roots) - 5
            suffix = f", ... {extra} more" if extra > 0 else ""
            print(f"  missing class roots: {missing_roots}{suffix}")
    print()

    print_table(result.rows)
    print()

    source_duplicate_groups = []
    normalized_duplicate_groups = []
    for inspection in result.datasets:
        for duplicate_key, original_labels in inspection.duplicate_source_labels.items():
            source_duplicate_groups.append((inspection.source_key, duplicate_key, original_labels))
        for normalized_label, original_labels in inspection.duplicate_normalized_labels.items():
            normalized_duplicate_groups.append((inspection.source_key, normalized_label, original_labels))

    print_duplicate_groups("Duplicate source labels", source_duplicate_groups)
    print()
    print_duplicate_groups("Duplicate normalized labels", normalized_duplicate_groups)
    print()

    print("Unmapped or non-disease labels")
    print("------------------------------")
    unmapped_rows = [row for row in result.rows if row.status != "mapped"]
    if not unmapped_rows:
        print("  - none")
    else:
        for row in sorted(unmapped_rows, key=lambda item: (item.source_key, item.original_label.lower())):
            print(f"  - {row.source_key}: {row.original_label} ({row.status}, {row.image_count} images)")
    print()

    print("Missing canonical disease labels")
    print("--------------------------------")
    if not result.missing_canonical_labels:
        print("  - none")
    else:
        for label in result.missing_canonical_labels:
            print(f"  - {label}")

    if result.alias_book.duplicate_aliases:
        print()
        print("Duplicate alias conflicts")
        print("-------------------------")
        for alias, canonical_labels in result.alias_book.duplicate_aliases.items():
            print(f"  - {alias}: {', '.join(canonical_labels)}")

    if args.report_csv:
        write_report_csv(args.report_csv, result.rows)
        print()
        print(f"Wrote CSV report: {args.report_csv}")


if __name__ == "__main__":
    main()
