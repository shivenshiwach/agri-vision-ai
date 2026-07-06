import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


DEFAULT_MAPPING_PATH = Path("data/taxonomy/dataset_mapping.csv")
DEFAULT_TAXONOMY_PATH = Path("configs/classes.yaml")


def load_mapping(mapping_path: Path) -> list[dict[str, str]]:
    with mapping_path.open("r", encoding="utf-8", newline="") as mapping_file:
        return list(csv.DictReader(mapping_file))


def load_taxonomy_classes(taxonomy_path: Path) -> dict[str, list[str]]:
    with taxonomy_path.open("r", encoding="utf-8") as taxonomy_file:
        data = yaml.safe_load(taxonomy_file)
    if not isinstance(data, dict):
        raise ValueError(f"{taxonomy_path} must contain a YAML mapping.")
    return data


def split_sources(source_value: str) -> list[str]:
    return [source.strip() for source in source_value.split(";") if source.strip()]


def print_classes_by_source(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        for source in split_sources(row["suggested_source"]):
            grouped[source].append(row)

    print("Classes grouped by suggested source")
    print("=" * 35)
    for source in sorted(grouped):
        source_rows = sorted(grouped[source], key=lambda item: (item["category"], item["class_name"]))
        print(f"\n{source} ({len(source_rows)} classes)")
        for row in source_rows:
            print(f"  - {row['class_name']} [{row['category']}, {row['dataset_available']}]")


def print_taxonomy_classes(rows: list[dict[str, str]], taxonomy_classes: dict[str, list[str]]) -> None:
    rows_by_name = {row["class_name"]: row for row in rows}

    print("\nConfigured taxonomy classes")
    print("=" * 27)
    total = 0
    missing = []

    for category, class_names in taxonomy_classes.items():
        print(f"\n{category} ({len(class_names)} classes)")
        for class_name in class_names:
            row = rows_by_name.get(class_name)
            total += 1
            if row is None:
                missing.append(class_name)
                print(f"  - {class_name} [missing from dataset mapping]")
                continue

            print(
                "  - "
                f"{class_name}: {row['dataset_available']} | "
                f"{row['suggested_source']} | custom data: {row['custom_data_required']}"
            )

    print(f"\nTaxonomy total classes: {total}")
    if missing:
        print("Missing dataset mapping rows:")
        for class_name in missing:
            print(f"  - {class_name}")
    else:
        print("All taxonomy classes are present in dataset_mapping.csv.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize planned dataset sources and configured taxonomy classes. This script does not download data."
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to dataset_mapping.csv")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH, help="Path to classes.yaml")
    args = parser.parse_args()

    try:
        rows = load_mapping(args.mapping)
        taxonomy_classes = load_taxonomy_classes(args.taxonomy)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print(f"Loaded {len(rows)} mapped classes from {args.mapping}")
    print("Download status: disabled. This is a planning summary only.\n")
    print_classes_by_source(rows)
    print_taxonomy_classes(rows, taxonomy_classes)


if __name__ == "__main__":
    main()
