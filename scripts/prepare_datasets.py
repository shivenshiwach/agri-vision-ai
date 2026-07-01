import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


DEFAULT_MAPPING_PATH = Path("data/taxonomy/dataset_mapping.csv")
DEFAULT_MVP_PATH = Path("configs/mvp_classes.yaml")


def load_mapping(mapping_path: Path) -> list[dict[str, str]]:
    with mapping_path.open("r", encoding="utf-8", newline="") as mapping_file:
        return list(csv.DictReader(mapping_file))


def load_mvp_classes(mvp_path: Path) -> dict[str, list[str]]:
    with mvp_path.open("r", encoding="utf-8") as mvp_file:
        return yaml.safe_load(mvp_file)


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


def print_mvp_classes(rows: list[dict[str, str]], mvp_classes: dict[str, list[str]]) -> None:
    rows_by_name = {row["class_name"]: row for row in rows}

    print("\nPriority MVP classes")
    print("=" * 20)
    total = 0
    missing = []

    for category, class_names in mvp_classes.items():
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

    print(f"\nMVP total classes: {total}")
    if missing:
        print("Missing MVP mappings:")
        for class_name in missing:
            print(f"  - {class_name}")
    else:
        print("All MVP classes are present in dataset_mapping.csv.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize planned dataset sources and priority MVP classes. This script does not download data."
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to dataset_mapping.csv")
    parser.add_argument("--mvp", type=Path, default=DEFAULT_MVP_PATH, help="Path to mvp_classes.yaml")
    args = parser.parse_args()

    rows = load_mapping(args.mapping)
    mvp_classes = load_mvp_classes(args.mvp)

    print(f"Loaded {len(rows)} mapped classes from {args.mapping}")
    print("Download status: disabled. This is a planning summary only.\n")
    print_classes_by_source(rows)
    print_mvp_classes(rows, mvp_classes)


if __name__ == "__main__":
    main()
