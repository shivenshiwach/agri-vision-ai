import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_MAPPING_PATH = Path("configs/label_mapping.yaml")
DEFAULT_MVP_PATH = Path("configs/mvp_classes.yaml")
DEFAULT_SOURCES_PATH = Path("configs/dataset_sources.yaml")

REQUIRED_MAPPING_FIELDS = {
    "source_dataset",
    "source_label",
    "target_group",
    "target_class",
    "review_status",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def expected_classes(mvp_config: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    class_to_group: dict[str, str] = {}
    classes_by_group: dict[str, list[str]] = {}

    for group, class_names in mvp_config.items():
        if not isinstance(class_names, list):
            raise ValueError(f"MVP group must be a list: {group}")

        classes_by_group[group] = []
        for class_name in class_names:
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(f"MVP group contains an empty class name: {group}")
            if class_name in class_to_group:
                raise ValueError(f"Duplicate MVP class in config: {class_name}")

            class_to_group[class_name] = group
            classes_by_group[group].append(class_name)

    return class_to_group, classes_by_group


def valid_source_names(sources_config: dict, sources_path: Path) -> set[str]:
    sources = sources_config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{sources_path} must contain a non-empty top-level sources mapping.")

    names: set[str] = set()
    for source_key, source_config in sources.items():
        if isinstance(source_key, str) and source_key.strip():
            names.add(source_key)
        if isinstance(source_config, dict):
            source_name = source_config.get("name")
            if isinstance(source_name, str) and source_name.strip():
                names.add(source_name)

    if not names:
        raise ValueError(f"{sources_path} does not define any valid source names.")

    return names


def require_non_empty_string(entry: dict, field: str, label: str, errors: list[str]) -> None:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: {field} must be a non-empty string.")


def validate_mappings(
    label_mapping: dict,
    class_to_group: dict[str, str],
    valid_sources: set[str],
) -> list[str]:
    entries = label_mapping.get("mappings")
    if not isinstance(entries, list):
        return ["Mapping config must contain a top-level mappings list."]

    errors: list[str] = []
    seen_source_labels: set[tuple[str, str]] = set()

    for index, entry in enumerate(entries, start=1):
        entry_label = f"Mapping {index}"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label}: entry must be a mapping.")
            continue

        missing = sorted(REQUIRED_MAPPING_FIELDS - set(entry))
        if missing:
            errors.append(f"{entry_label}: missing required fields: {', '.join(missing)}")
            continue

        source_dataset = entry.get("source_dataset")
        source_label = entry.get("source_label")
        target_group = entry.get("target_group")
        target_class = entry.get("target_class")
        review_status = entry.get("review_status")
        entry_label = f"{source_dataset}::{source_label} -> {target_class}"

        require_non_empty_string(entry, "source_dataset", entry_label, errors)
        require_non_empty_string(entry, "source_label", entry_label, errors)
        require_non_empty_string(entry, "target_group", entry_label, errors)
        require_non_empty_string(entry, "target_class", entry_label, errors)

        if isinstance(source_dataset, str) and source_dataset not in valid_sources:
            allowed_sources = ", ".join(sorted(valid_sources))
            errors.append(f"{entry_label}: source_dataset must be one of: {allowed_sources}.")

        if isinstance(target_class, str):
            expected_group = class_to_group.get(target_class)
            if expected_group is None:
                errors.append(f"{entry_label}: target_class is not present in configs/mvp_classes.yaml.")
            elif target_group != expected_group:
                errors.append(f"{entry_label}: target_group must be {expected_group}.")

        if review_status != "approved":
            errors.append(f"{entry_label}: review_status must be approved.")

        if isinstance(source_dataset, str) and isinstance(source_label, str):
            source_key = (source_dataset, source_label)
            if source_key in seen_source_labels:
                errors.append(f"{entry_label}: duplicate source dataset and source label mapping.")
            seen_source_labels.add(source_key)

    return errors


def ordered_classes(classes_by_group: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    for class_names in classes_by_group.values():
        ordered.extend(class_names)
    return ordered


def print_grouped_classes(
    title: str,
    class_names: set[str],
    classes_by_group: dict[str, list[str]],
) -> None:
    print(f"\n{title} ({len(class_names)}):")
    if not class_names:
        print("  - none")
        return

    for group, ordered_group_classes in classes_by_group.items():
        group_matches = [class_name for class_name in ordered_group_classes if class_name in class_names]
        if group_matches:
            print(f"  - {group}: {', '.join(group_matches)}")


def print_summary(
    mapping_path: Path,
    entries: list[dict],
    classes_by_group: dict[str, list[str]],
) -> None:
    valid_entries = [
        entry
        for entry in entries
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("source_dataset"), str)
            and isinstance(entry.get("target_class"), str)
        )
    ]
    target_counts = Counter(entry["target_class"] for entry in valid_entries)
    source_counts = Counter(entry.get("source_dataset") for entry in valid_entries)
    mapped_classes = set(target_counts)
    all_mvp_classes = set(ordered_classes(classes_by_group))
    unmapped_classes = all_mvp_classes - mapped_classes

    print(f"Label mapping: {mapping_path}")
    print(f"Mappings: {len(entries)}")

    print("\nSource dataset counts:")
    for source_name in sorted(source_counts):
        print(f"  - {source_name}: {source_counts[source_name]}")

    print_grouped_classes("Mapped MVP classes", mapped_classes, classes_by_group)
    print_grouped_classes("Unmapped MVP classes", unmapped_classes, classes_by_group)

    print("\nMapping counts by target class:")
    for class_name in ordered_classes(classes_by_group):
        if class_name in target_counts:
            print(f"  - {class_name}: {target_counts[class_name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate approved MVP label mappings. This does not train models.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to label mapping YAML.")
    parser.add_argument("--mvp", type=Path, default=DEFAULT_MVP_PATH, help="Path to MVP class YAML.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH, help="Path to dataset source YAML.")
    args = parser.parse_args()

    try:
        label_mapping = load_yaml(args.mapping)
        mvp_config = load_yaml(args.mvp)
        sources_config = load_yaml(args.sources)
        class_to_group, classes_by_group = expected_classes(mvp_config)
        valid_sources = valid_source_names(sources_config, args.sources)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    errors = validate_mappings(label_mapping, class_to_group, valid_sources)
    entries = label_mapping.get("mappings", [])

    if isinstance(entries, list):
        print_summary(args.mapping, entries, classes_by_group)

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nLabel mapping validation passed.")


if __name__ == "__main__":
    main()
