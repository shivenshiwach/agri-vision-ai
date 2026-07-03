import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_MANIFEST_PATH = Path("configs/mvp_dataset_manifest.yaml")
DEFAULT_MVP_PATH = Path("configs/mvp_classes.yaml")

REQUIRED_FIELDS = {
    "class_name",
    "category",
    "target_model",
    "dataset_source",
    "dataset_status",
    "license_review_required",
    "notes",
    "priority",
}
ALLOWED_STATUSES = {"planned", "public", "custom_needed"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
EXPECTED_TARGET_MODELS = {
    "crops": "crop_classifier",
    "pests": "pest_detector",
    "diseases": "disease_detector",
    "animals": "animal_detector",
}
EXPECTED_CATEGORIES = {
    "crops": {"crop"},
    "pests": {"pest"},
    "diseases": {"disease"},
    "animals": {"animal", "bird", "rodent", "reptile"},
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def expected_classes(mvp_config: dict) -> dict[str, str]:
    classes: dict[str, str] = {}
    for group, class_names in mvp_config.items():
        if group not in EXPECTED_TARGET_MODELS:
            raise ValueError(f"Unexpected MVP group in config: {group}")
        if not isinstance(class_names, list):
            raise ValueError(f"MVP group must be a list: {group}")
        for class_name in class_names:
            if class_name in classes:
                raise ValueError(f"Duplicate MVP class in config: {class_name}")
            classes[class_name] = group
    return classes


def validate_dataset_source(entry: dict, errors: list[str]) -> None:
    class_name = entry.get("class_name", "<missing class_name>")
    sources = entry.get("dataset_source")
    if isinstance(sources, str):
        if not sources.strip():
            errors.append(f"{class_name}: dataset_source must not be empty.")
        return

    if not isinstance(sources, list) or not sources:
        errors.append(f"{class_name}: dataset_source must be a non-empty string or list.")
        return

    for source in sources:
        if not isinstance(source, str) or not source.strip():
            errors.append(f"{class_name}: dataset_source contains an empty source.")


def validate_manifest(manifest: dict, expected: dict[str, str]) -> list[str]:
    errors: list[str] = []
    entries = manifest.get("classes")

    if not isinstance(entries, list):
        return ["Manifest must contain a top-level classes list."]

    seen: set[str] = set()
    expected_names = set(expected)

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"Entry {index}: class entry must be a mapping.")
            continue

        missing = sorted(REQUIRED_FIELDS - set(entry))
        class_name = entry.get("class_name", f"<entry {index}>")
        if missing:
            errors.append(f"{class_name}: missing required fields: {', '.join(missing)}")
            continue

        if class_name in seen:
            errors.append(f"{class_name}: duplicate manifest class.")
        seen.add(class_name)

        group = expected.get(class_name)
        if group is None:
            errors.append(f"{class_name}: not present in configs/mvp_classes.yaml.")
            continue

        expected_model = EXPECTED_TARGET_MODELS[group]
        if entry["target_model"] != expected_model:
            errors.append(f"{class_name}: target_model must be {expected_model}.")

        allowed_categories = EXPECTED_CATEGORIES[group]
        if entry["category"] not in allowed_categories:
            allowed_text = ", ".join(sorted(allowed_categories))
            errors.append(f"{class_name}: category must be one of: {allowed_text}.")

        if entry["dataset_status"] not in ALLOWED_STATUSES:
            allowed_text = ", ".join(sorted(ALLOWED_STATUSES))
            errors.append(f"{class_name}: dataset_status must be one of: {allowed_text}.")

        if entry["priority"] not in ALLOWED_PRIORITIES:
            allowed_text = ", ".join(sorted(ALLOWED_PRIORITIES))
            errors.append(f"{class_name}: priority must be one of: {allowed_text}.")

        if entry["license_review_required"] is not True:
            errors.append(f"{class_name}: license_review_required must be true.")

        if not isinstance(entry["notes"], str) or not entry["notes"].strip():
            errors.append(f"{class_name}: notes must be a non-empty string.")

        validate_dataset_source(entry, errors)

    missing_from_manifest = sorted(expected_names - seen)
    extra_in_manifest = sorted(seen - expected_names)

    if missing_from_manifest:
        errors.append(f"Missing MVP classes: {', '.join(missing_from_manifest)}")
    if extra_in_manifest:
        errors.append(f"Extra manifest classes: {', '.join(extra_in_manifest)}")

    return errors


def print_summary(manifest_path: Path, entries: list[dict]) -> None:
    status_counts = Counter(entry["dataset_status"] for entry in entries)
    priority_counts = Counter(entry["priority"] for entry in entries)
    model_counts = Counter(entry["target_model"] for entry in entries)
    category_counts = Counter(entry["category"] for entry in entries)
    license_review_count = sum(1 for entry in entries if entry["license_review_required"] is True)

    print(f"Manifest: {manifest_path}")
    print(f"Classes: {len(entries)}")
    print(f"License review required: {license_review_count}/{len(entries)}")

    print("\nDataset status counts:")
    for status in sorted(status_counts):
        print(f"  - {status}: {status_counts[status]}")

    print("\nPriority counts:")
    for priority in ("high", "medium", "low"):
        print(f"  - {priority}: {priority_counts[priority]}")

    print("\nTarget model counts:")
    for model in sorted(model_counts):
        print(f"  - {model}: {model_counts[model]}")

    print("\nCategory counts:")
    for category in sorted(category_counts):
        print(f"  - {category}: {category_counts[category]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MVP dataset manifest. This does not download data.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to manifest YAML.")
    parser.add_argument("--mvp", type=Path, default=DEFAULT_MVP_PATH, help="Path to MVP class YAML.")
    args = parser.parse_args()

    try:
        manifest = load_yaml(args.manifest)
        mvp_config = load_yaml(args.mvp)
        expected = expected_classes(mvp_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    errors = validate_manifest(manifest, expected)
    entries = manifest.get("classes", [])

    if isinstance(entries, list) and entries:
        print_summary(args.manifest, entries)

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nMVP dataset manifest validation passed.")


if __name__ == "__main__":
    main()
