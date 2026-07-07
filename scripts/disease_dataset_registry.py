import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/disease_dataset_registry.yaml")
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NON_DISEASE_MARKERS = {
    "background",
    "healthy",
    "normal",
    "non disease",
    "not diseased",
    "unlabeled",
    "unknown",
}
SPLIT_DIR_NAMES = {"train", "test", "val", "valid", "validation"}


@dataclass
class AliasBook:
    alias_to_canonical: dict[str, str]
    duplicate_aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class LabelDiscovery:
    original_label: str
    image_count: int = 0
    class_dirs: list[Path] = field(default_factory=list)


@dataclass
class ReportRow:
    source_key: str
    source_dataset: str
    original_label: str
    normalized_label: str | None
    image_count: int
    status: str
    class_dirs: list[Path] = field(default_factory=list)


@dataclass
class DatasetInspection:
    source_key: str
    source_dataset: str
    enabled: bool
    root: Path
    status: str
    rows: list[ReportRow] = field(default_factory=list)
    missing_class_roots: list[Path] = field(default_factory=list)
    duplicate_source_labels: dict[str, list[str]] = field(default_factory=dict)
    duplicate_normalized_labels: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class InspectionResult:
    registry_path: Path
    mode: str
    taxonomy_path: Path
    alias_path: Path
    image_extensions: set[str]
    canonical_labels: list[str]
    alias_book: AliasBook
    datasets: list[DatasetInspection]
    rows: list[ReportRow]
    missing_canonical_labels: list[str]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def normalize_key(value: str) -> str:
    value = value.replace("&", " and ")
    value = re.sub(r"[_/|+]+", " ", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "label"


def load_canonical_labels(taxonomy_path: Path) -> list[str]:
    taxonomy = load_yaml_mapping(taxonomy_path)
    diseases = taxonomy.get("diseases")
    if not isinstance(diseases, list) or not all(isinstance(label, str) for label in diseases):
        raise ValueError(f"{taxonomy_path} must contain a diseases list.")
    return diseases


def load_alias_book(alias_path: Path, canonical_labels: list[str]) -> AliasBook:
    canonical_by_key = {normalize_key(label): label for label in canonical_labels}
    alias_to_canonical = dict(canonical_by_key)
    duplicate_aliases: dict[str, set[str]] = defaultdict(set)

    with alias_path.open("r", encoding="utf-8", newline="") as alias_file:
        reader = csv.DictReader(alias_file)
        required_fields = {"canonical_label", "alias"}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"{alias_path} must contain canonical_label and alias columns.")

        for row in reader:
            canonical = (row.get("canonical_label") or "").strip()
            alias = (row.get("alias") or "").strip()
            if not canonical or not alias:
                continue
            if canonical not in canonical_labels:
                raise ValueError(f"{alias_path} references unknown canonical disease: {canonical}")

            alias_key = normalize_key(alias)
            existing = alias_to_canonical.get(alias_key)
            if existing and existing != canonical:
                duplicate_aliases[alias_key].update({existing, canonical})
                continue
            alias_to_canonical[alias_key] = canonical

    return AliasBook(
        alias_to_canonical=alias_to_canonical,
        duplicate_aliases={key: sorted(values) for key, values in duplicate_aliases.items()},
    )


def image_files(path: Path, image_extensions: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in image_extensions
    )


def should_skip_root_child(child: Path, configured_class_roots: list[str], current_class_root: str) -> bool:
    if current_class_root not in {"", "."}:
        return False
    if child.name.lower() not in SPLIT_DIR_NAMES:
        return False
    return child.name in configured_class_roots or child.name.lower() in configured_class_roots


def discover_labels(dataset_config: dict[str, Any], image_extensions: set[str]) -> tuple[Path, list[LabelDiscovery], list[Path], str]:
    root = Path(str(dataset_config.get("root", "")))
    if not root.exists():
        return root, [], [], "missing_source_root"
    if not root.is_dir():
        return root, [], [], "invalid_source_root"

    discovery = dataset_config.get("discovery", {})
    if not isinstance(discovery, dict):
        raise ValueError(f"Dataset {dataset_config.get('name')} discovery must be a mapping.")

    discovery_type = discovery.get("type", "class_folders")
    if discovery_type != "class_folders":
        raise ValueError(f"Unsupported discovery type {discovery_type!r}; expected class_folders.")

    class_roots = discovery.get("class_roots", ["."])
    if not isinstance(class_roots, list) or not all(isinstance(item, str) for item in class_roots):
        raise ValueError(f"Dataset {dataset_config.get('name')} discovery.class_roots must be a list of paths.")

    configured_class_roots = [item.strip() or "." for item in class_roots]
    missing_class_roots: list[Path] = []
    discoveries: dict[str, LabelDiscovery] = {}

    for relative_class_root in configured_class_roots:
        class_root = root if relative_class_root == "." else root / relative_class_root
        if not class_root.exists():
            missing_class_roots.append(class_root)
            continue
        if not class_root.is_dir():
            continue

        for child in sorted(class_root.iterdir()):
            if not child.is_dir():
                continue
            if should_skip_root_child(child, configured_class_roots, relative_class_root):
                continue

            files = image_files(child, image_extensions)
            discovery_record = discoveries.setdefault(child.name, LabelDiscovery(original_label=child.name))
            discovery_record.image_count += len(files)
            discovery_record.class_dirs.append(child)

    status = "found" if discoveries else "found_no_class_labels"
    return root, sorted(discoveries.values(), key=lambda item: item.original_label.lower()), missing_class_roots, status


def label_candidates(original_label: str) -> list[str]:
    candidates = [original_label]
    separators = ("___", "__", " - ", ":", "|")

    for separator in separators:
        if separator in original_label:
            candidates.append(original_label.rsplit(separator, 1)[-1])

    cleaned = re.sub(r"\([^)]*\)", " ", original_label)
    if cleaned != original_label:
        candidates.append(cleaned)

    return list(dict.fromkeys(candidates))


def is_non_disease_label(original_label: str) -> bool:
    source_key = normalize_key(original_label)
    return any(marker in source_key for marker in NON_DISEASE_MARKERS)


def normalize_disease_label(
    original_label: str,
    alias_book: AliasBook,
    canonical_labels: list[str],
) -> tuple[str | None, str]:
    if is_non_disease_label(original_label):
        return None, "non_disease"

    candidate_keys = [normalize_key(candidate) for candidate in label_candidates(original_label)]
    for candidate_key in candidate_keys:
        canonical = alias_book.alias_to_canonical.get(candidate_key)
        if canonical:
            return canonical, "mapped"

    canonical_candidates: list[tuple[int, str]] = []
    for canonical in canonical_labels:
        canonical_key = normalize_key(canonical)
        for candidate_key in candidate_keys:
            if canonical_key and canonical_key in candidate_key:
                canonical_candidates.append((len(canonical_key), canonical))

    if canonical_candidates:
        canonical_candidates.sort(reverse=True)
        return canonical_candidates[0][1], "mapped"

    return None, "unmapped"


def duplicate_source_labels(discoveries: list[LabelDiscovery]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for discovery in discoveries:
        grouped[normalize_key(discovery.original_label)].append(discovery.original_label)
    return {key: sorted(set(values)) for key, values in grouped.items() if len(set(values)) > 1}


def duplicate_normalized_labels(rows: list[ReportRow]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.normalized_label and row.status == "mapped":
            grouped[row.normalized_label].append(row.original_label)
    return {key: sorted(set(values)) for key, values in grouped.items() if len(set(values)) > 1}


def enabled_dataset_items(
    registry: dict[str, Any],
    dataset_keys: list[str] | None = None,
    include_disabled: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Registry must contain a datasets mapping.")

    if dataset_keys:
        missing = sorted(set(dataset_keys) - set(datasets))
        if missing:
            raise ValueError(f"Unknown dataset key(s): {', '.join(missing)}")
        return [(key, datasets[key]) for key in dataset_keys]

    return [
        (key, config)
        for key, config in datasets.items()
        if include_disabled or bool(config.get("enabled", False))
    ]


def run_inspection(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    dataset_keys: list[str] | None = None,
    include_disabled: bool = False,
    mode: str = "dry-run",
) -> InspectionResult:
    registry = load_yaml_mapping(registry_path)
    taxonomy_path = Path(str(registry.get("taxonomy_path", "configs/classes.yaml")))
    alias_path = Path(str(registry.get("label_aliases_path", "data/taxonomy/disease_label_aliases.csv")))
    canonical_labels = load_canonical_labels(taxonomy_path)
    alias_book = load_alias_book(alias_path, canonical_labels)

    configured_extensions = registry.get("image_extensions", sorted(DEFAULT_IMAGE_EXTENSIONS))
    if not isinstance(configured_extensions, list) or not all(isinstance(item, str) for item in configured_extensions):
        raise ValueError("Registry image_extensions must be a list of suffix strings.")
    image_extensions = {item.lower() for item in configured_extensions}

    inspections: list[DatasetInspection] = []
    all_rows: list[ReportRow] = []

    for source_key, dataset_config in enabled_dataset_items(registry, dataset_keys, include_disabled):
        source_dataset = str(dataset_config.get("name") or source_key)
        root, discoveries, missing_class_roots, status = discover_labels(dataset_config, image_extensions)
        rows: list[ReportRow] = []

        for discovery in discoveries:
            normalized_label, row_status = normalize_disease_label(
                discovery.original_label,
                alias_book,
                canonical_labels,
            )
            row = ReportRow(
                source_key=source_key,
                source_dataset=source_dataset,
                original_label=discovery.original_label,
                normalized_label=normalized_label,
                image_count=discovery.image_count,
                status=row_status,
                class_dirs=discovery.class_dirs,
            )
            rows.append(row)
            all_rows.append(row)

        inspections.append(
            DatasetInspection(
                source_key=source_key,
                source_dataset=source_dataset,
                enabled=bool(dataset_config.get("enabled", False)),
                root=root,
                status=status,
                rows=rows,
                missing_class_roots=missing_class_roots,
                duplicate_source_labels=duplicate_source_labels(discoveries),
                duplicate_normalized_labels=duplicate_normalized_labels(rows),
            )
        )

    mapped_counts: Counter[str] = Counter()
    for row in all_rows:
        if row.normalized_label and row.status == "mapped":
            mapped_counts[row.normalized_label] += row.image_count

    missing_canonical_labels = [label for label in canonical_labels if mapped_counts[label] == 0]

    return InspectionResult(
        registry_path=registry_path,
        mode=mode,
        taxonomy_path=taxonomy_path,
        alias_path=alias_path,
        image_extensions=image_extensions,
        canonical_labels=canonical_labels,
        alias_book=alias_book,
        datasets=inspections,
        rows=all_rows,
        missing_canonical_labels=missing_canonical_labels,
    )


def iter_row_images(row: ReportRow, image_extensions: set[str] | None = None) -> list[tuple[Path, Path]]:
    extensions = image_extensions or DEFAULT_IMAGE_EXTENSIONS
    pairs: list[tuple[Path, Path]] = []
    for class_dir in row.class_dirs:
        for image_path in image_files(class_dir, extensions):
            pairs.append((class_dir, image_path))
    return pairs


def output_image_name(source_key: str, original_label: str, class_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(class_dir).as_posix()
    digest = hashlib.sha1(f"{source_key}/{original_label}/{class_dir}/{relative}".encode("utf-8")).hexdigest()[:12]
    return f"{safe_slug(source_key)}__{safe_slug(original_label)}__{digest}__{safe_slug(image_path.name)}"
