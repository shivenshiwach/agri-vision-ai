import argparse
import re
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MVP_PATH = Path("configs/mvp_classes.yaml")
DEFAULT_RAW_ROOT = Path("datasets/raw")


def load_mvp_classes(mvp_path: Path) -> dict:
    with mvp_path.open("r", encoding="utf-8") as mvp_file:
        data = yaml.safe_load(mvp_file)
    if not isinstance(data, dict):
        raise ValueError(f"{mvp_path} must contain a YAML mapping.")
    return data


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def token_set(value: str) -> set[str]:
    return set(normalize_label(value).split())


def possible_matches(source_labels: list[str], target_labels: list[str]) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    normalized_targets = [(target, normalize_label(target), token_set(target)) for target in target_labels]

    for source in source_labels:
        source_norm = normalize_label(source)
        source_tokens = token_set(source)

        for target, target_norm, target_tokens in normalized_targets:
            if not source_norm or not target_norm:
                continue
            if source_norm == target_norm:
                matches.append((source, target, "exact"))
            elif target_norm in source_norm or source_norm in target_norm:
                matches.append((source, target, "substring"))
            elif source_tokens & target_tokens:
                matches.append((source, target, "token_overlap"))

    return matches


def count_images_by_class(class_root: Path) -> Counter:
    counts: Counter = Counter()
    if not class_root.exists():
        return counts

    for item in sorted(class_root.iterdir()):
        if not item.is_dir():
            continue
        counts[item.name] = sum(
            1 for path in item.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def inspect_plantvillage(raw_root: Path, mvp_classes: dict) -> None:
    plantvillage_root = raw_root / "plantvillage"
    color_root = plantvillage_root / "raw" / "color"
    disease_targets = mvp_classes.get("diseases", [])

    print("PlantVillage")
    print("============")
    print(f"Root: {plantvillage_root}")
    print(f"Expected raw/color path: {color_root}")

    if not color_root.exists():
        print("Status: not found")
        print("Class folders: 0")
        print("Image count: 0")
        print()
        return

    counts = count_images_by_class(color_root)
    print("Status: found")
    print(f"Class folders: {len(counts)}")
    print(f"Image count: {sum(counts.values())}")
    print()
    print("Classes:")
    for class_name, count in counts.items():
        print(f"  - {class_name}: {count}")

    matches = possible_matches(list(counts), disease_targets)
    print()
    print("Possible MVP disease matches:")
    if not matches:
        print("  - none")
    else:
        for source, target, match_type in matches:
            print(f"  - {source} -> {target} ({match_type})")
    print()


def read_classes_txt(path: Path) -> list[str]:
    if not path.exists():
        return []

    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            labels.append(value)
    return labels


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(f"*{suffix}") if item.is_file())


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_ip102(raw_root: Path, mvp_classes: dict) -> None:
    ip102_root = raw_root / "ip102"
    classes_txt = ip102_root / "classes.txt"
    classification_tar = ip102_root / "Classification" / "ip102_v1.1.tar"
    classification_dir = ip102_root / "Classification" / "ip102_v1.1"
    annotations_tar = ip102_root / "Detection" / "VOC2007" / "Annotations.tar"
    annotations_dir = ip102_root / "Detection" / "VOC2007" / "Annotations"
    jpeg_tar = ip102_root / "Detection" / "VOC2007" / "JPEGImages.tar"
    jpeg_dir = ip102_root / "Detection" / "VOC2007" / "JPEGImages"
    pest_targets = mvp_classes.get("pests", [])

    print("IP102")
    print("=====")
    print(f"Root: {ip102_root}")
    print(f"classes.txt: {'found' if classes_txt.exists() else 'missing'} ({classes_txt})")

    ip102_classes = read_classes_txt(classes_txt)
    print(f"Classification class count: {len(ip102_classes)}")
    if ip102_classes:
        print("Classification classes:")
        for class_name in ip102_classes:
            print(f"  - {class_name}")

    print()
    print("Detection VOC2007 files:")
    print(f"  - Annotations tar: {'found' if annotations_tar.exists() else 'missing'} ({annotations_tar})")
    print(f"  - JPEGImages tar: {'found' if jpeg_tar.exists() else 'missing'} ({jpeg_tar})")
    print(f"  - Extracted Annotations: {'found' if annotations_dir.exists() else 'missing'} ({annotations_dir})")
    print(f"  - Extracted JPEGImages: {'found' if jpeg_dir.exists() else 'missing'} ({jpeg_dir})")

    print()
    print("Extracted counts:")
    print(f"  - Classification images: {count_images(classification_dir)}")
    print(f"  - Detection images: {count_images(jpeg_dir)}")
    print(f"  - Detection XML annotations: {count_files(annotations_dir, '.xml')}")

    print()
    print("Archive checks:")
    print(f"  - Classification tar: {'found' if classification_tar.exists() else 'missing'} ({classification_tar})")
    print(f"  - Detection Annotations tar: {'found' if annotations_tar.exists() else 'missing'} ({annotations_tar})")
    print(f"  - Detection JPEGImages tar: {'found' if jpeg_tar.exists() else 'missing'} ({jpeg_tar})")

    matches = possible_matches(ip102_classes, pest_targets)
    print()
    print("Possible MVP pest class matches:")
    if not matches:
        print("  - none")
    else:
        for source, target, match_type in matches:
            print(f"  - {source} -> {target} ({match_type})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local raw datasets without modifying files.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Raw dataset root directory.")
    parser.add_argument("--mvp-config", type=Path, default=DEFAULT_MVP_PATH, help="Path to MVP classes YAML.")
    args = parser.parse_args()

    try:
        mvp_classes = load_mvp_classes(args.mvp_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print("Dataset inspection")
    print("==================")
    print(f"Raw root: {args.raw_root}")
    print(f"MVP config: {args.mvp_config}")
    print()

    inspect_plantvillage(args.raw_root, mvp_classes)
    inspect_ip102(args.raw_root, mvp_classes)
    print("Inspection complete. No files were created, moved, extracted, or deleted.")


if __name__ == "__main__":
    main()
