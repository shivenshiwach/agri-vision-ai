import argparse
import hashlib
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_MAPPING_PATH = Path("configs/label_mapping.yaml")
DEFAULT_SOURCE_ROOT = Path("datasets/raw/plantvillage/raw/color")
DEFAULT_OUTPUT_ROOT = Path("datasets/processed/disease_classifier")
DEFAULT_VAL_RATIO = 0.2
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SourceMapping:
    source_label: str
    target_class: str


@dataclass
class SourcePlan:
    mapping: SourceMapping
    source_dir: Path
    available_count: int
    selected_count: int
    train_images: list[Path]
    val_images: list[Path]
    missing: bool = False


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def approved_plantvillage_disease_mappings(mapping_path: Path) -> list[SourceMapping]:
    label_mapping = load_yaml(mapping_path)
    entries = label_mapping.get("mappings")
    if not isinstance(entries, list):
        raise ValueError(f"{mapping_path} must contain a top-level mappings list.")

    mappings: list[SourceMapping] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Mapping {index} must be a YAML mapping.")

        if entry.get("source_dataset") != "PlantVillage":
            continue
        if entry.get("review_status") != "approved":
            continue
        if entry.get("target_group") != "diseases":
            continue

        source_label = entry.get("source_label")
        target_class = entry.get("target_class")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(f"Mapping {index} has an empty PlantVillage source_label.")
        if not isinstance(target_class, str) or not target_class.strip():
            raise ValueError(f"Mapping {index} has an empty target_class.")

        mappings.append(SourceMapping(source_label=source_label, target_class=target_class))

    if not mappings:
        raise ValueError(f"{mapping_path} does not contain approved PlantVillage disease mappings.")

    return mappings


def image_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def val_count(total_count: int, val_ratio: float) -> int:
    if total_count <= 1 or val_ratio <= 0:
        return 0

    count = round(total_count * val_ratio)
    return min(max(count, 1), total_count - 1)


def build_plan(
    mappings: list[SourceMapping],
    source_root: Path,
    max_per_source_class: int | None,
    val_ratio: float,
) -> list[SourcePlan]:
    rng = random.Random(RANDOM_SEED)
    plans: list[SourcePlan] = []

    for mapping in mappings:
        source_dir = source_root / mapping.source_label
        if not source_dir.is_dir():
            plans.append(
                SourcePlan(
                    mapping=mapping,
                    source_dir=source_dir,
                    available_count=0,
                    selected_count=0,
                    train_images=[],
                    val_images=[],
                    missing=True,
                )
            )
            continue

        available_images = image_files(source_dir)
        selected_images = available_images
        if max_per_source_class is not None and len(available_images) > max_per_source_class:
            selected_images = sorted(rng.sample(available_images, max_per_source_class))

        shuffled_images = list(selected_images)
        rng.shuffle(shuffled_images)
        split_count = val_count(len(shuffled_images), val_ratio)
        val_images = sorted(shuffled_images[:split_count])
        train_images = sorted(shuffled_images[split_count:])

        plans.append(
            SourcePlan(
                mapping=mapping,
                source_dir=source_dir,
                available_count=len(available_images),
                selected_count=len(selected_images),
                train_images=train_images,
                val_images=val_images,
            )
        )

    return plans


def ordered_target_classes(mappings: list[SourceMapping]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for mapping in mappings:
        if mapping.target_class not in seen:
            seen.add(mapping.target_class)
            ordered.append(mapping.target_class)
    return ordered


def planned_counts(plans: list[SourcePlan]) -> tuple[Counter[str], Counter[str]]:
    train_counts: Counter[str] = Counter()
    val_counts: Counter[str] = Counter()
    for plan in plans:
        target_class = plan.mapping.target_class
        train_counts[target_class] += len(plan.train_images)
        val_counts[target_class] += len(plan.val_images)
    return train_counts, val_counts


def print_plan(
    mapping_path: Path,
    source_root: Path,
    output_root: Path,
    plans: list[SourcePlan],
    target_classes: list[str],
    max_per_source_class: int | None,
    val_ratio: float,
    confirm: bool,
    overwrite: bool,
) -> None:
    train_counts, val_counts = planned_counts(plans)

    print("Disease classifier dataset build plan")
    print("=====================================")
    print(f"Mode: {'copy files' if confirm else 'dry-run'}")
    print(f"Label mapping: {mapping_path}")
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Validation ratio: {val_ratio}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Max per source class: {max_per_source_class if max_per_source_class is not None else 'all'}")
    print(f"Overwrite output: {overwrite}")

    missing_plans = [plan for plan in plans if plan.missing]
    if missing_plans:
        print("\nWarnings:")
        for plan in missing_plans:
            print(f"  - Missing source folder, skipped: {plan.source_dir}")

    print("\nSource class plan:")
    for plan in plans:
        status = "missing" if plan.missing else "ok"
        print(
            "  - "
            f"{plan.mapping.source_label} -> {plan.mapping.target_class}: "
            f"{plan.selected_count}/{plan.available_count} images, "
            f"{len(plan.train_images)} train, {len(plan.val_images)} val "
            f"({status})"
        )

    print("\nPlanned image counts by target class:")
    for target_class in target_classes:
        train_count = train_counts[target_class]
        current_val_count = val_counts[target_class]
        total_count = train_count + current_val_count
        print(f"  - {target_class}: {total_count} total, {train_count} train, {current_val_count} val")

    total_images = sum(train_counts.values()) + sum(val_counts.values())
    print(f"\nTotal planned images: {total_images}")
    if not confirm:
        print("No files were copied. Re-run with --confirm to create the processed dataset.")


def safe_source_prefix(source_label: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", source_label)
    return prefix.strip("_") or "source"


def output_filename(source_label: str, source_dir: Path, image_path: Path) -> str:
    relative_path = image_path.relative_to(source_dir).as_posix()
    digest = hashlib.sha1(f"{source_label}/{relative_path}".encode("utf-8")).hexdigest()[:10]
    return f"{safe_source_prefix(source_label)}__{digest}__{image_path.name}"


def prepare_output_root(output_root: Path, target_classes: list[str], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    for split_name in ("train", "val"):
        for target_class in target_classes:
            (output_root / split_name / target_class).mkdir(parents=True, exist_ok=True)


def copy_dataset(plans: list[SourcePlan], output_root: Path) -> int:
    copied_count = 0
    for plan in plans:
        for split_name, images in (("train", plan.train_images), ("val", plan.val_images)):
            output_dir = output_root / split_name / plan.mapping.target_class
            for image_path in images:
                destination = output_dir / output_filename(plan.mapping.source_label, plan.source_dir, image_path)
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
                shutil.copy2(image_path, destination)
                copied_count += 1
    return copied_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PlantVillage disease classifier dataset from approved label mappings."
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to approved label mapping YAML.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="PlantVillage color image root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Processed classifier dataset root.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying files. This is the default.")
    parser.add_argument("--confirm", action="store_true", help="Copy files into the processed classifier dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output root when used with --confirm.")
    parser.add_argument(
        "--max-per-source-class",
        type=int,
        default=None,
        help="Optional cap per PlantVillage source class. By default all available images are used.",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)
    if args.max_per_source_class is not None and args.max_per_source_class <= 0:
        print("ERROR: --max-per-source-class must be a positive integer.")
        sys.exit(1)
    if args.val_ratio < 0 or args.val_ratio >= 1:
        print("ERROR: --val-ratio must be greater than or equal to 0 and less than 1.")
        sys.exit(1)

    confirm = args.confirm

    try:
        mappings = approved_plantvillage_disease_mappings(args.mapping)
        target_classes = ordered_target_classes(mappings)
        plans = build_plan(
            mappings=mappings,
            source_root=args.source_root,
            max_per_source_class=args.max_per_source_class,
            val_ratio=args.val_ratio,
        )
        print_plan(
            mapping_path=args.mapping,
            source_root=args.source_root,
            output_root=args.output_root,
            plans=plans,
            target_classes=target_classes,
            max_per_source_class=args.max_per_source_class,
            val_ratio=args.val_ratio,
            confirm=confirm,
            overwrite=args.overwrite,
        )

        if not confirm:
            return

        prepare_output_root(args.output_root, target_classes, args.overwrite)
        copied_count = copy_dataset(plans, args.output_root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\nCopied images: {copied_count}")
    print("Disease classifier dataset build complete.")


if __name__ == "__main__":
    main()
