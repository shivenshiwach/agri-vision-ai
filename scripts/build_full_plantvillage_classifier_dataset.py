import argparse
import hashlib
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("datasets/raw/plantvillage/raw/color")
DEFAULT_OUTPUT_ROOT = Path("datasets/processed/full_plantvillage_classifier")
DEFAULT_VAL_RATIO = 0.2
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ClassPlan:
    class_name: str
    source_dir: Path
    image_count: int
    train_images: list[Path]
    val_images: list[Path]


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


def discover_class_dirs(source_root: Path, confirm: bool) -> list[Path]:
    if not source_root.exists():
        if confirm:
            raise FileNotFoundError(f"Source root not found: {source_root}")
        return []
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {source_root}")

    return sorted(path for path in source_root.iterdir() if path.is_dir())


def build_plan(source_root: Path, val_ratio: float, confirm: bool) -> tuple[list[ClassPlan], list[Path]]:
    rng = random.Random(RANDOM_SEED)
    plans: list[ClassPlan] = []
    skipped_empty_dirs: list[Path] = []

    for class_dir in discover_class_dirs(source_root, confirm):
        images = image_files(class_dir)
        if not images:
            skipped_empty_dirs.append(class_dir)
            continue

        shuffled_images = list(images)
        rng.shuffle(shuffled_images)
        split_count = val_count(len(shuffled_images), val_ratio)
        val_images = sorted(shuffled_images[:split_count])
        train_images = sorted(shuffled_images[split_count:])

        plans.append(
            ClassPlan(
                class_name=class_dir.name,
                source_dir=class_dir,
                image_count=len(images),
                train_images=train_images,
                val_images=val_images,
            )
        )

    return plans, skipped_empty_dirs


def safe_source_prefix(source_label: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", source_label)
    return prefix.strip("_") or "source"


def output_filename(class_name: str, source_dir: Path, image_path: Path) -> str:
    relative_path = image_path.relative_to(source_dir).as_posix()
    digest = hashlib.sha1(f"{class_name}/{relative_path}".encode("utf-8")).hexdigest()[:10]
    return f"{safe_source_prefix(class_name)}__{digest}__{image_path.name}"


def total_train_count(plans: list[ClassPlan]) -> int:
    return sum(len(plan.train_images) for plan in plans)


def total_val_count(plans: list[ClassPlan]) -> int:
    return sum(len(plan.val_images) for plan in plans)


def print_plan(
    source_root: Path,
    output_root: Path,
    plans: list[ClassPlan],
    skipped_empty_dirs: list[Path],
    val_ratio: float,
    confirm: bool,
    overwrite: bool,
) -> None:
    print("Full PlantVillage classifier dataset build plan")
    print("==============================================")
    print(f"Mode: {'copy files' if confirm else 'dry-run'}")
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Validation ratio: {val_ratio}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Use all images: yes")
    print(f"Overwrite output: {overwrite}")

    if not source_root.exists():
        print(f"\nWarning: source root not found: {source_root}")

    if skipped_empty_dirs:
        print("\nSkipped empty class folders:")
        for skipped_dir in skipped_empty_dirs:
            print(f"  - {skipped_dir.name}")

    print(f"\nClass count: {len(plans)}")
    print("\nImage count per class:")
    if plans:
        for plan in plans:
            print(
                "  - "
                f"{plan.class_name}: {plan.image_count} total, "
                f"{len(plan.train_images)} train, {len(plan.val_images)} val"
            )
    else:
        print("  - none")

    print(f"\nTotal train images: {total_train_count(plans)}")
    print(f"Total val images: {total_val_count(plans)}")
    print(f"Total images: {total_train_count(plans) + total_val_count(plans)}")

    if not confirm:
        print("\nNo files were copied. Re-run with --confirm to create the processed dataset.")


def prepare_output_root(output_root: Path, class_names: list[str], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    for split_name in ("train", "val"):
        for class_name in class_names:
            (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def copy_dataset(plans: list[ClassPlan], output_root: Path) -> int:
    copied_count = 0
    for plan in plans:
        for split_name, images in (("train", plan.train_images), ("val", plan.val_images)):
            output_dir = output_root / split_name / plan.class_name
            for image_path in images:
                destination = output_dir / output_filename(plan.class_name, plan.source_dir, image_path)
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
                shutil.copy2(image_path, destination)
                copied_count += 1
    return copied_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full PlantVillage image-classification dataset from every raw/color class folder."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="PlantVillage raw/color image root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Processed classifier dataset root.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying files. This is the default.")
    parser.add_argument("--confirm", action="store_true", help="Copy files into the processed classifier dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output root when used with --confirm.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)
    if args.val_ratio < 0 or args.val_ratio >= 1:
        print("ERROR: --val-ratio must be greater than or equal to 0 and less than 1.")
        sys.exit(1)

    confirm = args.confirm

    try:
        plans, skipped_empty_dirs = build_plan(args.source_root, args.val_ratio, confirm)
        print_plan(
            source_root=args.source_root,
            output_root=args.output_root,
            plans=plans,
            skipped_empty_dirs=skipped_empty_dirs,
            val_ratio=args.val_ratio,
            confirm=confirm,
            overwrite=args.overwrite,
        )

        if not confirm:
            return
        if not plans:
            raise ValueError(f"No non-empty class folders found under {args.source_root}")

        prepare_output_root(args.output_root, [plan.class_name for plan in plans], args.overwrite)
        copied_count = copy_dataset(plans, args.output_root)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\nCopied images: {copied_count}")
    print("Full PlantVillage classifier dataset build complete.")


if __name__ == "__main__":
    main()
