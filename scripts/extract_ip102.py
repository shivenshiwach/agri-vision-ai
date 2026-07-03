import argparse
import shutil
import sys
import tarfile
from pathlib import Path


DEFAULT_ROOT = Path("datasets/raw/ip102")
EXTRACTION_TARGETS = (
    (
        Path("Detection/VOC2007/Annotations.tar"),
        Path("Detection/VOC2007/Annotations"),
    ),
    (
        Path("Detection/VOC2007/JPEGImages.tar"),
        Path("Detection/VOC2007/JPEGImages"),
    ),
    (
        Path("Classification/ip102_v1.1.tar"),
        Path("Classification/ip102_v1.1"),
    ),
)


def format_bytes(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "missing"

    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def is_within_directory(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            if not is_within_directory(destination, member_path):
                raise RuntimeError(f"Unsafe tar member path blocked: {member.name}")
        archive.extractall(destination)


def planned_items(root: Path) -> list[dict]:
    items = []
    for relative_tar, relative_destination in EXTRACTION_TARGETS:
        tar_path = root / relative_tar
        destination = root / relative_destination
        tar_size = tar_path.stat().st_size if tar_path.exists() and tar_path.is_file() else None
        items.append(
            {
                "tar_path": tar_path,
                "destination": destination,
                "tar_size": tar_size,
                "tar_exists": tar_path.exists(),
                "destination_exists": destination.exists(),
            }
        )
    return items


def print_plan(items: list[dict], confirm: bool, overwrite: bool) -> None:
    mode = "confirm" if confirm else "dry-run"
    print("IP102 extraction plan")
    print("=====================")
    print(f"Mode: {mode}")
    print(f"Overwrite existing destinations: {overwrite}")
    print()

    for item in items:
        status = "ready" if item["tar_exists"] else "missing source tar"
        if item["destination_exists"] and not overwrite:
            status = "skip existing destination"

        print(f"Source tar: {item['tar_path']}")
        print(f"Tar size: {format_bytes(item['tar_size'])}")
        print(f"Target path: {item['destination']}")
        print(f"Action: {status}")
        print()

    if not confirm:
        print("No files were extracted. Pass --confirm to extract archives.")


def extract_items(items: list[dict], overwrite: bool) -> int:
    errors = 0

    for item in items:
        tar_path = item["tar_path"]
        destination = item["destination"]

        if not tar_path.exists():
            print(f"ERROR: missing source tar: {tar_path}")
            errors += 1
            continue

        if destination.exists():
            if not overwrite:
                print(f"Skipping existing destination: {destination}")
                continue
            print(f"Removing existing destination: {destination}")
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        print(f"Extracting {tar_path} -> {destination}")
        try:
            safe_extract_tar(tar_path, destination)
        except (tarfile.TarError, OSError, RuntimeError) as exc:
            print(f"ERROR: failed to extract {tar_path}: {exc}")
            errors += 1

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract approved IP102 tar archives.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="IP102 raw dataset root.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned extraction without extracting.")
    parser.add_argument("--confirm", action="store_true", help="Actually extract archives.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted destinations.")
    args = parser.parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)

    confirm = args.confirm
    items = planned_items(args.root)
    print_plan(items, confirm=confirm, overwrite=args.overwrite)

    if not confirm:
        return

    sys.exit(extract_items(items, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
