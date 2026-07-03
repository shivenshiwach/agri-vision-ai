import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_DATASETS = ("plantvillage", "ip102")


def folder_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total

    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def image_count(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def dataset_status(path: Path) -> str:
    if not path.exists():
        return "not_downloaded"
    if path.is_file():
        return "file"
    if any(path.iterdir()):
        return "present"
    return "empty"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show local raw dataset inventory.")
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"), help="Raw dataset root directory.")
    args = parser.parse_args()

    print("Dataset inventory")
    print("=================")
    print(f"Root: {args.root}")
    print()
    print(f"{'dataset':<15} {'status':<15} {'images':>10} {'size':>12} path")
    print("-" * 72)

    for dataset in DEFAULT_DATASETS:
        path = args.root / dataset
        status = dataset_status(path)
        images = image_count(path)
        size = format_bytes(folder_size(path))
        print(f"{dataset:<15} {status:<15} {images:>10} {size:>12} {path}")


if __name__ == "__main__":
    main()
