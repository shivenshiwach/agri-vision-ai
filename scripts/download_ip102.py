import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE_NAME = "IP102"
SOURCE_URL = "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing"
REFERENCE_URL = "https://github.com/xpwu95/IP102"
DEFAULT_OUTPUT_DIR = Path("datasets/raw/ip102")


def confirm_overwrite(destination: Path) -> bool:
    if not destination.exists():
        return True

    print(f"Destination already exists: {destination}")
    response = input("Type 'yes' to delete it and download again: ").strip().lower()
    return response == "yes"


def remove_existing(destination: Path) -> None:
    if destination.is_dir():
        shutil.rmtree(destination)
    else:
        destination.unlink()


def print_plan(destination: Path) -> None:
    print(f"{SOURCE_NAME} download plan")
    print("=" * 20)
    print("Mode: dry-run")
    print(f"Official source URL: {SOURCE_URL}")
    print(f"Reference URL: {REFERENCE_URL}")
    print("Download method: gdown --folder")
    print(f"Destination: {destination}")
    print("\nNo files were downloaded. Pass --confirm to start the real download.")


def download(destination: Path, confirm: bool) -> int:
    if not confirm:
        print_plan(destination)
        return 0

    gdown_path = shutil.which("gdown")
    if gdown_path is None:
        print("ERROR: gdown is required to download the official IP102 Google Drive folder.")
        print("Install it on the training server only after license approval: pip install gdown")
        return 1

    if destination.exists():
        if not confirm_overwrite(destination):
            print("Download cancelled. Existing destination was left unchanged.")
            return 1
        remove_existing(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [gdown_path, "--folder", SOURCE_URL, "-O", str(destination)]
    print(f"Downloading {SOURCE_NAME} to {destination}")
    print("Command: gdown --folder <official IP102 Drive folder> -O <destination>")
    return subprocess.run(command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IP102 after explicit confirmation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination dataset directory.")
    parser.add_argument("--confirm", action="store_true", help="Actually download the dataset.")
    args = parser.parse_args()

    sys.exit(download(args.output_dir, args.confirm))


if __name__ == "__main__":
    main()
