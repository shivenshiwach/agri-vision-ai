import csv
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EUROSAT_NAME = "EuroSAT RGB"
EUROSAT_SOURCE_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EUROSAT_REFERENCE_URL = "https://github.com/phelber/EuroSAT"
EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

DEEPWEEDS_NAME = "DeepWeeds crop/land staging"
DEEPWEEDS_IMAGES_FILE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
DEEPWEEDS_IMAGES_URL = f"https://drive.google.com/uc?id={DEEPWEEDS_IMAGES_FILE_ID}"
DEEPWEEDS_LABELS_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
DEEPWEEDS_REFERENCE_URL = "https://github.com/AlexOlsen/DeepWeeds"
DEEPWEEDS_NEGATIVE_REVIEW_DIR = "deepweeds_negative_review"

PLANTDOC_NAME = "PlantDoc crop closeup staging"
PLANTDOC_SOURCE_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PLANTDOC_REFERENCE_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"


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


def prepare_destination(destination: Path) -> bool:
    if destination.exists():
        if not confirm_overwrite(destination):
            print("Download cancelled. Existing destination was left unchanged.")
            return False
        remove_existing(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    return True


def safe_slug(value: str) -> str:
    safe = []
    for character in value:
        if character.isalnum() or character in {".", "_", "-"}:
            safe.append(character)
        else:
            safe.append("_")
    return "".join(safe).strip("._-") or "item"


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Archive contains unsafe path: {member.filename}") from exc
        archive.extractall(destination)


def find_class_folder_root(extract_root: Path, expected_classes: tuple[str, ...]) -> Path:
    candidates = [extract_root]
    candidates.extend(path for path in extract_root.rglob("*") if path.is_dir())

    for candidate in candidates:
        child_names = {child.name for child in candidate.iterdir() if child.is_dir()}
        if set(expected_classes).issubset(child_names):
            return candidate

    raise ValueError(f"Could not find class folders: {', '.join(expected_classes)}")


def write_source_notes(destination: Path, lines: list[str]) -> None:
    with (destination / "SOURCE_NOTES.txt").open("w", encoding="utf-8") as notes_file:
        notes_file.write("\n".join(lines).rstrip() + "\n")


def print_direct_plan(source_name: str, destination: Path, source_url: str, reference_url: str, method: str) -> None:
    print(f"{source_name} download plan")
    print("=" * (len(source_name) + 14))
    print("Mode: dry-run")
    print(f"Official source URL: {source_url}")
    print(f"Reference URL: {reference_url}")
    print(f"Download method: {method}")
    print(f"Destination: {destination}")
    print("\nNo files were downloaded. Pass --confirm through scripts/download_sources.py to start the real download.")


def download_eurosat(destination: Path, confirm: bool) -> int:
    if not confirm:
        print_direct_plan(EUROSAT_NAME, destination, EUROSAT_SOURCE_URL, EUROSAT_REFERENCE_URL, "download Zenodo RGB zip and extract class folders")
        return 0
    if not prepare_destination(destination):
        return 1

    with tempfile.TemporaryDirectory(prefix="eurosat_", dir=destination.parent) as temp_name:
        temp_root = Path(temp_name)
        archive_path = temp_root / "EuroSAT_RGB.zip"
        extract_root = temp_root / "extract"

        print(f"Downloading {EUROSAT_NAME} to temporary archive")
        print("Command: direct HTTPS download from official Zenodo record 7711810")
        download_file(EUROSAT_SOURCE_URL, archive_path)
        safe_extract_zip(archive_path, extract_root)

        class_root = find_class_folder_root(extract_root, EUROSAT_CLASSES)
        destination.mkdir(parents=True, exist_ok=True)
        for class_name in EUROSAT_CLASSES:
            shutil.move(str(class_root / class_name), destination / class_name)

    write_source_notes(
        destination,
        [
            "Source: EuroSAT RGB",
            f"Official download: {EUROSAT_SOURCE_URL}",
            f"Reference: {EUROSAT_REFERENCE_URL}",
            "Version: Zenodo record 7711810 v2, published 2018-07-22",
            "License: MIT; Copernicus Sentinel data terms also apply.",
            "Local preparation: extracted RGB class folders for crop/land proxy experiments.",
        ],
    )
    print(f"Prepared EuroSAT class folders at {destination}")
    return 0


def require_gdown() -> str | None:
    gdown_path = shutil.which("gdown")
    if gdown_path is None:
        print("ERROR: gdown is required to download the official DeepWeeds Google Drive images archive.")
        print("Install it on the training server only after source approval: pip install gdown")
        return None
    return gdown_path


def read_deepweeds_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        if not reader.fieldnames or "Filename" not in reader.fieldnames or "Species" not in reader.fieldnames:
            raise ValueError("DeepWeeds labels.csv must contain Filename and Species columns.")
        return [row for row in reader if (row.get("Filename") or "").strip()]


def copy_deepweeds_images(extract_root: Path, labels_path: Path, destination: Path) -> tuple[int, int, list[str]]:
    image_by_name = {
        image_path.name: image_path
        for image_path in extract_root.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    weed_count = 0
    negative_review_count = 0
    missing_images: list[str] = []

    for row in read_deepweeds_labels(labels_path):
        filename = str(row.get("Filename") or "").strip()
        species = str(row.get("Species") or "").strip()
        source_image = image_by_name.get(filename)
        if source_image is None:
            missing_images.append(filename)
            continue

        if species.lower() == "negative":
            output_dir = destination / DEEPWEEDS_NEGATIVE_REVIEW_DIR
            negative_review_count += 1
        else:
            output_dir = destination / "weed_dominant_field"
            weed_count += 1

        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{safe_slug(species)}__{safe_slug(filename)}"
        shutil.copy2(source_image, output_dir / output_name)

    return weed_count, negative_review_count, missing_images


def download_deepweeds(destination: Path, confirm: bool) -> int:
    if not confirm:
        print_direct_plan(DEEPWEEDS_NAME, destination, DEEPWEEDS_IMAGES_URL, DEEPWEEDS_REFERENCE_URL, "gdown images.zip, fetch official labels.csv, stage class folders")
        return 0

    gdown_path = require_gdown()
    if gdown_path is None:
        return 1
    if not prepare_destination(destination):
        return 1

    with tempfile.TemporaryDirectory(prefix="deepweeds_", dir=destination.parent) as temp_name:
        temp_root = Path(temp_name)
        archive_path = temp_root / "images.zip"
        labels_path = temp_root / "labels.csv"
        extract_root = temp_root / "extract"

        print(f"Downloading {DEEPWEEDS_NAME} images to temporary archive")
        print("Command: gdown <official DeepWeeds images.zip Google Drive file> -O <temporary archive>")
        result = subprocess.run([gdown_path, DEEPWEEDS_IMAGES_URL, "-O", str(archive_path)], check=False)
        if result.returncode != 0:
            return result.returncode

        download_file(DEEPWEEDS_LABELS_URL, labels_path)
        safe_extract_zip(archive_path, extract_root)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_path, destination / "deepweeds_labels.csv")
        weed_count, negative_review_count, missing_images = copy_deepweeds_images(extract_root, labels_path, destination)

    if missing_images:
        print(f"ERROR: {len(missing_images)} labeled DeepWeeds images were missing from the archive.")
        print(f"First missing image: {missing_images[0]}")
        return 1
    if weed_count == 0:
        print("ERROR: no DeepWeeds weed images were staged.")
        return 1

    write_source_notes(
        destination,
        [
            "Source: DeepWeeds crop/land staging",
            f"Official image archive: {DEEPWEEDS_IMAGES_URL}",
            f"Official labels CSV: {DEEPWEEDS_LABELS_URL}",
            f"Reference: {DEEPWEEDS_REFERENCE_URL}",
            "License: images and annotations CC BY 4.0; repository code Apache-2.0.",
            "Local preparation: non-negative species staged as weed_dominant_field.",
            f"Negative rows staged under {DEEPWEEDS_NEGATIVE_REVIEW_DIR} for manual review and are intentionally unmapped.",
        ],
    )
    print(f"Prepared DeepWeeds crop/land folders at {destination}")
    print(f"Staged weed_dominant_field images: {weed_count}")
    print(f"Staged negative review images: {negative_review_count}")
    return 0


def plantdoc_image_roots(repo_root: Path) -> list[Path]:
    roots = [repo_root / "train", repo_root / "test"]
    existing_roots = [root for root in roots if root.exists() and root.is_dir()]
    return existing_roots or [repo_root]


def stage_plantdoc_crop_closeups(repo_root: Path, destination: Path) -> int:
    output_dir = destination / "crop_closeup"
    copied_count = 0
    for image_root in plantdoc_image_roots(repo_root):
        for image_path in sorted(image_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(repo_root).as_posix()
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_dir / safe_slug(relative))
            copied_count += 1
    return copied_count


def download_plantdoc_crop_land(destination: Path, confirm: bool) -> int:
    if not confirm:
        print_direct_plan(PLANTDOC_NAME, destination, PLANTDOC_SOURCE_URL, PLANTDOC_REFERENCE_URL, "git clone and stage all images as crop_closeup")
        return 0

    git_path = shutil.which("git")
    if git_path is None:
        print("ERROR: git is required to clone the official PlantDoc repository.")
        return 1
    if not prepare_destination(destination):
        return 1

    with tempfile.TemporaryDirectory(prefix="plantdoc_", dir=destination.parent) as temp_name:
        repo_root = Path(temp_name) / "PlantDoc-Dataset"
        command = [git_path, "clone", "--depth", "1", PLANTDOC_SOURCE_URL, str(repo_root)]
        print(f"Downloading {PLANTDOC_NAME} to temporary repository")
        print("Command: git clone --depth 1 <official PlantDoc repo> <temporary repository>")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode

        destination.mkdir(parents=True, exist_ok=True)
        copied_count = stage_plantdoc_crop_closeups(repo_root, destination)

    if copied_count == 0:
        print("ERROR: no PlantDoc images were staged.")
        return 1

    write_source_notes(
        destination,
        [
            "Source: PlantDoc crop closeup staging",
            f"Official repository: {PLANTDOC_SOURCE_URL}",
            f"Reference: {PLANTDOC_REFERENCE_URL}",
            "License: Creative Commons Attribution 4.0 International.",
            "Local preparation: all train/test images staged under crop_closeup for crop/land scene experiments.",
            "Limit: this does not provide crop_field, bare soil, harvested field, waterlogging, or unknown classes.",
        ],
    )
    print(f"Prepared PlantDoc crop_closeup folder at {destination}")
    print(f"Staged crop_closeup images: {copied_count}")
    return 0
