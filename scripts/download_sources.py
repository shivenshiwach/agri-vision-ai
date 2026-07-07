import argparse
import sys
from pathlib import Path

import yaml

import download_ip102
import download_plantvillage


CONFIG_PATH = Path("configs/download_sources.yaml")
SUPPORTED_SOURCES = (
    "plantvillage",
    "ip102",
    "plantdoc",
    "plantdoc_plus",
    "plantnet",
    "kaggle_field_diseases",
    "custom_field_diseases",
    "roboflow",
    "inaturalist",
    "open_images",
)
CONFIRMED_DOWNLOADERS = {
    "plantvillage": download_plantvillage.download,
    "ip102": download_ip102.download,
}


def load_sources(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("sources"), dict):
        raise ValueError(f"{config_path} must contain a top-level sources mapping.")

    missing_sources = sorted(set(SUPPORTED_SOURCES) - set(config["sources"]))
    if missing_sources:
        raise ValueError(f"{config_path} is missing source entries: {', '.join(missing_sources)}")

    return config["sources"]


def output_path(source_key: str, source_config: dict, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir / source_key
    configured_path = source_config.get("output_path")
    if configured_path:
        return Path(configured_path)
    return Path("datasets/raw") / source_key


def print_dry_run(source_key: str, source_config: dict, destination: Path) -> None:
    planned_for = source_config.get("planned_for", [])

    print("Dataset source dry run")
    print("======================")
    print(f"Source key: {source_key}")
    print(f"Name: {source_config.get('name')}")
    print(f"Type: {source_config.get('type')}")
    print(f"Dataset type: {source_config.get('dataset_type')}")
    print(f"Download method: {source_config.get('download_method')}")
    print(f"Download enabled: {source_config.get('download_enabled')}")
    print(f"License review required: {source_config.get('license_review_required')}")
    print(f"Source URL: {source_config.get('source_url')}")
    if source_config.get("reference_url"):
        print(f"Reference URL: {source_config.get('reference_url')}")
    print(f"Planned output directory: {destination}")

    print("\nPlanned use:")
    if isinstance(planned_for, list):
        for item in planned_for:
            print(f"  - {item}")
    else:
        print(f"  - {planned_for}")

    print("\nNotes:")
    print(f"  {source_config.get('notes')}")

    print("\nNo files were downloaded. No directories were created.")


def run_confirmed_download(source_key: str, source_config: dict, destination: Path) -> int:
    if source_config.get("download_enabled") is not True:
        print(f"ERROR: downloads are disabled in config for source: {source_key}")
        return 1
    if source_key not in CONFIRMED_DOWNLOADERS:
        print(f"ERROR: real downloads are not implemented for source: {source_key}")
        return 1

    return CONFIRMED_DOWNLOADERS[source_key](destination=destination, confirm=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset source acquisition. Dry-run is the default.")
    parser.add_argument("--source", required=True, choices=SUPPORTED_SOURCES, help="Dataset source to inspect or download.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without downloading data.")
    parser.add_argument("--confirm", action="store_true", help="Actually download supported approved sources.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Raw dataset root directory. The source key is appended to this path.",
    )
    args = parser.parse_args()

    if args.dry_run and args.confirm:
        print("ERROR: use either --dry-run or --confirm, not both.")
        sys.exit(1)

    try:
        sources = load_sources(CONFIG_PATH)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    source_config = sources[args.source]
    destination = output_path(args.source, source_config, args.output_dir)

    if args.confirm:
        sys.exit(run_confirmed_download(args.source, source_config, destination))

    print_dry_run(args.source, source_config, destination)


if __name__ == "__main__":
    main()
