import argparse
import sys
from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/download_sources.yaml")
SUPPORTED_SOURCES = ("plantvillage", "ip102", "plantdoc", "roboflow", "inaturalist", "open_images")
DISABLED_DOWNLOAD_MESSAGE = "Real downloads are disabled until source licenses and URLs are approved."


def load_sources(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("sources"), dict):
        raise ValueError(f"{config_path} must contain a top-level sources mapping.")

    missing_sources = sorted(set(SUPPORTED_SOURCES) - set(config["sources"]))
    if missing_sources:
        raise ValueError(f"{config_path} is missing source entries: {', '.join(missing_sources)}")

    return config["sources"]


def print_dry_run(source_key: str, source_config: dict, output_dir: Path) -> None:
    planned_output = output_dir / source_key
    planned_for = source_config.get("planned_for", [])

    print("Dataset source dry run")
    print("======================")
    print(f"Source key: {source_key}")
    print(f"Name: {source_config.get('name')}")
    print(f"Type: {source_config.get('type')}")
    print(f"Download enabled: {source_config.get('download_enabled')}")
    print(f"License review required: {source_config.get('license_review_required')}")
    print(f"Source URL: {source_config.get('source_url')}")
    print(f"Planned output directory: {planned_output}")

    print("\nPlanned use:")
    if isinstance(planned_for, list):
        for item in planned_for:
            print(f"  - {item}")
    else:
        print(f"  - {planned_for}")

    print("\nNotes:")
    print(f"  {source_config.get('notes')}")

    print("\nNo files were downloaded. No directories were created.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run dataset source acquisition planning. Real downloads are disabled."
    )
    parser.add_argument("--source", required=True, choices=SUPPORTED_SOURCES, help="Planned dataset source to inspect.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned download actions without downloading data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw"),
        help="Planned root directory for raw datasets.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        print(DISABLED_DOWNLOAD_MESSAGE)
        sys.exit(1)

    try:
        sources = load_sources(CONFIG_PATH)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    source_config = sources[args.source]
    print_dry_run(args.source, source_config, args.output_dir)


if __name__ == "__main__":
    main()
