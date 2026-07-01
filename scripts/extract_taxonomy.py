from pathlib import Path

import yaml


TAXONOMY_PATH = Path("configs/classes.yaml")


def main() -> None:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as taxonomy_file:
        taxonomy = yaml.safe_load(taxonomy_file)

    for section, items in taxonomy.items():
        print(f"{section}: {len(items)} classes")
        for item in items:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
