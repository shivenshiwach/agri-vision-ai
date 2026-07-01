from pathlib import Path

import yaml


CLASSES_PATH = Path("configs/classes.yaml")


def main() -> None:
    with CLASSES_PATH.open("r", encoding="utf-8") as classes_file:
        classes = yaml.safe_load(classes_file)

    total = 0
    for section, items in classes.items():
        count = len(items)
        total += count
        print(f"{section}: {count}")

    animal_related = sum(len(classes.get(section, [])) for section in ("animals", "birds", "rodents", "reptiles"))
    print(f"animal_related_total: {animal_related}")
    print(f"total_classes: {total}")


if __name__ == "__main__":
    main()
