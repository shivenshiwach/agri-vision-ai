import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/training.yaml")
DEFAULT_DATASET_PATH = Path("datasets/processed/disease_classifier")
DEFAULT_OUTPUT_DIR = Path("models/disease_classifier")
DEFAULT_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "best.pt"
DEFAULT_CLASSES_PATH = DEFAULT_OUTPUT_DIR / "classes.json"
DEFAULT_CONFUSION_MATRIX_PATH = DEFAULT_OUTPUT_DIR / "confusion_matrix.png"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
MODEL_NAME = "efficientnet_b0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EfficientNet-B0 disease classifier checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Training YAML config path.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Dataset root with train/ and val/ folders.")
    parser.add_argument("--split", default="val", help="Dataset split folder to evaluate. Defaults to val.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to models/disease_classifier/best.pt.")
    parser.add_argument("--classes", type=Path, default=None, help="classes.json path. Defaults to models/disease_classifier/classes.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for confusion_matrix.png.")
    parser.add_argument("--confusion-matrix", type=Path, default=None, help="Confusion matrix PNG path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size. Defaults to config value or 64.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size. Defaults to config value, checkpoint value, or 224.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count. Defaults to config value or 4.")
    return parser.parse_args()


def resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config_value(config, "output_dir", DEFAULT_OUTPUT_DIR))

    settings = {
        "config_path": str(args.config),
        "dataset_path": args.dataset_path or Path(config_value(config, "dataset_path", DEFAULT_DATASET_PATH)),
        "split": args.split,
        "checkpoint": args.checkpoint or output_dir / "best.pt",
        "classes_path": args.classes or output_dir / "classes.json",
        "output_dir": output_dir,
        "confusion_matrix": args.confusion_matrix or output_dir / "confusion_matrix.png",
        "model_name": str(config_value(config, "model_name", MODEL_NAME)),
        "image_size": args.image_size if args.image_size is not None else config.get("image_size"),
        "batch_size": args.batch_size if args.batch_size is not None else int(config_value(config, "batch_size", DEFAULT_BATCH_SIZE)),
        "num_workers": (
            args.num_workers
            if args.num_workers is not None
            else int(config_value(config, "num_workers", DEFAULT_NUM_WORKERS))
        ),
    }

    if settings["model_name"] != MODEL_NAME:
        raise ValueError(f"Only {MODEL_NAME} is supported, got {settings['model_name']!r}.")
    if not settings["split"] or "/" in settings["split"] or "\\" in settings["split"]:
        raise ValueError("--split must be a simple folder name such as val or test.")
    if settings["batch_size"] <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if settings["num_workers"] < 0:
        raise ValueError("--num-workers must be greater than or equal to 0.")
    if settings["image_size"] is not None:
        settings["image_size"] = int(settings["image_size"])
        if settings["image_size"] <= 0:
            raise ValueError("--image-size must be a positive integer.")

    return settings


def choose_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classes(classes_path: Path, checkpoint: dict[str, Any]) -> list[str]:
    if classes_path.exists():
        with classes_path.open("r", encoding="utf-8") as classes_file:
            data = json.load(classes_file)
        if isinstance(data, dict) and isinstance(data.get("classes"), list):
            return [str(class_name) for class_name in data["classes"]]
        if isinstance(data, list):
            return [str(class_name) for class_name in data]
        raise ValueError(f"{classes_path} must contain a class list or a mapping with a classes list.")

    checkpoint_classes = checkpoint.get("class_names")
    if isinstance(checkpoint_classes, list):
        return [str(class_name) for class_name in checkpoint_classes]

    raise FileNotFoundError(f"Class names not found. Expected {classes_path} or class_names in checkpoint.")


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def load_dataset(dataset_path: Path, split: str, image_size: int, class_names: list[str]):
    from torchvision.datasets import ImageFolder

    split_dir = dataset_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {split_dir}")
    if image_count(split_dir) == 0:
        raise ValueError(f"No evaluation images found under {split_dir}")

    dataset = ImageFolder(split_dir, transform=build_transform(image_size))
    if dataset.classes != class_names:
        raise ValueError(
            "Evaluation class folders do not match saved class names. "
            f"Expected {class_names}, found {dataset.classes}."
        )
    return dataset


def create_dataloader(dataset, batch_size: int, num_workers: int, device):
    from torch.utils.data import DataLoader

    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
    return DataLoader(dataset, **loader_options)


def create_model(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    try:
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a supported disease classifier checkpoint.")
    return checkpoint


def evaluate(model, dataloader, num_classes: int, device):
    import torch

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            for actual, predicted in zip(labels.cpu(), predictions.cpu(), strict=False):
                confusion[int(actual), int(predicted)] += 1

    correct = int(confusion.diag().sum().item())
    total = int(confusion.sum().item())
    overall_accuracy = correct / total if total else 0.0
    per_class_accuracy = []
    for class_index in range(num_classes):
        class_total = int(confusion[class_index].sum().item())
        class_correct = int(confusion[class_index, class_index].item())
        class_accuracy = class_correct / class_total if class_total else 0.0
        per_class_accuracy.append((class_correct, class_total, class_accuracy))

    return confusion.tolist(), overall_accuracy, correct, total, per_class_accuracy


def text_size(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def save_confusion_matrix_png(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    class_count = len(class_names)
    cell_size = max(42, min(80, 720 // max(class_count, 1)))
    left_margin = 210
    top_margin = 160
    right_margin = 40
    bottom_margin = 120
    width = left_margin + class_count * cell_size + right_margin
    height = top_margin + class_count * cell_size + bottom_margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_value = max((value for row in matrix for value in row), default=0)
    draw.text((20, 20), "Confusion Matrix", fill="black", font=font)
    draw.text((left_margin, 52), "Predicted label", fill="black", font=font)
    draw.text((20, top_margin - 28), "Actual label", fill="black", font=font)

    for index, class_name in enumerate(class_names):
        label = f"{index}: {class_name}"
        y = top_margin + index * cell_size + cell_size // 2 - 5
        draw.text((20, y), label[:28], fill="black", font=font)

        x = left_margin + index * cell_size + 4
        draw.text((x, top_margin - 24), str(index), fill="black", font=font)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(255 - (180 * value / max_value)) if max_value else 255
            fill = (intensity, intensity, 255)
            x0 = left_margin + col_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(80, 80, 80))

            value_text = str(value)
            text_width, text_height = text_size(draw, value_text, font)
            draw.text(
                (x0 + (cell_size - text_width) / 2, y0 + (cell_size - text_height) / 2),
                value_text,
                fill="black",
                font=font,
            )

    legend_y = top_margin + class_count * cell_size + 24
    draw.text((20, legend_y), "Index mapping is shown on the left. Rows are actual classes; columns are predictions.", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()

    try:
        settings = resolved_settings(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not settings["checkpoint"].is_file():
        print(f"ERROR: Checkpoint not found: {settings['checkpoint']}")
        sys.exit(1)

    device = choose_device()
    try:
        checkpoint = load_checkpoint(settings["checkpoint"])
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if "model_state_dict" not in checkpoint:
        print(f"ERROR: {settings['checkpoint']} is not a supported disease classifier checkpoint.")
        sys.exit(1)

    try:
        class_names = load_classes(settings["classes_path"], checkpoint)
        image_size = settings["image_size"] or int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        dataset = load_dataset(settings["dataset_path"], settings["split"], image_size, class_names)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    dataloader = create_dataloader(dataset, settings["batch_size"], settings["num_workers"], device)
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    matrix, overall_accuracy, correct, total, per_class_accuracy = evaluate(
        model=model,
        dataloader=dataloader,
        num_classes=len(class_names),
        device=device,
    )

    save_confusion_matrix_png(matrix, class_names, settings["confusion_matrix"])

    print("Disease classifier evaluation")
    print("=============================")
    print(f"Checkpoint: {settings['checkpoint']}")
    print(f"Dataset: {settings['dataset_path'] / settings['split']}")
    print(f"Device: {device}")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    print("Per-class accuracy:")
    for class_name, (class_correct, class_total, class_accuracy) in zip(class_names, per_class_accuracy, strict=False):
        print(f"  - {class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")
    print(f"Confusion matrix: {settings['confusion_matrix']}")


if __name__ == "__main__":
    main()
