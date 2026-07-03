import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


DEFAULT_MODEL_PATH = Path("models/disease_classifier/best.pt")
DEFAULT_CLASSES_PATH = Path("models/disease_classifier/classes.json")
IMAGE_SIZE = 224
DEFAULT_TOP_K = 3
MODEL_NAME = "efficientnet_b0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EfficientNet-B0 disease classifier inference on one image."
    )
    parser.add_argument("image_path", nargs="?", type=Path, help="Path to a single image.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a single image.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top predictions to print.")
    return parser.parse_args()


def resolve_image_path(args: argparse.Namespace) -> Path:
    if args.image_path is not None and args.image is not None and args.image_path != args.image:
        raise ValueError("Provide the image path either positionally or with --image, not both.")

    image_path = args.image or args.image_path
    if image_path is None:
        raise ValueError("Missing image path. Use: python scripts/predict_disease.py image.jpg")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def load_classes(classes_path: Path) -> list[str]:
    if not classes_path.is_file():
        raise FileNotFoundError(f"Class names file not found: {classes_path}")

    with classes_path.open("r", encoding="utf-8") as classes_file:
        data = json.load(classes_file)

    if isinstance(data, dict) and isinstance(data.get("classes"), list):
        classes = [str(class_name) for class_name in data["classes"]]
    elif isinstance(data, list):
        classes = [str(class_name) for class_name in data]
    else:
        raise ValueError(f"{classes_path} must contain a class list or a mapping with a classes list.")

    if not classes:
        raise ValueError(f"{classes_path} does not contain any class names.")
    return classes


def load_checkpoint(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{model_path} is not a supported disease classifier checkpoint.")
    return checkpoint


def choose_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    try:
        model = models.efficientnet_b0(weights=None)
    except (AttributeError, TypeError):
        model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_image(image_path: Path):
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc


def predict(image_path: Path, class_names: list[str], checkpoint: dict[str, Any], top_k: int) -> list[tuple[str, float]]:
    import torch

    device = choose_device()
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform()
    image = load_image(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    return [(class_names[int(index)], float(value.item())) for value, index in zip(values.cpu(), indices.cpu(), strict=False)]


def print_predictions(predictions: list[tuple[str, float]]) -> None:
    best_class, best_confidence = predictions[0]

    print(f"Prediction: {best_class}")
    print(f"Confidence: {best_confidence * 100:.1f}%")
    print()
    print("Top predictions:")
    for rank, (class_name, confidence) in enumerate(predictions, start=1):
        print(f"{rank}. {class_name} {confidence * 100:.1f}%")


def main() -> None:
    args = parse_args()

    try:
        image_path = resolve_image_path(args)
        if args.top_k <= 0:
            raise ValueError("--top-k must be a positive integer.")
        class_names = load_classes(DEFAULT_CLASSES_PATH)
        checkpoint = load_checkpoint(DEFAULT_MODEL_PATH)
        predictions = predict(image_path, class_names, checkpoint, args.top_k)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print_predictions(predictions)


if __name__ == "__main__":
    main()
