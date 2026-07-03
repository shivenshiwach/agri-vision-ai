import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/training.yaml")
DEFAULT_DATASET_PATH = Path("datasets/processed/disease_classifier")
DEFAULT_OUTPUT_DIR = Path("models/disease_classifier")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42
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
        description="Train an EfficientNet-B0 disease classifier with PyTorch and torchvision."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Training YAML config path.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Dataset root with train/ and val/ folders.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for best.pt, last.pt, classes.json, and history.json.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs. Defaults to config value or 20.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size. Defaults to config value or 64.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size. Defaults to config value or 224.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate. Defaults to config value or 0.001.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count. Defaults to config value or 4.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config value or 42.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained EfficientNet-B0 weights.")
    return parser.parse_args()


def resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)

    settings = {
        "config_path": str(args.config),
        "dataset_path": args.dataset_path or Path(config_value(config, "dataset_path", DEFAULT_DATASET_PATH)),
        "output_dir": args.output_dir or Path(config_value(config, "output_dir", DEFAULT_OUTPUT_DIR)),
        "model_name": str(config_value(config, "model_name", MODEL_NAME)),
        "image_size": args.image_size if args.image_size is not None else int(config_value(config, "image_size", DEFAULT_IMAGE_SIZE)),
        "batch_size": args.batch_size if args.batch_size is not None else int(config_value(config, "batch_size", DEFAULT_BATCH_SIZE)),
        "epochs": args.epochs if args.epochs is not None else int(config_value(config, "epochs", DEFAULT_EPOCHS)),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else float(config_value(config, "learning_rate", DEFAULT_LEARNING_RATE))
        ),
        "num_workers": (
            args.num_workers
            if args.num_workers is not None
            else int(config_value(config, "num_workers", DEFAULT_NUM_WORKERS))
        ),
        "seed": args.seed if args.seed is not None else int(config_value(config, "seed", DEFAULT_SEED)),
        "pretrained": not args.no_pretrained,
    }

    if settings["model_name"] != MODEL_NAME:
        raise ValueError(f"Only {MODEL_NAME} is supported, got {settings['model_name']!r}.")
    if settings["image_size"] <= 0:
        raise ValueError("--image-size must be a positive integer.")
    if settings["batch_size"] <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if settings["epochs"] <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if settings["learning_rate"] <= 0:
        raise ValueError("--learning-rate must be greater than 0.")
    if settings["num_workers"] < 0:
        raise ValueError("--num-workers must be greater than or equal to 0.")

    return settings


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int):
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def image_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def load_datasets(dataset_path: Path, image_size: int):
    from torchvision.datasets import ImageFolder

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")
    if image_count(train_dir) == 0:
        raise ValueError(f"No training images found under {train_dir}")
    if image_count(val_dir) == 0:
        raise ValueError(f"No validation images found under {val_dir}")

    train_transform, val_transform = build_transforms(image_size)
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train/ and val/ class folders must match exactly.")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size: int, num_workers: int, device):
    from torch.utils.data import DataLoader

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader


def create_model(num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, dataloader, criterion, optimizer, device, training: bool) -> tuple[float, float]:
    import torch

    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def checkpoint_payload(
    *,
    epoch: int,
    model,
    optimizer,
    class_names: list[str],
    class_to_idx: dict[str, int],
    settings: dict[str, Any],
    metrics: dict[str, float],
    include_optimizer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_name": MODEL_NAME,
        "image_size": settings["image_size"],
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": {
            "dataset_path": str(settings["dataset_path"]),
            "output_dir": str(settings["output_dir"]),
            "epochs": settings["epochs"],
            "batch_size": settings["batch_size"],
            "learning_rate": settings["learning_rate"],
            "num_workers": settings["num_workers"],
            "seed": settings["seed"],
            "pretrained": settings["pretrained"],
        },
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def main() -> None:
    args = parse_args()

    try:
        settings = resolved_settings(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    import torch
    import torch.nn as nn

    set_seed(settings["seed"])
    device = choose_device()

    try:
        train_dataset, val_dataset = load_datasets(settings["dataset_path"], settings["image_size"])
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    class_names = list(train_dataset.classes)
    class_to_idx = dict(train_dataset.class_to_idx)
    settings["output_dir"].mkdir(parents=True, exist_ok=True)

    save_json(settings["output_dir"] / "classes.json", {"classes": class_names, "class_to_idx": class_to_idx})

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        device=device,
    )

    model = create_model(num_classes=len(class_names), pretrained=settings["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    history: dict[str, Any] = {
        "config": {
            "dataset_path": str(settings["dataset_path"]),
            "output_dir": str(settings["output_dir"]),
            "model_name": MODEL_NAME,
            "image_size": settings["image_size"],
            "batch_size": settings["batch_size"],
            "epochs": settings["epochs"],
            "learning_rate": settings["learning_rate"],
            "optimizer": "adam",
            "loss": "cross_entropy",
            "device": str(device),
            "num_workers": settings["num_workers"],
            "seed": settings["seed"],
            "pretrained": settings["pretrained"],
        },
        "classes": class_names,
        "epochs": [],
        "best_epoch": None,
        "best_val_accuracy": None,
    }

    print("Disease classifier training")
    print("===========================")
    print(f"Dataset: {settings['dataset_path']}")
    print(f"Output: {settings['output_dir']}")
    print(f"Classes: {len(class_names)}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Device: {device}")

    best_val_accuracy = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, training=False)

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": settings["learning_rate"],
        }
        history["epochs"].append({"epoch": epoch, **metrics})

        last_payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            class_names=class_names,
            class_to_idx=class_to_idx,
            settings=settings,
            metrics=metrics,
            include_optimizer=True,
        )
        torch.save(last_payload, settings["output_dir"] / "last.pt")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            history["best_epoch"] = epoch
            history["best_val_accuracy"] = best_val_accuracy
            best_payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                class_names=class_names,
                class_to_idx=class_to_idx,
                settings=settings,
                metrics=metrics,
                include_optimizer=False,
            )
            torch.save(best_payload, settings["output_dir"] / "best.pt")

        save_json(settings["output_dir"] / "history.json", history)

        print(
            f"Epoch {epoch}/{settings['epochs']} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"best val acc {best_val_accuracy:.4f}"
        )

    print("Training complete.")
    print(f"Best model: {settings['output_dir'] / 'best.pt'}")
    print(f"Latest checkpoint: {settings['output_dir'] / 'last.pt'}")


if __name__ == "__main__":
    main()
