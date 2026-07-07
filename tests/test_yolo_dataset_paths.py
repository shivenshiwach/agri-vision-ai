from pathlib import Path

from scripts.validate_yolo_dataset import labels_dir_for_images_dir, resolve_split_path


def test_relative_yolo_dataset_root_resolves_from_working_directory(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs" / "yolo_dataset_template.yaml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.chdir(repo_root)

    config = {
        "path": "datasets/yolo_detector",
        "train": "images/train",
        "names": {0: "Pink Bollworm"},
    }

    assert resolve_split_path(config_path, config, "train") == (
        repo_root / "datasets" / "yolo_detector" / "images" / "train"
    ).resolve()


def test_yolo_labels_dir_matches_images_dir_split():
    images_dir = Path("/tmp/example/datasets/yolo_detector/images/val")

    assert labels_dir_for_images_dir(images_dir) == Path("/tmp/example/datasets/yolo_detector/labels/val")
