import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_crop_land_downloader():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "download_crop_land_sources.py"
    spec = importlib.util.spec_from_file_location("download_crop_land_sources_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_download_sources_plantvillage_dry_run_does_not_download():
    result = run_script("scripts/download_sources.py", "--source", "plantvillage", "--dry-run")

    assert result.returncode == 0
    assert "Dataset source dry run" in result.stdout
    assert "No files were downloaded. No directories were created." in result.stdout


@pytest.mark.parametrize("source", ["eurosat", "deepweeds", "plantdoc_crop_land"])
def test_download_sources_crop_land_dry_runs_do_not_download(source, tmp_path):
    output_dir = tmp_path / "raw"
    result = run_script("scripts/download_sources.py", "--source", source, "--output-dir", str(output_dir), "--dry-run")

    assert result.returncode == 0
    assert "Dataset source dry run" in result.stdout
    assert f"Source key: {source}" in result.stdout
    assert "No files were downloaded. No directories were created." in result.stdout
    assert not output_dir.exists()


def test_deepweeds_gdown_detection_prefers_current_python_module(monkeypatch):
    module = load_crop_land_downloader()

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object() if name == "gdown" else None)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    assert module.require_gdown() == [sys.executable, "-m", "gdown"]


def test_download_sources_rejects_unimplemented_confirmed_download():
    result = run_script("scripts/download_sources.py", "--source", "open_images", "--confirm")

    assert result.returncode == 1
    assert "downloads are disabled in config for source: open_images" in result.stdout


def test_full_plantvillage_builder_dry_run_does_not_copy_files():
    result = run_script("scripts/build_full_plantvillage_classifier_dataset.py", "--dry-run")

    assert result.returncode == 0
    assert "Full PlantVillage classifier dataset build plan" in result.stdout
    assert "No files were copied." in result.stdout


def test_yolo_dataset_planner_does_not_create_files():
    result = run_script("scripts/build_yolo_dataset.py")

    assert result.returncode == 0
    assert "YOLO dataset build plan" in result.stdout
    assert "No images or labels were moved, copied, converted, or created." in result.stdout
