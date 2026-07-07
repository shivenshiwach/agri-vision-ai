import subprocess
import sys


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
