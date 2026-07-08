import subprocess
import sys


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_crop_land_inspector_dry_run_does_not_copy_or_train():
    result = run_script("scripts/inspect_crop_land_datasets.py", "--dry-run")

    assert result.returncode == 0
    assert "Crop/land dataset inspection" in result.stdout
    assert "No images were copied. No training was started." in result.stdout


def test_crop_land_builder_dry_run_does_not_copy_files():
    result = run_script("scripts/build_crop_land_classifier_dataset.py", "--dry-run")

    assert result.returncode == 0
    assert "Crop/land classifier dataset preparation plan" in result.stdout
    assert "No files were copied." in result.stdout


def test_crop_land_builder_rejects_disabled_dataset_without_override():
    result = run_script("scripts/build_crop_land_classifier_dataset.py", "--dataset", "plantdoc_crop_land", "--dry-run")

    assert result.returncode == 1
    assert "Dataset key(s) are disabled: plantdoc_crop_land" in result.stdout
    assert "Pass --include-disabled" in result.stdout
