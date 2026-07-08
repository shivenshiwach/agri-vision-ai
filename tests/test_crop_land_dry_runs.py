import importlib.util
import subprocess
import sys
from pathlib import Path


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def load_crop_land_registry():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "crop_land_dataset_registry.py"
    spec = importlib.util.spec_from_file_location("crop_land_dataset_registry_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_crop_land_output_image_name_is_short_deterministic_and_preserves_extension():
    registry = load_crop_land_registry()
    class_dir = Path("datasets/raw/plantdoc_crop_land/crop_closeup")
    long_source_name = (
        "Corn_leaf_blight_train_image_with_a_very_long_scraped_source_name_"
        "and_extra_annotation_tokens_that_used_to_make_processed_paths_too_long.JPG"
    )
    image_path = class_dir / long_source_name

    filename = registry.output_image_name(
        "plantdoc_crop_land",
        "crop_closeup",
        "crop_closeup",
        "train",
        class_dir,
        image_path,
    )
    repeated_filename = registry.output_image_name(
        "plantdoc_crop_land",
        "crop_closeup",
        "crop_closeup",
        "train",
        class_dir,
        image_path,
    )
    sibling_filename = registry.output_image_name(
        "plantdoc_crop_land",
        "crop_closeup",
        "crop_closeup",
        "train",
        class_dir,
        class_dir / f"sibling_{long_source_name}",
    )

    assert filename == repeated_filename
    assert filename != sibling_filename
    assert filename.startswith("plantdoc_crop_land__crop_closeup__train__")
    assert filename.endswith(".jpg")
    assert len(filename) < 120
    assert long_source_name not in filename


def test_crop_land_builder_rejects_disabled_dataset_without_override():
    result = run_script("scripts/build_crop_land_classifier_dataset.py", "--dataset", "plantdoc_crop_land", "--dry-run")

    assert result.returncode == 1
    assert "Dataset key(s) are disabled: plantdoc_crop_land" in result.stdout
    assert "Pass --include-disabled" in result.stdout
