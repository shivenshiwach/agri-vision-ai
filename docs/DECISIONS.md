# Engineering Decision Log

This file is the permanent engineering decision log for Agri Vision AI.

Rules:

- Every major technical decision gets a unique ID.
- Never rewrite history.
- Append new decisions instead of modifying old ones unless correcting factual mistakes.
- If a decision changes, add a new entry and mark the old entry as superseded.

## DEC-0001: Use PlantVillage Classifier As Current Baseline

**Date:** 2026-07-07

**Status:** Accepted

**Context:** The repository has a working full PlantVillage ImageFolder preparation path plus reusable EfficientNet-B0 training and evaluation scripts. No local datasets, trained checkpoints, or model artifacts exist in this workspace.

**Decision:** The current baseline focus is PlantVillage-style plant disease image classification using the full PlantVillage `raw/color` class-folder layout.

**Reason:** This path is the most complete implemented workflow and gives the project a controlled classification baseline before adding more modules or production inference.

**Consequences:** Work should prioritize dataset verification, PlantVillage baseline training, evaluation, and documented checkpoint readiness. Other model families should not interrupt this path unless explicitly requested.

**Related files:** `configs/full_plantvillage_classifier.yaml`, `configs/full_plantvillage_training.yaml`, `scripts/build_full_plantvillage_classifier_dataset.py`, `scripts/train_image_classifier.py`, `scripts/evaluate_image_classifier.py`, `docs/FULL_PLANTVILLAGE_CLASSIFIER.md`, `docs/FULL_PLANTVILLAGE_TRAINING.md`

**Related milestone:** Milestone 12 / Phase 3

## DEC-0002: Keep API Inference Mock-Only Until A Validated Checkpoint Exists

**Date:** 2026-07-07

**Status:** Accepted

**Context:** The FastAPI app exposes `/health` and `/predict`, and `api/inference.py` currently returns deterministic mock predictions. No trained checkpoint exists locally, and preprocessing plus validation metrics are not yet documented for an inference artifact.

**Decision:** Keep API inference mock-only until a validated checkpoint, preprocessing path, metrics, and failure behavior are documented.

**Reason:** A fake or silently degraded model integration would make API behavior misleading. The current mock keeps the response contract available while training and validation mature.

**Consequences:** Real model loading must be introduced deliberately after checkpoint validation. Missing artifacts should fail clearly instead of silently falling back to mock behavior.

**Related files:** `api/main.py`, `api/inference.py`, `api/schemas.py`, `tests/test_api_predict.py`, `docs/CURRENT_STATUS.md`, `docs/ROADMAP.md`

**Related milestone:** Milestone 14 / Phase 5

## DEC-0003: Keep Heavy Dataset And Model Artifacts Out Of Git

**Date:** 2026-07-07

**Status:** Accepted

**Context:** The project will handle raw datasets, processed datasets, checkpoints, exported models, runs, metrics, and generated outputs. These assets can be large, licensed, reproducibility-sensitive, or environment-specific.

**Decision:** Keep `datasets/`, `models/`, `runs/`, `outputs/`, raw data under `data/*`, checkpoints, ONNX/TensorRT exports, `.env`, and virtual environments outside Git.

**Reason:** Git should track code, configs, tests, lightweight taxonomy metadata, and documentation. Heavy or licensed artifacts belong in ignored local/server storage with separate provenance.

**Consequences:** Dataset and model workflows must document source metadata, local paths, license status, and validation outputs without committing the artifacts themselves.

**Related files:** `.gitignore`, `data/README.md`, `docs/REAL_DATASETS.md`, `docs/DATA_ACQUISITION_WORKFLOW.md`, `docs/GIT_GUIDE.md`, `AGENTS.md`

**Related milestone:** Milestone 4 / Phase 2

## DEC-0004: Follow Milestone-Based Development

**Date:** 2026-07-07

**Status:** Accepted

**Context:** The repository has evolved through staged commits for taxonomy, dataset planning, guarded downloads, inspection, PlantVillage classifier work, disease preparation, stabilization, and operating documentation.

**Decision:** Continue using milestone-based development with scoped phases and explicit validation before moving to later phases.

**Reason:** The project combines ML data, training, API behavior, and future product surfaces. Milestone boundaries reduce accidental scope creep and protect reproducibility.

**Consequences:** Future work should state the milestone, affected files, risks, and validation commands. Do not jump phases unless explicitly requested.

**Related files:** `docs/MILESTONES.md`, `docs/ROADMAP.md`, `docs/WORKFLOW.md`, `AGENTS.md`

**Related milestone:** All milestones

## DEC-0005: Complete Repository Stabilization Before Training

**Date:** 2026-07-07

**Status:** Accepted

**Context:** Before any dataset download or model training, the repo needed basic stabilization: dependency verification, YOLO path handling, `/predict` tests, dry-run tests, and dataset source metadata requirements.

**Decision:** Milestone 14 stabilization is completed before moving to dataset acquisition or PlantVillage baseline training.

**Reason:** Training should start from a predictable repository state with known test commands, documented environment behavior, and safe dry-run coverage.

**Consequences:** The next milestone can focus on dataset verification/acquisition. Any future stabilization gaps should be logged separately rather than mixed into training work.

**Related files:** `requirements.txt`, `scripts/validate_yolo_dataset.py`, `tests/test_api_predict.py`, `tests/test_safe_script_dry_runs.py`, `tests/test_yolo_dataset_paths.py`, `docs/REAL_DATASETS.md`, `docs/TESTING_GUIDE.md`

**Related milestone:** Milestone 14 / Phase 1

## DEC-0006: Postpone Future Modules Until The Classifier Pipeline Is Complete

**Date:** 2026-07-07

**Status:** Accepted

**Context:** Long-term modules may include crop identification, pest detection, weed detection, animal/livestock/intrusion detection, severity estimation, advisory layers, frontend demo work, and deployment.

**Decision:** Future modules are postponed until the PlantVillage classifier pipeline reaches dataset verification, baseline training/evaluation, and documented checkpoint readiness, unless the user explicitly requests otherwise.

**Reason:** The classifier pipeline is the most mature path. Starting additional modules too early would fragment effort and introduce unvalidated assumptions.

**Consequences:** Pest, weed, animal, detector, frontend, and deployment work remain roadmap items. Planning docs may mention them, but implementation should stay focused on the current phase.

**Related files:** `docs/ROADMAP.md`, `docs/MODEL_STRATEGY.md`, `docs/PROJECT_CONTEXT.md`, `AGENTS.md`

**Related milestone:** Phase 8

## DEC-0007: Use Dry-Run-First Dataset Workflows

**Date:** 2026-07-07

**Status:** Accepted

**Context:** Dataset operations can download large assets, copy images, stage normalized datasets, or overwrite local directories. Several scripts support `--dry-run` and require `--confirm` before writes.

**Decision:** Dataset acquisition, inspection, and preparation should use dry-run-first workflows. Confirmed writes require explicit user intent and source review.

**Reason:** Dry-runs make planned filesystem changes visible and reduce the risk of accidental downloads, overwrites, or undocumented data mixing.

**Consequences:** New data-touching scripts should keep this convention. Confirmed downloads or copies must wait for license, version, storage path, and label mapping review.

**Related files:** `scripts/download_sources.py`, `scripts/build_full_plantvillage_classifier_dataset.py`, `scripts/build_multi_disease_classifier_dataset.py`, `scripts/inspect_disease_datasets.py`, `docs/DATA_ACQUISITION_WORKFLOW.md`, `docs/REAL_DATASETS.md`

**Related milestone:** Milestone 4 / Milestone 13 / Phase 2
