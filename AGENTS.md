# Agri Vision AI Project Brain

This file is the primary operating guide for future Codex work in this repository. Use it together with:

- `docs/PROJECT_CONTEXT.md`
- `docs/CURRENT_STATUS.md`
- `docs/ROADMAP.md`
- `docs/WORKFLOW.md`
- `docs/TESTING_GUIDE.md`
- `docs/GIT_GUIDE.md`
- `docs/MILESTONES.md`
- `docs/DECISIONS.md`

If this file and the repository disagree, inspect the repository first, state the discrepancy, and update documentation before making implementation decisions.

## Project Identity

Project name: **Agri Vision AI**.

Agri Vision AI is an agriculture AI vision platform. The long-term product direction is to analyze farm images and return structured predictions, severity, confidence, boxes when relevant, and farmer-facing recommendations.

The current working focus is **PlantVillage-style plant disease image classification**. The active implemented training path is an EfficientNet-B0 ImageFolder classifier using the full PlantVillage `raw/color` class-folder layout.

The API currently has **mock inference only**. `api/inference.py` returns deterministic placeholder predictions. Real model integration is future work and must wait until a checkpoint, preprocessing path, validation metrics, and failure behavior are documented.

## Long-Term Vision

Future modules may include:

- crop identification
- plant disease classification
- healthy vs diseased classification
- pest detection
- weed detection
- animal, livestock, or intrusion detection
- severity estimation
- advisory and recommendation layer
- API, frontend demo, and deployment workflow

Do not start these future modules unless explicitly requested. The current priority is repository stabilization and the PlantVillage classifier workflow.

## Current Repo Truth

Use repo inspection only; do not assume artifacts exist.

- Branch inspected: `shiven`.
- No local `datasets/` directory exists in this workspace.
- No local `models/` directory exists in this workspace.
- No trained checkpoint exists in this workspace.
- No generated evaluation artifacts exist in this workspace.
- API inference is mock-only.
- EfficientNet-B0 PlantVillage classifier scripts/configs exist:
  - `scripts/build_full_plantvillage_classifier_dataset.py`
  - `scripts/train_image_classifier.py`
  - `scripts/evaluate_image_classifier.py`
  - `configs/full_plantvillage_classifier.yaml`
  - `configs/full_plantvillage_training.yaml`
- Multi-dataset disease preparation exists as staging/inspection only; it does not train.
- YOLO support is planning/validation only; conversion and detector training are not implemented.
- Dataset/model artifacts must stay out of Git.

Ignored heavy paths include `datasets/`, `models/`, `runs/`, `outputs/`, raw data under `data/*`, checkpoints, ONNX/TensorRT exports, `.env`, and virtual environments.

## Required Phase Order

Do not jump phases unless the user explicitly says to.

1. **Repository stabilization**
2. **Dataset verification/acquisition**
3. **PlantVillage baseline training/evaluation**
4. **Inference script**
5. **Real API model integration**
6. **Frontend/demo**
7. **Deployment/production readiness**
8. **Future modules such as pest, animal, weed, or detector work**

Current recommended next milestone: **Milestone 14: repository stabilization before training**.

Milestone 14 includes:

- verify `requirements.txt`, especially the `httpx2` issue
- fix or document YOLO path resolution
- add `/predict` tests
- add dry-run tests for scripts
- document dataset license/version requirements

## Mandatory Task Workflow

For every future task:

1. Inspect files first. Use `rg --files`, `rg`, `sed`, and existing docs before deciding.
2. Explain the current understanding briefly.
3. List affected files before editing.
4. Give a short plan for non-trivial work.
5. Mention risks, missing artifacts, or assumptions.
6. Implement the smallest scoped change that satisfies the request.
7. Run relevant tests/checks.
8. Give a final summary with changed files, validation, and remaining risks.

Do not skip inspection because earlier context seemed sufficient. This repository changes through staged milestones, and local workspace state matters.

## Strict Guardrails

- Do not make broad refactors.
- Do not modify unrelated files.
- Do not retrain automatically.
- Do not download datasets unless explicitly asked.
- Do not overwrite checkpoints.
- Do not commit unless asked.
- Do not push unless asked.
- Do not force-push unless explicitly asked.
- Do not delete `datasets/`, `models/`, `runs/`, or `outputs/`.
- Do not silently fallback in model or API logic.
- Keep current API contracts stable unless asked.
- Keep generated datasets, model artifacts, metrics, and exports out of Git.
- Preserve mock inference until real model integration is explicitly requested and documented.

When model or API behavior is involved, fail clearly if a required artifact/config is missing. Do not hide missing model files behind fake success.

## Git Workflow

Before changes, run:

```bash
git status --short
git branch --show-current
```

If the working tree is dirty:

- explain existing changes
- distinguish user/pre-existing changes from your intended edits
- do not reset, stash, checkout, or commit unless explicitly asked
- work around unrelated dirty files

After changes, run:

```bash
git diff --stat
git diff --check
```

Then run relevant tests or dry-runs. If only documentation changed, `git diff --check` is usually sufficient, but run tests when documentation claims executable behavior.

## Testing Workflow

Use the repo's actual known-good environment:

```bash
venv/bin/python -m pytest
venv/bin/python scripts/check_env.py
```

Relevant lightweight checks include:

```bash
venv/bin/python scripts/train_image_classifier.py --help
venv/bin/python scripts/evaluate_image_classifier.py --help
venv/bin/python scripts/inspect_disease_datasets.py --help
venv/bin/python scripts/build_multi_disease_classifier_dataset.py --help
```

Dataset-related changes should use dry-runs first:

```bash
venv/bin/python scripts/download_sources.py --source plantvillage --dry-run
venv/bin/python scripts/inspect_datasets.py
venv/bin/python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
venv/bin/python scripts/inspect_disease_datasets.py --dry-run
venv/bin/python scripts/build_multi_disease_classifier_dataset.py --dry-run
```

Known local environment status:

- `venv/bin/python -m pytest` passed with 1 test.
- Default `python -m pytest` failed because FastAPI was missing from the default environment.
- `venv/bin/python scripts/check_env.py` reported Torch and Ultralytics installed, CUDA unavailable.

If commands fail due to environment or dependency issues, report the failure clearly. Do not hide or reinterpret failures as success.

## Coding And Documentation Preferences

- Keep code changes minimal.
- Prefer targeted patches over rewriting full files.
- Avoid excessive comments.
- Follow existing Python style: 4-space indentation, `snake_case` functions/files, `UPPER_SNAKE_CASE` constants, `PascalCase` Pydantic models.
- Keep API contracts in `api/schemas.py`, endpoint wiring in `api/main.py`, and model/inference logic behind `api/inference.py`.
- Prefer argparse `--dry-run` and `--confirm` for scripts that touch data.
- Keep final summaries short but complete.

## Project Structure

```text
api/                  FastAPI app, schemas, mock inference entry point
configs/              Taxonomy, source, YOLO, and classifier training config
data/taxonomy/        Lightweight tracked CSV metadata only
docs/                 Project context, status, roadmap, workflow, testing, git, milestones
scripts/              Dataset, inspection, preparation, training, evaluation utilities
tests/                Pytest tests
training/             Placeholder for future training workflows
```

## Current Commands

Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

PlantVillage dataset preview:

```bash
venv/bin/python scripts/build_full_plantvillage_classifier_dataset.py --dry-run
```

PlantVillage training command, only after dataset verification:

```bash
venv/bin/python scripts/train_image_classifier.py --config configs/full_plantvillage_training.yaml
```

PlantVillage evaluation command, only after a checkpoint exists:

```bash
venv/bin/python scripts/evaluate_image_classifier.py --config configs/full_plantvillage_training.yaml
```

## Commit And PR Expectations

Recent history uses concise imperative or milestone-style subjects, for example `Make full PlantVillage classifier the main training path` and `milestone 11`.

Pull requests should include:

- what changed and why
- commands run
- dataset/model artifact status
- license/source assumptions when data is involved
- screenshots or sample API responses for user-facing API changes
- known limitations and follow-up tasks

Do not create commits or PRs unless the user asks.
