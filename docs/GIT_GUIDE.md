# Git Guide

## Branches

Current branch during inspection: `shiven`.

Known branches:

- `main` / `origin/main`: taxonomy and initial dataset mapping baseline.
- `shiven` / `origin/shiven`: active branch with PlantVillage classifier path.

## Commit Style

Recent history uses short imperative subjects and milestone labels:

```text
Make full PlantVillage classifier the main training path
Add disease classifier training pipeline
milestone 11
```

Prefer concise commit subjects that name the behavior or milestone. Keep unrelated changes in separate commits.

## What Belongs In Git

Commit:

- API source code.
- Scripts.
- Config YAML.
- Docs.
- Lightweight taxonomy CSV files.
- Tests.

Do not commit:

- `datasets/`
- `models/`
- `runs/`
- `outputs/`
- raw images, annotations, checkpoints, `.pt`, `.onnx`, `.engine`
- `.env`
- virtual environments

## Current Dirty Workspace

Before these documentation changes, the workspace already had modified and untracked files related to multi-dataset disease preparation. Treat them as active local work. Do not revert them unless explicitly requested.

## Pre-Commit Checklist

Run:

```bash
git status --short
venv/bin/python -m pytest
git diff --check
```

For script changes, also run relevant `--help` and `--dry-run` commands. For docs-only changes, verify links, commands, and current status statements.

## Pull Request Checklist

Include:

- What changed and why.
- Commands run.
- Dataset/model artifact status.
- Any license or source review assumptions.
- Screenshots or sample API responses for user-facing API behavior.
- Known limitations and follow-up tasks.

