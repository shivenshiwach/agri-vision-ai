# Agri Vision AI

AI vision module scaffold for detecting crop, pest, disease, and animal problems from uploaded farm images.

This initial setup does not include datasets, trained models, checkpoints, or generated artifacts. The current API returns deterministic mock predictions so the project structure and integration contract can be developed safely.

## Project Structure

- `api/`: FastAPI app, response schemas, and placeholder inference function.
- `configs/`: Taxonomy and future model configuration.
- `docs/`: Project plan, dataset research notes, and taxonomy explanation.
- `scripts/`: Development utility scripts.
- `training/`: Placeholder for future training workflows.
- `tests/`: API tests.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
pytest
```

## Development Checks

```bash
python scripts/check_env.py
pytest
```

## API

Start the API:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/farm-image.jpg"
```
