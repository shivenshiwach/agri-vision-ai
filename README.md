# Agri Vision AI

AI vision module scaffold for detecting crop, pest, disease, and animal problems from uploaded farm images.

This setup does not include datasets, trained models, checkpoints, or generated artifacts. The current API returns deterministic mock predictions so the project structure and integration contract can be developed safely.

## Project Structure

- `api/`: FastAPI app, response schemas, and placeholder inference function.
- `configs/`: Taxonomy and future model configuration.
- `data/taxonomy/`: Lightweight CSV taxonomy and dataset mapping metadata.
- `docs/`: Project plan, dataset research notes, and taxonomy explanation.
- `scripts/`: Development utility scripts.
- `training/`: Placeholder for future training workflows.
- `tests/`: API tests.

## AI Roadmap

Milestone 1 defines the Maharashtra agriculture taxonomy and dataset mapping. It covers crop, pest, disease, animal, bird, rodent, and reptile classes from the project document.

Planned model path:

1. Build a crop classifier to route uploaded images by crop.
2. Train focused pest detectors for high-priority visible insects and damage.
3. Train disease detectors or classifiers for common leaf, fruit, and storage symptoms.
4. Train animal detectors for wildlife, birds, rodents, and reptiles in field or storage settings.
5. Add severity estimation using detected object counts, affected area, crop stage, and economic thresholds.

Recommended first training targets:

- Crops: Cotton, Soybean, Sugarcane, Paddy (Rice), Wheat, Grapes, Pomegranate, Banana, Tomato, and Onion.
- Pests: Pink Bollworm, Whitefly, Thrips, Aphids, Fruit Borer, Stem Borer, and Fall Armyworm.
- Diseases: Rust, Powdery Mildew, Downy Mildew, Early Blight, Late Blight, Bacterial Blight, Leaf Curl Virus, and Blast Disease.
- Animals: Wild Boar, Monkeys, Birds, Parrots, Sparrows, Rats, and Snakes.

Before training, review `docs/DATASET_GAPS.md` and confirm dataset licenses, class mappings, and annotation formats. Keep raw data under ignored dataset directories.

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
