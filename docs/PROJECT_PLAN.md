# Project Plan

Agri Vision AI will provide an image-based assistant for identifying crops, pests, diseases, and animal-related farm problems from uploaded field images.

## Phase 1: Taxonomy

- Define supported crops, pests, diseases, and animals.
- Map common names to scientific names where useful.
- Group labels by model type and expected user workflow.
- Review labels with agronomy and local farming context before dataset work begins.

## Phase 2: Dataset Collection

- Research public datasets and licensing.
- Create dataset manifests before downloading any files.
- Track source, class mapping, image counts, annotation type, and quality notes.
- Keep raw datasets outside Git and under ignored storage paths.

## Phase 3: Training

- Start with baseline models for crop, pest, disease, and animal detection or classification.
- Use reproducible configuration files for model paths, thresholds, and training parameters.
- Save checkpoints and runs only in ignored artifact directories.
- Compare models using validation metrics before API integration.

## Phase 4: Inference API

- Keep the FastAPI contract stable while model internals evolve.
- Add image validation and preprocessing.
- Load configured model artifacts when available.
- Return structured predictions with confidence, severity, detection boxes, and recommendations.

## Phase 5: App Integration

- Connect the API to the farm image upload experience.
- Provide clear recommendations and uncertainty handling.
- Add feedback capture so predictions can improve over time.
- Monitor latency, errors, and low-confidence results.
