from api.schemas import DetectionBox, PredictionResult, Recommendation


def predict_image(image_bytes: bytes) -> PredictionResult:
    """Return a deterministic mock prediction until real models are wired in."""
    _ = image_bytes

    return PredictionResult(
        crop="Cotton",
        problem_type="pest",
        detected_problem="Pink Bollworm",
        scientific_name="Pectinophora gossypiella",
        severity="medium",
        confidence=0.86,
        recommendations=[
            Recommendation(
                action="Inspect affected bolls",
                details="Open a few suspect bolls and check for larval damage before treatment decisions.",
                priority="high",
            ),
            Recommendation(
                action="Use integrated pest management",
                details="Prefer pheromone traps, field sanitation, and locally recommended controls.",
                priority="medium",
            ),
        ],
        boxes=[
            DetectionBox(
                x_min=0.21,
                y_min=0.18,
                x_max=0.74,
                y_max=0.69,
                label="Pink Bollworm damage",
                confidence=0.86,
            )
        ],
    )
