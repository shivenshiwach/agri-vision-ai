from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_predict_returns_mock_prediction_for_image_upload():
    response = client.post(
        "/predict",
        files={"file": ("field.jpg", b"not-real-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["crop"] == "Cotton"
    assert payload["problem_type"] == "pest"
    assert payload["detected_problem"] == "Pink Bollworm"
    assert payload["confidence"] == 0.86
    assert len(payload["recommendations"]) == 2
    assert payload["boxes"] == [
        {
            "x_min": 0.21,
            "y_min": 0.18,
            "x_max": 0.74,
            "y_max": 0.69,
            "label": "Pink Bollworm damage",
            "confidence": 0.86,
        }
    ]


def test_predict_rejects_non_image_upload():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file must be an image."}


def test_predict_rejects_empty_image_upload():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded image is empty."}
