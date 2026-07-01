from fastapi import FastAPI, File, HTTPException, UploadFile

from api.inference import predict_image
from api.schemas import PredictionResult


app = FastAPI(title="Agri Vision AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "agri-vision-ai"}


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    return predict_image(image_bytes)
