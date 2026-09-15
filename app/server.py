import io
 
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
 
IMG_SIZE = 224 
MODEL_PATH = "app/best_model.keras"
 
app = FastAPI(title="Age Estimator API")
 
model = tf.keras.models.load_model(MODEL_PATH)
 
 
class PredictionResponse(BaseModel):
    predicted_age: float
 
 
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.array(image, dtype=np.float32)          # shape (224, 224, 3), 0-255
    array = np.expand_dims(array, axis=0)               # shape (1, 224, 224, 3)
    return array
 
 
@app.get("/")
def index():
    return {"message": "Welcome to the Facial Age Estimator!"}
 
 
@app.post("/predict", response_model=PredictionResponse)
async def predict_age(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
 
    image_bytes = await file.read()
 
    try:
        input_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")
 
    prediction = model.predict(input_array, verbose=0)
    predicted_age = round(float(prediction[0][0]), 1)
 
    return PredictionResponse(predicted_age=predicted_age)
 
 
@app.get("/health")
async def health_check():
    return {"status": "ok"}
 