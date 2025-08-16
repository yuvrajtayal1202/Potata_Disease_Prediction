from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# CORS Middleware (for frontend support if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving URL
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# Class names from your model
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Image preprocessing function
def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize if your model was trained that way
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension

        json_data = {
            "instances": img_batch.tolist()
        }

        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()  # raise exception for bad status

        prediction = np.array(response.json()["predictions"][0])
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except requests.exceptions.RequestException as e:
        return {"error": "TensorFlow Serving not available", "details": str(e)}
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
