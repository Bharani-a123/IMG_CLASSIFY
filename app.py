# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# ✅ Allow frontend (HTML/JS) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Download model from Hugging Face Hub (cached after first download)
MODEL_REPO = "Bharani555/IMG_CLASSIFY"   # change to your repo
MODEL_FILE = "my_model.keras"            # adjust if different filename

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = tf.keras.models.load_model(model_path)

# ✅ Preprocess function (resize depends on your model input size)
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))  # change size if needed
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Serve index.html at root (same folder as app.py)
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "front.html"))

# ✅ Prediction Route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess(contents)

    preds = model.predict(input_tensor)
    predicted_class = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    # 👇 Add your class names in the correct order
    CLASS_NAMES = ["Battery", "Keyboard", "Microwave", "Mobile","Mouse", "PCB", "Player", "Printer", "Television", "Washing Machine"]  

    return JSONResponse({
        "prediction": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence
    })

