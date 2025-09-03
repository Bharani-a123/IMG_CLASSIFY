# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

# ✅ Download model from Hugging Face (cached locally after first time)
MODEL_REPO = "Bharani555/IMG_CLASSIFY"   
MODEL_FILE = "my_model.keras"            

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = tf.keras.models.load_model(model_path)

# ✅ Class names
CLASS_NAMES = ["Battery", "Keyboard", "Microwave", "Mobile",
               "Mouse", "PCB", "Player", "Printer", "Television", "Washing Machine"]

# ✅ Preprocess
def preprocess(image):
    img = image.resize((224, 224))  # adjust size to match your model
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Streamlit UI
st.title("🔍 Image Classification App")
st.write("Upload an image to predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    input_tensor = preprocess(image)
    preds = model.predict(input_tensor)
    predicted_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    st.success(f"✅ Prediction: **{CLASS_NAMES[predicted_class]}** "
               f"(Confidence: {confidence:.2f})")
