import joblib
import os
import streamlit as st

model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please run train_model.py first.")
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
# app.py
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Character Recognition", layout="centered")
st.title("✏️ Handwritten Character Recognition (A–Z & 0–9)")
st.markdown("Draw any **alphabet (A–Z)** or **digit (0–9)** below and click *Predict*!")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction logic
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = (255 - img_array) / 255.0  # invert + normalize
    img_flat = img_array.flatten().reshape(1, -1)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_flat)[0]
        pred_char = chr(int(prediction) + 65)
        st.success(f"🧠 Predicted Character: **{pred_char}**")

        st.image(img, caption="Processed Input", width=100)
