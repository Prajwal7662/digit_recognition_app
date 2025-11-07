# app.py
# Author: Prajwal Mavkar
# Project: Digit Recognition App
# Description: Streamlit app to draw digits and predict using Gradient Boosting model

import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
import os

st.set_page_config(page_title="Digit Recognition App", page_icon="🔢")

st.title("🔢 Digit Recognition App")
st.markdown("Draw a digit (0–9) below, and the model will predict what you drew!")

# Check and load model safely
if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        st.success("✅ Model loaded successfully!")
    except Exception:
        st.warning("⚠️ Model corrupted — retraining...")
        X, y = load_digits(return_X_y=True)
        model = GradientBoostingClassifier().fit(X, y)
        joblib.dump(model, "model.pkl")
else:
    st.info("🧠 Training new model...")
    X, y = load_digits(return_X_y=True)
    model = GradientBoostingClassifier().fit(X, y)
    joblib.dump(model, "model.pkl")
    st.success("✅ New model trained and saved!")

# Canvas setup
st.subheader("🖌️ Draw a digit")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        # Convert drawing to grayscale and resize to 8x8 like MNIST digits
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.convert("L").resize((8, 8))
        img_array = np.array(img).reshape(1, -1)
        img_array = (16 - (img_array / 16)).astype(int)  # Normalize to MNIST scale

        prediction = model.predict(img_array)[0]
        st.success(f"🎯 Predicted Digit: **{prediction}**")
    else:
        st.warning("Please draw a digit before predicting!")

st.caption("Developed by Prajwal Mavkar 🧠")
