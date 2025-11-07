import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import load_digits

# Load trained model
model = joblib.load("model.pkl")

st.title("✏️ Handwritten Digit Recognition using Gradient Boosting")
st.write("Draw a digit (0–9) below and let the model predict it!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black color
    stroke_width=10,
    stroke_color="#FFFFFF",  # White pen
    background_color="#000000",  # Black background
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

# When user draws something
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((8, 8))  # Digits dataset images are 8x8
    img_array = np.array(img)

    # Preprocess: scale to match the dataset (0–16 range)
    img_array = (16 - (img_array / 16)).astype(np.float64)
    img_flat = img_array.flatten().reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(img_flat)[0]
        st.success(f"🧠 Predicted Digit: **{prediction}**")

        # Optional: visualize grayscale input
        st.image(img_array, caption="Processed Input", width=100)
