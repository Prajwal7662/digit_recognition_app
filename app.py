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
