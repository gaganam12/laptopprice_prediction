import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load encoders, scaler, and model
feature_encoders = joblib.load("feature_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("ID3_entropy.pkl")  # Change to another model if needed

# App Title
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")

# Sidebar: Welcome & Instructions
st.sidebar.title("👋 Welcome")
st.sidebar.info(
    """
    **Developed by Gagana M**

    Use this app to predict the **price category** of a laptop based on its specifications.

    🔧 **Instructions**:
    - Select or enter the laptop's specifications.
    - Click on the predicted category to see the estimated price range.
    """
)

# User inputs
manufacturer = st.selectbox("Manufacturer", feature_encoders["Manufacturer"].classes_)
category = st.selectbox("Category", feature_encoders["Category"].classes_)
cpu = st.selectbox("CPU", feature_encoders["CPU"].classes_)
gpu = st.selectbox("GPU", feature_encoders["GPU"].classes_)
os = st.selectbox("Operating System", feature_encoders["Operating System"].classes_)
ram = st.number_input("RAM (GB)", min_value=2, max_value=128, step=2)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, step=0.1)
hdd = st.number_input("HDD Storage (GB)", min_value=0, max_value=2000, step=100)
ssd = st.number_input("SSD Storage (GB)", min_value=0, max_value=2000, step=100)

# Encode categorical inputs
input_dict = {
    "Manufacturer": feature_encoders["Manufacturer"].transform([manufacturer])[0],
    "Category": feature_encoders["Category"].transform([category])[0],
    "CPU": feature_encoders["CPU"].transform([cpu])[0],
    "GPU": feature_encoders["GPU"].transform([gpu])[0],
    "Operating System": feature_encoders["Operating System"].transform([os])[0],
    "RAM": ram,
    "Weight": weight,
    "Screen Size": screen_size,
    "HDD": hdd,
    "SSD": ssd
}

# Create DataFrame in correct order
input_df = pd.DataFrame([input_dict])
expected_columns = scaler.feature_names_in_
input_df = input_df[expected_columns]

# Scale and predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
predicted_label = target_encoder.inverse_transform(prediction)[0]

# Define price range descriptions
price_ranges = {
    "Low": "₹20,000 – ₹40,000",
    "Medium": "₹50,000 – ₹90,000",
    "High": "₹1,00,000 – ₹3,00,000"
}

# Display prediction
st.subheader("🎯 Predicted Price Category:")
st.success(f"{predicted_label} ({price_ranges[predicted_label]})")

# Footer
st.markdown("---")
st.markdown("<center>Developed by Gagana M</center>", unsafe_allow_html=True)
