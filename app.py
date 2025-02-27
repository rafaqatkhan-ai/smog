import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained models
cnn_model = tf.keras.models.load_model("cnn_model.h5")
dnn_model = tf.keras.models.load_model("dnn_model.h5")
lstm_model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training

# Define smog levels
smog_levels = {2: "Moderate", 3: "Unhealthy Sensitive", 4: "Unhealthy", 5: "Very Unhealthy", 6: "Hazardous"}

st.title("PM10 & PM2.5 Smog Level Prediction")

# Input fields
scaled_aqi = st.number_input("Scaled AQI (1-5)", min_value=1.0, max_value=5.0, step=0.1)
co = st.number_input("CO (µg/m³)")
no = st.number_input("NO (µg/m³)")
no2 = st.number_input("NO2 (µg/m³)")
o3 = st.number_input("O3 (µg/m³)")
so2 = st.number_input("SO2 (µg/m³)")
pm25 = st.number_input("PM2.5 (µg/m³)")
pm10 = st.number_input("PM10 (µg/m³)")
nh3 = st.number_input("NH3 (µg/m³)")

if st.button("Predict Smog Level"):
    input_features = np.array([[scaled_aqi, co, no, no2, o3, so2, pm25, pm10, nh3]])
    input_scaled = scaler.transform(input_features)
    input_cnn = input_scaled[..., np.newaxis]
    input_lstm = input_scaled[..., np.newaxis]
    
    # Predictions
    cnn_pred = np.argmax(cnn_model.predict(input_cnn), axis=1)[0] + 2  # Adjusting label indexing
    dnn_pred = np.argmax(dnn_model.predict(input_scaled), axis=1)[0] + 2
    lstm_pred = np.argmax(lstm_model.predict(input_lstm), axis=1)[0] + 2
    
    st.subheader("Predicted Smog Levels:")
    st.write(f"**CNN Model:** {smog_levels[cnn_pred]}")
    st.write(f"**DNN Model:** {smog_levels[dnn_pred]}")
    st.write(f"**LSTM Model:** {smog_levels[lstm_pred]}")

st.write("This app predicts PM10 & PM2.5 smog levels using deep learning models trained on air quality data.")
