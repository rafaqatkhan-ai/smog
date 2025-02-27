import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests

# Streamlit UI
st.title("PM10 & PM2.5 Smog Level Prediction and Model Training")

# GitHub repository dataset URL
github_repo_url = "https://raw.githubusercontent.com/rafaqatkhan-ai/smog/main/"
dataset_files = ["smog.csv", "smog.xlsx"]  # List of available datasets

# Dropdown to select dataset from GitHub
selected_file = st.selectbox("Select a dataset from GitHub:", dataset_files)

def load_data_from_github(file_name):
    url = github_repo_url + file_name
    try:
        return pd.read_csv(url) if file_name.endswith(".csv") else pd.read_excel(url)
    except Exception as e:
        st.error(f"Failed to load dataset from GitHub repository. Error: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
else:
    data = load_data_from_github(selected_file)

if data is not None:
    st.write("### Dataset Preview:")
    st.write(data.head())
    
    # Data preprocessing
    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values.astype(int)  # Labels
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    
    # Reshape for CNN
    X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)
    
    # Model definitions
    def build_cnn(input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Train CNN Model
    num_classes = len(np.unique(Y))
    input_shape = (X_train_scaled.shape[1], 1)
    
    # Adjust labels if needed
    Y_train = Y_train - min(Y_train)  # Ensures labels start from 0
    
    cnn_model = build_cnn(input_shape, num_classes)
    cnn_model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32, verbose=1)
    cnn_model.save("cnn_model.h5")
    
    st.success("Model trained and saved successfully!")
    
    # Reload model for prediction
    if os.path.exists("cnn_model.h5"):
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        scaler = joblib.load("scaler.pkl")
    
        # Prediction UI
        smog_levels = {2: "Moderate", 3: "Unhealthy Sensitive", 4: "Unhealthy", 5: "Very Unhealthy", 6: "Hazardous"}
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
            input_cnn = np.expand_dims(input_scaled, axis=-1)
            
            cnn_pred = np.argmax(cnn_model.predict(input_cnn), axis=1)[0] + 2
            
            st.subheader("Predicted Smog Level:")
            st.write(f"**CNN Model:** {smog_levels[cnn_pred]}")

st.write("This app trains a CNN model and predicts PM10 & PM2.5 smog levels using air quality data.")
