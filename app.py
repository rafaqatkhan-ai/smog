import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests
from io import BytesIO

# Streamlit UI
st.title("PM10 & PM2.5 Smog Level Prediction and Model Training")

# GitHub repository details
GITHUB_REPO = "https://api.github.com/repos/rafaqatkhan-ai/smog/contents/"

def list_files_in_repo(repo_url):
    """Fetch the list of files in the GitHub repository."""
    try:
        response = requests.get(repo_url)
        response.raise_for_status()
        files = [file['name'] for file in response.json() if file['type'] == 'file']
        return files
    except Exception as e:
        st.error(f"Error fetching files from GitHub: {e}")
        return []

def fetch_file_from_github(repo_url, filename):
    """Fetch the selected file from the GitHub repository."""
    try:
        raw_url = f"https://raw.githubusercontent.com/rafaqatkhan-ai/smog/main/{filename}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error fetching file from GitHub: {e}")
        return None

# Fetch the list of files in the repository
files = list_files_in_repo(GITHUB_REPO)

if files:
    # Create a dropdown to select a file
    selected_file = st.selectbox("Select a file from the GitHub repository:", files)
    
    if selected_file:
        # Fetch the selected file
        file_content = fetch_file_from_github(GITHUB_REPO, selected_file)
        if file_content:
            # Load the file into a pandas DataFrame
            if selected_file.endswith(".csv"):
                data = pd.read_csv(file_content)
            elif selected_file.endswith(".xlsx"):
                data = pd.read_excel(file_content)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                data = None
else:
    st.error("No files found in the GitHub repository.")

if data is not None:
    st.write("### Dataset Preview:")
    st.write(data.head())
    
    # Data preprocessing
    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values.astype(int)  # Labels
    
    # Shift labels to start from 0
    Y = Y - 2
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    
    # Reshape for CNN
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Debugging: Print shapes
    st.write(f"X_train_scaled shape: {X_train_scaled.shape}")
    st.write(f"Y_train shape: {Y_train.shape}")
    st.write(f"Unique labels in Y_train: {np.unique(Y_train)}")
    
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
    num_classes = len(np.unique(Y_train))
    input_shape = (X_train_scaled.shape[1], 1)
    
    cnn_model = build_cnn(input_shape, num_classes)
    history = cnn_model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32, verbose=1)
    cnn_model.save("cnn_model.h5")
    
    st.success("Model trained and saved successfully!")
    
    # Reload model for prediction
    if os.path.exists("cnn_model.h5"):
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
        scaler = joblib.load("scaler.pkl")
    
        # Prediction UI
        smog_levels = {0: "Moderate", 1: "Unhealthy Sensitive", 2: "Unhealthy", 3: "Very Unhealthy", 4: "Hazardous"}
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
            input_cnn = input_scaled.reshape(input_scaled.shape[0], input_scaled.shape[1], 1)
            
            cnn_pred = np.argmax(cnn_model.predict(input_cnn), axis=1)
            predicted_label = cnn_pred[0] + 2  # Shift back to original label range
            
            st.subheader("Predicted Smog Level:")
            st.write(f"**CNN Model:** {smog_levels[cnn_pred[0]]}")
    
st.write("This app trains a CNN model and predicts PM10 & PM2.5 smog levels using air quality data.")
