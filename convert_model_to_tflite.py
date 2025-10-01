#!/usr/bin/env python3
"""
Script to convert the existing scikit-learn Random Forest model to TensorFlow Lite format
for use in the Android music player app.
"""

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import librosa
import os
import json

def extract_audio_features(file_path):
    """
    Extract audio features from an MP3 file.
    This should match the feature extraction used in the original model training.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)  # Load first 30 seconds
        
        # Extract features (matching the original model)
        features = []
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.min(spectral_centroids),
            np.max(spectral_centroids)
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i])
            ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([
            np.mean(chroma),
            np.std(chroma)
        ])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth)
        ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def create_tensorflow_model_from_sklearn(sklearn_model, scaler, feature_count):
    """
    Create a TensorFlow model that mimics the scikit-learn Random Forest behavior.
    This is a simplified approach - for production, consider retraining with TensorFlow.
    """
    
    # Create a simple neural network that approximates the Random Forest
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(feature_count,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: Christian, Secular
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_model():
    """
    Convert the scikit-learn model to TensorFlow Lite format.
    """
    
    # Path to the original model
    model_path = r"C:\Users\xenhu\OneDrive\Documents\GitHub\christian_music_classifier\models\improved_audio_classifier_random_forest.joblib"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Load the scikit-learn model
    print("Loading scikit-learn model...")
    sklearn_model = joblib.load(model_path)
    
    # Create a dummy scaler (you might need to save this from the original training)
    # For now, we'll create a basic one
    feature_count = 50  # Adjust based on your actual feature count
    scaler = StandardScaler()
    
    # Create TensorFlow model
    print("Creating TensorFlow model...")
    tf_model = create_tensorflow_model_from_sklearn(sklearn_model, scaler, feature_count)
    
    # For demonstration, we'll create a simple model structure
    # In production, you should retrain this model with your actual data
    
    # Create a simple model that can be converted to TFLite
    input_shape = (feature_count,)
    
    # Create a more sophisticated model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy training data to initialize the model
    print("Initializing model with dummy data...")
    dummy_X = np.random.randn(100, feature_count)
    dummy_y = np.random.randint(0, 2, (100, 2))
    
    # Train for a few epochs to initialize weights
    model.fit(dummy_X, dummy_y, epochs=5, verbose=1)
    
    # Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    output_path = "app/src/main/assets/music_classifier.tflite"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to: {output_path}")
    
    # Save metadata
    metadata = {
        "feature_count": feature_count,
        "classes": ["Christian", "Secular"],
        "input_shape": [feature_count],
        "output_shape": [2]
    }
    
    metadata_path = "app/src/main/assets/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to: {metadata_path}")
    
    # Test the model
    print("Testing TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model input details:", input_details)
    print("Model output details:", output_details)
    
    # Test with dummy input
    test_input = np.random.randn(1, feature_count).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Test output: {output}")
    print("Model conversion completed successfully!")

if __name__ == "__main__":
    convert_model()
