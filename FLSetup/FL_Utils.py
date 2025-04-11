import pickle
import numpy as np
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras
import os

def load_model(model_path: str):
    """Load model from file (supports .pkl, .h5, .keras)"""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_path.endswith(('.h5', '.keras')):
        return keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

def preprocess_data(X, y=None):
    """Basic preprocessing - replace with your actual preprocessing"""
    X = np.array(X, dtype=np.float32)
    if y is not None:
        y = np.array(y, dtype=np.float32)
    return X, y

def get_client_data(client_id: int):
    """Mock data loading - REPLACE WITH YOUR ACTUAL DATA LOADING"""
    # This is just example data - replace with your actual dataset loading
    num_samples = 1000
    num_features = 50
    
    X_train = np.random.rand(num_samples, num_features)
    y_train = np.random.randint(0, 2, size=num_samples)
    X_val = np.random.rand(num_samples//5, num_features)
    y_val = np.random.randint(0, 2, size=num_samples//5)
    
    return preprocess_data(X_train, y_train), preprocess_data(X_val, y_val)

def save_global_model(model, round_num):
    """Save global model after aggregation"""
    os.makedirs('global_models', exist_ok=True)
    if isinstance(model, keras.Model):
        model.save(f'global_models/global_round_{round_num}.keras')
    else:
        with open(f'global_models/global_round_{round_num}.pkl', 'wb') as f:
            pickle.dump(model, f)