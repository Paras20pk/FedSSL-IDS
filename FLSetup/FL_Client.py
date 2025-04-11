import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from FL_Utils import load_model, preprocess_data, get_client_data
import argparse
import os

class IDSClient(fl.client.NumPyClient):
    def __init__(self, model_path: str, client_id: int):
        """
        Initialize client with model and data
        :param model_path: Path to initial model file
        :param client_id: Unique ID for this client
        """
        self.client_id = client_id
        self.model = load_model(model_path)
        
        # Load client-specific data
        (self.x_train, self.y_train), (self.x_val, self.y_val) = get_client_data(client_id)
        
        print(f"Client {client_id} initialized with {len(self.x_train)} training samples")

    def get_parameters(self, config: Dict):
        """Return current model parameters"""
        if isinstance(self.model, keras.Model):
            return self.model.get_weights()
        else:
            # For scikit-learn models, return as numpy array
            if hasattr(self.model, 'coef_'):
                return [self.model.coef_]
            return []

    def set_parameters(self, parameters):
        """Update model with new parameters from server"""
        if isinstance(self.model, keras.Model):
            self.model.set_weights(parameters)
        else:
            # For scikit-learn models
            if hasattr(self.model, 'coef_') and parameters:
                self.model.coef_ = parameters[0]
        return self.model

    def fit(self, parameters, config: Dict):
        """Train model on local data"""
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Get training config
        batch_size = config.get("batch_size", 32)
        epochs = config.get("local_epochs", 1)
        lr = config.get("learning_rate", 0.001)
        
        # Train model
        if isinstance(self.model, keras.Model):
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            history = self.model.fit(
                self.x_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.x_val, self.y_val),
                verbose=0
            )
            print(f"Client {self.client_id} training complete - loss: {history.history['loss'][-1]:.4f}")
        else:
            # For scikit-learn models
            self.model.fit(self.x_train, self.y_train)
        
        # Return updated parameters and sample count
        return self.get_parameters({}), len(self.x_train), {}

    def evaluate(self, parameters, config: Dict):
        """Evaluate model on local validation data"""
        self.set_parameters(parameters)
        
        if isinstance(self.model, keras.Model):
            loss, accuracy = self.model.evaluate(
                self.x_val, self.y_val,
                batch_size=config.get("batch_size", 32),
                verbose=0
            )
        else:
            # For scikit-learn models
            accuracy = self.model.score(self.x_val, self.y_val)
            loss = 1 - accuracy  # Simple loss approximation
        
        return float(loss), len(self.x_val), {"accuracy": float(accuracy)}

def start_client(client_id: int, server_address: str, model_path: str):
    """Start Flower client"""
    client = IDSClient(model_path, client_id)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024*1024*1024,  # 1GB max message size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="localhost:8080",
                       help="Server address (IP:PORT)")
    parser.add_argument("--model", type=str, default="models/global_model.pkl",
                       help="Path to initial model")
    args = parser.parse_args()
    
    print(f"🔌 Starting client {args.cid} connecting to {args.server}")
    start_client(args.cid, args.server, args.model)