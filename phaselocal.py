"""
FedSSL-IDS: Local Simulation Version (No Physical Edge Devices Needed)
Components:
1. Federated Learning Simulation with Flower
2. DDoS Detection using Contrastive Learning
3. Real-time attack classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import Metrics
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------
# 1. Data Preprocessing (DDoS Attack Data)
# --------------------------
def preprocess_ddos(data_path: str = r"C:\Users\Paras\Desktop\IDSFedSSL\Small Training Set.csv", client_id: Optional[int] = None):
    """Load and preprocess DDoS attack data with optional client partitioning"""
    try:
        df = pd.read_csv(data_path, header=None)  # Assuming no header
        
        # Assign column names based on your dataset structure
        # (Modify this according to your actual dataset columns)
        feature_columns = list(range(0, 41))  # First 41 columns as features
        label_column = 41  # Last column as label
        
        # Feature engineering for DDoS detection
        features = df.iloc[:, feature_columns]
        
        # Label encoding (1 for attack, 0 for normal)
        # Modify this based on your actual label encoding
        labels = df.iloc[:, label_column].apply(lambda x: 1 if x != 'normal' else 0)
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Normalize
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Split into multiple clients if specified
        if client_id is not None:
            # Create non-IID distribution for simulation
            np.random.seed(42)
            client_samples = len(features) // 5  # Split among 5 clients
            start_idx = client_id * client_samples
            end_idx = (client_id + 1) * client_samples if client_id < 4 else len(features)
            features = features[start_idx:end_idx]
            labels = labels[start_idx:end_idx]
        
        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        train_data = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train.values)
        )
        test_data = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test.values)
        )
        
        return DataLoader(train_data, batch_size=32, shuffle=True), \
               DataLoader(test_data, batch_size=32)
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to synthetic data
        print("Using synthetic data for simulation")
        X = np.random.rand(100, 41)  # Match your feature dimension
        y = np.random.randint(0, 2, 100)
        train_data = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return DataLoader(train_data, batch_size=32), DataLoader(train_data, batch_size=32)

# --------------------------
# 2. Enhanced DDoS Detection Model
# --------------------------
class DDoSDetector(nn.Module):
    """Combines SSL pretraining and supervised classification"""
    def __init__(self, input_dim=41, latent_dim=128):  # Updated input_dim to match your features
        super().__init__()
        # SSL Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Attack Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

def contrastive_loss(z1, z2, temperature=0.1):
    """SSL Loss Function"""
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    labels = torch.arange(z.shape[0]).to(z.device)
    return nn.CrossEntropyLoss()(sim, labels)

# --------------------------
# 3. Federated Learning Client Simulation
# --------------------------
class DDoSClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.trainloader, self.testloader = preprocess_ddos(client_id=client_id)
        self.model = DDoSDetector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ssl_pretrain()  # Phase 1: SSL Pretraining
        
    def ssl_pretrain(self):
        """Self-supervised pretraining on unlabeled data"""
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        
        for _ in range(3):  # Short pretraining
            for inputs, _ in self.trainloader:
                inputs = inputs.to(self.device)
                # SimCLR-style augmentation
                inputs_aug = inputs + torch.randn_like(inputs) * 0.1
                z1 = self.model.encoder(inputs)
                z2 = self.model.encoder(inputs_aug)
                loss = contrastive_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Supervised training (Phase 2)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        
        for epoch in range(5):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        correct, total, tp, fp, loss = 0, 0, 0, 0, 0.0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Attack-specific metrics
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
        
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        loss /= len(self.testloader)
        return float(loss), len(self.testloader), {
            "accuracy": accuracy,
            "precision": precision,
            "attack_detected": tp
        }

# --------------------------
# 4. Simulation Functions
# --------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics across clients"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    total = sum([num_examples for num_examples, _ in metrics])
    
    return {
        "accuracy": sum(accuracies) / total,
        "precision": sum(precisions) / total,
        "total_attacks": sum([m["attack_detected"] for _, m in metrics])
    }

def client_fn(client_id: int) -> DDoSClient:
    """Create a client instance for simulation"""
    return DDoSClient(client_id)

def run_simulation():
    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

if __name__ == "__main__":
    print("Starting Federated DDoS Detection Simulation...")
    run_simulation()