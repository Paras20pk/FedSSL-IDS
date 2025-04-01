"""
FedSSL-IDS: Integrated Phase 1 (FL Setup) + Phase 2 (DDoS Detection)
Components:
1. Federated Learning with Flower
2. DDoS Detection using Contrastive Learning
3. Real-time attack classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import Metrics
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import numpy as np

# --------------------------
# 1. Data Preprocessing (DDoS Attack Data)
# --------------------------
def preprocess_ddos(data_path=r"C:\Users\Paras\Desktop\IDSFedSSL\Small Training Set.csv"):
    """Load and preprocess DDoS attack data"""
    try:
        df = pd.read_csv(data_path)
        
        # Feature engineering for DDoS detection
        features = df[[
            'Flow Duration', 
            'Total Fwd Packets',
            'Total Bwd Packets',
            'Flow Bytes/s',
            'Packet Length Mean'
        ]]
        
        # Label encoding (1 for attack, 0 for normal)
        labels = df['Label'].apply(lambda x: 1 if 'DDoS' in str(x) else 0)
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Normalize
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Split data
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
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        train_data = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return DataLoader(train_data, batch_size=32), DataLoader(train_data, batch_size=32)

# --------------------------
# 2. Enhanced DDoS Detection Model
# --------------------------
class DDoSDetector(nn.Module):
    """Combines SSL pretraining and supervised classification"""
    def __init__(self, input_dim=5, latent_dim=128):
        super().__init__()
        # SSL Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Attack Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification
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
# 3. Federated Learning with DDoS Detection
# --------------------------
class DDoSClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.trainloader, self.testloader = preprocess_ddos(f"client_{client_id}_data.csv")
        self.model = DDoSDetector()
        self.ssl_pretrain()  # Phase 1: SSL Pretraining
        
    def ssl_pretrain(self):
        """Self-supervised pretraining on unlabeled data"""
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        for _ in range(3):  # Short pretraining
            for inputs, _ in self.trainloader:
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
    
    def fit(self, parameters, config):
        # Update model with global parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        
        # Supervised training (Phase 2)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):
            for inputs, labels in self.trainloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.fit(parameters, config)
        correct, total, tp, fp = 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Attack-specific metrics
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
        
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        return float(loss), len(self.testloader), {
            "accuracy": accuracy,
            "precision": precision,
            "attack_detected": tp
        }

# --------------------------
# 4. Enhanced FL Simulation
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

def start_simulation():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    # Start clients
    for client_id in range(5):
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=DDoSClient(client_id)
        )

if __name__ == "__main__":
    start_simulation()