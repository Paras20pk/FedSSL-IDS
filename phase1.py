"""
FedSSL-IDS Phase 1: Federated Learning Setup with SSL Pretraining
Updated for Flower 1.17.0 compatibility
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

# --------------------------
# 1. Data Preprocessing (DDoS Example)
# --------------------------
def preprocess_ddos(data_path="CIC-IDS2017.csv"):
    """Extract features from network traffic data"""
    df = pd.read_csv(data_path)
    
    # Select key features for DDoS detection
    features = df[[
        'Flow Duration', 
        'Total Fwd Packets',
        'Total Bwd Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets'
    ]]
    
    # Label: 1 for DDoS, 0 for normal (assuming column exists)
    labels = df['Label'].apply(lambda x: 1 if x == 'DDoS' else 0)
    
    # Normalize
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data (80% train, 20% test)
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

# --------------------------
# 2. SSL Model (Contrastive Learning)
# --------------------------
class SSLModel(nn.Module):
    """Self-Supervised Model using Contrastive Learning"""
    def __init__(self, input_dim=5, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

def contrastive_loss(z1, z2, temperature=0.1):
    """NT-Xent Loss for SSL"""
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    labels = torch.arange(z.shape[0]).to(z.device)
    return nn.CrossEntropyLoss()(sim, labels)

# --------------------------
# 3. Federated Learning Setup
# --------------------------
class EdgeClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.trainloader, self.testloader = preprocess_ddos(
            f"client_{client_id}_data.csv"  # Simulated per-client data
        )
        self.model = SSLModel()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # Update local model with global parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        
        # Local training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):  # Local epochs
            for inputs, labels in self.trainloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.fit(parameters, config)  # Update model first
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return float(loss), len(self.testloader), {"accuracy": accuracy}

# --------------------------
# 4. Start Simulation (Updated for Flower 1.17.0)
# --------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for federated evaluation"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def start_simulation():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=5,  # Minimum number of clients to be sampled for training
        min_evaluate_clients=5,  # Minimum number of clients to be sampled for evaluation
        min_available_clients=5,  # Minimum number of total clients in the system
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate accuracy
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    # Start Flower clients
    for client_id in range(5):
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=EdgeClient(client_id)
        )

if __name__ == "__main__":
    start_simulation()