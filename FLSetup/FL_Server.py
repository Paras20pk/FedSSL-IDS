import flwr as fl
from flwr.server import strategy
from flwr.server.history import History
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from FL_Utils import save_global_model
import argparse

def get_fedavg_strategy():
    """Configure federated averaging strategy"""
    return fl.server.strategy.FedAvg(
        fraction_fit=0.5,    # Sample 50% of clients for training
        fraction_evaluate=0.5,  # Sample 50% for evaluation
        min_fit_clients=2,   # Minimum 2 clients for training
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

def weighted_average_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate metrics with weighting by number of examples"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples)
    }

def fit_config(server_round: int):
    """Return training configuration for each round"""
    return {
        "batch_size": 32,
        "local_epochs": 3 if server_round < 10 else 5,  # Increase epochs later
        "learning_rate": 0.001 * (0.9 ** server_round),  # Decay learning rate
        "current_round": server_round
    }

def evaluate_config(server_round: int):
    """Return evaluation configuration"""
    return {
        "batch_size": 32,
        "current_round": server_round
    }

def start_server(num_rounds: int = 10, server_address: str = "0.0.0.0:8080"):
    """Start Flower server for federated learning"""
    strategy = get_fedavg_strategy()
    
    # Add model saving callback
    def save_model_after_aggregation(server_round, results, failures):
        if server_round % 2 == 0:  # Save every 2 rounds
            save_global_model(strategy.global_model, server_round)
        return None
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024,  # 1GB max message size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", 
                       help="Server address (IP:PORT)")
    args = parser.parse_args()
    
    print(f"🚀 Starting FL server on {args.address} for {args.rounds} rounds")
    start_server(num_rounds=args.rounds, server_address=args.address)