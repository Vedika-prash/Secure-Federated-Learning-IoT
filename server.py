import os
import argparse
import json
import flwr as fl
import tensorflow as tf

import model as net
import utils

# Set TensorFlow logging to suppress info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def aggregate_fit_metrics(results):
    """Aggregates client training metrics by computing weighted average accuracy and loss."""
    if not results:
        return {}
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in results]) / total_examples
    weighted_loss = sum([num_examples * m["loss"] for num_examples, m in results]) / total_examples
    return {"accuracy": weighted_acc, "loss": weighted_loss}

def aggregate_evaluate_metrics(results):
    """Aggregates client evaluation metrics by computing weighted average accuracy."""
    if not results:
        return {}
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in results]) / total_examples
    return {"accuracy": weighted_acc}

def get_evaluate_fn(input_dim, x_test, y_test, results_file):
    """Returns a function that evaluates the global model on the server's test set."""
    # Create an instance of the model for evaluation
    model = net.create_model(input_dim)
    
    # Store training metrics over rounds
    history = {"rounds": [], "accuracy": [], "loss": []}
    
    # Remove old results file if it exists to start fresh
    if os.path.exists(results_file):
        try:
            os.remove(results_file)
        except OSError:
            pass
            
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        print(f"\n==================================================")
        print(f"ROUND {server_round}: Global Model Evaluation")
        print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
        print(f"==================================================\n")
        
        # Update and save training metrics
        history["rounds"].append(server_round)
        history["loss"].append(float(loss))
        history["accuracy"].append(float(accuracy))
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(history, f, indent=4)
            
        return float(loss), {"accuracy": float(accuracy)}
        
    return evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Federated Learning Server")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Address of the server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum number of clients")
    parser.add_argument("--results-file", type=str, default="results/history_clean.json", help="Path to save results history")
    parser.add_argument("--full-dataset", action="store_true", help="Use full UNSW_NB15 dataset instead of 20%")
    args = parser.parse_args()
    
    # Load and preprocess data
    _, _, x_test, y_test = utils.load_and_preprocess_data(args.full_dataset)
    input_dim = x_test.shape[1]
    
    # Define federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Train on all available clients (3/3)
        fraction_evaluate=1.0,      # Evaluate on all available clients (3/3)
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_fn=get_evaluate_fn(input_dim, x_test, y_test, args.results_file),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    
    # Start Flower server
    print(f"Starting Flower Federated Learning server on {args.server_address}...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )
