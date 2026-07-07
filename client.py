import os
import argparse
import numpy as np
import tensorflow as tf
import flwr as fl

import model as net
import privacy
import attack
import utils

# Set TensorFlow logging to suppress info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class IoTFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, client_name, x_train, y_train, x_val, y_val, 
                 input_dim, epochs=3, batch_size=32, privacy_noise=0.0, 
                 is_malicious=False, attack_type=None):
        self.client_id = client_id
        self.client_name = client_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.privacy_noise = privacy_noise
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        
        # Instantiate local model
        self.model = net.create_model(input_dim)
        
        # Log device type
        print(f"Initialized Client {client_id} ({client_name}): "
              f"Train samples = {len(x_train)}, Val samples = {len(x_val)}, "
              f"Malicious = {is_malicious} (Attack: {attack_type if is_malicious else 'None'})")

    def get_parameters(self, config):
        """Returns the local model parameters (weights)."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Trains the local model on the client data."""
        # Set weights received from the server
        self.model.set_weights(parameters)
        
        # Prepare training data (apply data poisoning attack if malicious label_flip is active)
        x_train_fit = self.x_train
        y_train_fit = self.y_train
        
        if self.is_malicious and self.attack_type == "label_flip":
            print(f"[{self.client_name}] Malicious client performing LABEL FLIPPING attack on local training data!")
            y_train_fit = attack.poison_labels(self.y_train)
            
        # Fit local model
        history = self.model.fit(
            x_train_fit, y_train_fit,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_data=(self.x_val, self.y_val)
        )
        
        # Extract training metrics
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        print(f"[{self.client_name}] Local training completed: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        # Retrieve trained weights
        weights = self.model.get_weights()
        
        # Apply model-level poisoning attack if malicious and not a label flip attack
        if self.is_malicious and self.attack_type in ["sign_flip", "random_noise", "zero_out", "scale_distortion"]:
            print(f"[{self.client_name}] Malicious client performing {self.attack_type.upper()} weight poisoning attack!")
            weights = attack.poison_weights(weights, self.attack_type)
            
        # Apply differential privacy (Gaussian noise) if configured
        if self.privacy_noise > 0.0:
            print(f"[{self.client_name}] Applying Differential Privacy noise (sd={self.privacy_noise})...")
            weights = privacy.add_dp_noise(weights, self.privacy_noise)
            
        return weights, len(self.x_train), {"loss": float(train_loss), "accuracy": float(train_acc)}

    def evaluate(self, parameters, config):
        """Evaluates the model on local validation data."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        print(f"[{self.client_name}] Evaluation complete: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        return float(loss), len(self.x_val), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True, choices=[1, 2, 3], help="ID of the client (1, 2, or 3)")
    parser.add_argument("--client-name", type=str, default="IoT Device", help="Readable name of the device")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Address of the server")
    parser.add_argument("--epochs", type=int, default=3, help="Number of local epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Local training batch size")
    parser.add_argument("--non-iid", action="store_true", help="Use non-I.I.D. data partitioning")
    parser.add_argument("--privacy-noise", type=float, default=0.0, help="DP Gaussian noise multiplier (0 for no noise)")
    parser.add_argument("--is-malicious", action="store_true", help="Simulate this client as malicious")
    parser.add_argument("--attack-type", type=str, default="sign_flip", choices=["label_flip", "sign_flip", "random_noise", "zero_out", "scale_distortion"], help="Type of attack to perform")
    parser.add_argument("--full-dataset", action="store_true", help="Use full UNSW_NB15 dataset instead of 20%")
    args = parser.parse_args()
    
    # Load and preprocess data
    x_train_full, y_train_full, x_test_full, y_test_full = utils.load_and_preprocess_data(args.full_dataset)
    
    # Partition data
    partitions = utils.partition_data(x_train_full, y_train_full, num_clients=3, non_iid=args.non_iid)
    
    # Select this client's partition
    x_c, y_c = partitions[args.client_id - 1]
    
    # Perform a 80-20 train-validation split on client's local dataset
    val_split = int(0.8 * len(x_c))
    x_train, x_val = x_c[:val_split], x_c[val_split:]
    y_train, y_val = y_c[:val_split], y_c[val_split:]
    
    input_dim = x_train.shape[1]
    
    # Create client
    client = IoTFlowerClient(
        client_id=args.client_id,
        client_name=args.client_name,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_dim=input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        privacy_noise=args.privacy_noise,
        is_malicious=args.is_malicious,
        attack_type=args.attack_type
    )
    
    # Start numpy client
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )
