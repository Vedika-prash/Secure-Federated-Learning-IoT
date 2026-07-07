import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# UNSW_NB15 column names for reference
COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", 
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", 
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", 
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", 
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", 
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label"
]

DATASET_DIR = "dataset"
RESULTS_DIR = "results"

# Github URLs for raw UNSW_NB15 dataset (training & testing sets)
URL_TRAIN = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
URL_TEST = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

def ensure_datasets_exist(use_full_train=False):
    """Ensures dataset directory and files exist, downloads them if missing."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_path = os.path.join(DATASET_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATASET_DIR, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_path):
        download_file(URL_TRAIN, train_path)
    else:
        print(f"Train dataset found at {train_path}")
        
    if not os.path.exists(test_path):
        download_file(URL_TEST, test_path)
    else:
        print(f"Test dataset found at {test_path}")
        
    return train_path, test_path

def load_and_preprocess_data(use_full_train=False):
    """Loads, preprocesses, cleans, encodes, and normalizes UNSW_NB15 train/test data."""
    train_path, test_path = ensure_datasets_exist(use_full_train)
    
    # Load dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop unnecessary columns
    train_df = train_df.drop(columns=["id", "attack_cat"], errors="ignore")
    test_df = test_df.drop(columns=["id", "attack_cat"], errors="ignore")
    
    # Handle NaNs just in case
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Apply dynamic subsampling on the training set if not using full dataset (20%)
    if not use_full_train:
        train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        
    # Separate numeric and categorical
    cat_cols = ["proto", "service", "state"]
    num_cols = [c for c in train_df.columns if c not in cat_cols and c != "label"]
    
    # One-hot encode categorical features (combine first to ensure same column structure)
    train_len = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    
    # Convert all boolean columns to float32
    bool_cols = combined_encoded.select_dtypes(include=["bool"]).columns
    combined_encoded[bool_cols] = combined_encoded[bool_cols].astype("float32")
    
    # Split back
    train_encoded = combined_encoded.iloc[:train_len].copy()
    test_encoded = combined_encoded.iloc[train_len:].copy()
    
    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    train_encoded[num_cols] = scaler.fit_transform(train_encoded[num_cols])
    test_encoded[num_cols] = scaler.transform(test_encoded[num_cols])
    
    # Separate features and labels
    x_train = train_encoded.drop(columns=["label"]).values.astype("float32")
    y_train = train_encoded["label"].values.astype("float32")
    x_test = test_encoded.drop(columns=["label"]).values.astype("float32")
    y_test = test_encoded["label"].values.astype("float32")
    
    return x_train, y_train, x_test, y_test

def partition_data(x_train, y_train, num_clients=3, non_iid=False):
    """
    Splits the training data among multiple clients.
    If non_iid is True, partitions data such that each device has a distinct distribution of attack/normal traffic.
    """
    np.random.seed(42)
    
    if not non_iid:
        # I.I.D. Partition: Shuffle and split evenly
        indices = np.random.permutation(len(x_train))
        splits = np.array_split(indices, num_clients)
        
        partitions = []
        for split in splits:
            partitions.append((x_train[split], y_train[split]))
        return partitions
    else:
        # Non-I.I.D. Partition: Assign specific percentages of normal/attack traffic to devices
        # Client 1: Smart Camera -> Mostly Normal traffic (50% of all normal, 10% of all attack)
        # Client 2: Smart Bulb   -> Heavy Attack traffic  (15% of all normal, 60% of all attack)
        # Client 3: Smart Thermostat -> Balanced traffic  (35% of all normal, 30% of all attack)
        
        normal_idx = np.where(y_train == 0)[0]
        attack_idx = np.where(y_train == 1)[0]
        
        np.random.shuffle(normal_idx)
        np.random.shuffle(attack_idx)
        
        # Define normal split bounds
        n1 = int(0.50 * len(normal_idx))
        n2 = n1 + int(0.15 * len(normal_idx))
        
        # Define attack split bounds
        a1 = int(0.10 * len(attack_idx))
        a2 = a1 + int(0.60 * len(attack_idx))
        
        # Client index splits
        c1_idx = np.concatenate([normal_idx[:n1], attack_idx[:a1]])
        c2_idx = np.concatenate([normal_idx[n1:n2], attack_idx[a1:a2]])
        c3_idx = np.concatenate([normal_idx[n2:], attack_idx[a2:]])
        
        # Shuffle each client's indices
        np.random.shuffle(c1_idx)
        np.random.shuffle(c2_idx)
        np.random.shuffle(c3_idx)
        
        partitions = [
            (x_train[c1_idx], y_train[c1_idx]),
            (x_train[c2_idx], y_train[c2_idx]),
            (x_train[c3_idx], y_train[c3_idx])
        ]
        return partitions

def plot_results(history, save_name="fl_results.png"):
    """Plots training accuracy and loss over federated learning rounds."""
    rounds = history.get("rounds", [])
    accuracy = history.get("accuracy", [])
    loss = history.get("loss", [])
    
    if not rounds:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(rounds, accuracy, marker='o', color='#4CAF50', linewidth=2, label='Global Accuracy')
    ax1.set_title('Global Model Accuracy vs. Communication Rounds', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rounds', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Loss Plot
    ax2.plot(rounds, loss, marker='x', color='#F44336', linewidth=2, label='Global Loss')
    ax2.set_title('Global Model Loss vs. Communication Rounds', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rounds', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Results plot saved at {plot_path}")
