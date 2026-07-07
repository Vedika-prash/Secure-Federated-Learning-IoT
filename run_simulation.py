import os
import sys
import time
import argparse
import subprocess

def run_fl_simulation(rounds=5, epochs=3, batch_size=32, non_iid=False, 
                      privacy_noise=0.0, malicious_client_id=0, attack_type="sign_flip",
                      use_full_dataset=False):
    """
    Spawns a Flower Server and 3 IoT clients as subprocesses.
    Redirects stdout/stderr of each process to files in the results/ directory.
    """
    os.makedirs("results", exist_ok=True)
    
    # Determine result output JSON filename
    is_attack = malicious_client_id in [1, 2, 3]
    results_file = "results/history_attack.json" if is_attack else "results/history_clean.json"
    
    # Clean previous result files for these runs
    if os.path.exists(results_file):
        try:
            os.remove(results_file)
        except OSError:
            pass
            
    print("==================================================")
    print("Starting Federated Learning Simulation")
    print(f"Rounds: {rounds} | Local Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Non-I.I.D.: {non_iid} | DP Noise (SD): {privacy_noise}")
    if is_attack:
        print(f"Malicious Client: Client {malicious_client_id} performing {attack_type.upper()}")
    else:
        print("Malicious Client: None (Clean Run)")
    print("==================================================\n")
    
    # Helper to check python command name
    python_cmd = sys.executable
    
    # 1. Start Server
    server_log = open("results/server.log", "w")
    server_args = [
        python_cmd, "server.py",
        "--rounds", str(rounds),
        "--results-file", results_file
    ]
    if use_full_dataset:
        server_args.append("--full-dataset")
        
    print("Launching Federated Learning Server...")
    server_process = subprocess.Popen(server_args, stdout=server_log, stderr=server_log)
    
    # Give the server a few seconds to initialize and open port 8080
    time.sleep(4)
    
    # 2. Launch Clients
    client_processes = []
    client_logs = []
    
    clients = [
        (1, "Smart Camera"),
        (2, "Smart Bulb"),
        (3, "Smart Thermostat")
    ]
    
    for cid, name in clients:
        log_file = open(f"results/client_{cid}.log", "w")
        client_logs.append(log_file)
        
        client_args = [
            python_cmd, "client.py",
            "--client-id", str(cid),
            "--client-name", name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--privacy-noise", str(privacy_noise)
        ]
        
        if non_iid:
            client_args.append("--non-iid")
        if use_full_dataset:
            client_args.append("--full-dataset")
            
        # Configure malicious parameters
        if cid == malicious_client_id:
            client_args.append("--is-malicious")
            client_args.append("--attack-type")
            client_args.append(attack_type)
            
        print(f"Launching Client {cid} ({name})...")
        p = subprocess.Popen(client_args, stdout=log_file, stderr=log_file)
        client_processes.append(p)
        
    print("\nSimulation is running. Monitoring progress...")
    
    # 3. Wait for all processes to complete
    try:
        # Wait for client processes first
        for idx, p in enumerate(client_processes):
            cid = idx + 1
            name = clients[idx][1]
            p.wait()
            print(f"Client {cid} ({name}) finished.")
            
        # Give server process some time to shut down gracefully after clients disconnect
        server_process.wait(timeout=10)
        print("Server finished.")
        
    except KeyboardInterrupt:
        print("\nTermination requested. Killing all processes...")
        server_process.terminate()
        for p in client_processes:
            p.terminate()
    except subprocess.TimeoutExpired:
        print("\nServer shutdown timed out. Force closing server...")
        server_process.kill()
    finally:
        # Close log files
        server_log.close()
        for log in client_logs:
            log.close()
            
    print("\nSimulation completed successfully.")
    print("Log files generated:")
    print(" - Server log: results/server.log")
    print(" - Client logs: results/client_1.log, results/client_2.log, results/client_3.log")
    print(f" - Global metrics file: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Federated Learning simulation")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=3, help="Number of local epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Local training batch size")
    parser.add_argument("--non-iid", action="store_true", help="Use non-I.I.D. data partitioning")
    parser.add_argument("--privacy-noise", type=float, default=0.0, help="DP Gaussian noise level")
    parser.add_argument("--malicious-client", type=int, default=0, choices=[0, 1, 2, 3], help="ID of malicious client (0 for none)")
    parser.add_argument("--attack-type", type=str, default="sign_flip", help="Type of attack if malicious client is active")
    parser.add_argument("--full-dataset", action="store_true", help="Use full UNSW_NB15 dataset instead of 20%")
    args = parser.parse_args()
    
    run_fl_simulation(
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        non_iid=args.non_iid,
        privacy_noise=args.privacy_noise,
        malicious_client_id=args.malicious_client,
        attack_type=args.attack_type,
        use_full_dataset=args.full_dataset
    )
