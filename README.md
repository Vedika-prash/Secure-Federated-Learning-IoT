# Secure-Federated-Learning-IoT
Privacy-preserving Federated Learning framework for IoT intrusion detection using Differential Privacy and poisoning attack simulation.
# Secure Federated Learning Framework for IoT Intrusion Detection

##  Overview
This project presents a privacy-preserving Federated Learning framework for IoT intrusion detection. Instead of sharing raw data with a central server, multiple clients collaboratively train a global intrusion detection model using the Federated Averaging (FedAvg) algorithm. The framework integrates Differential Privacy to protect sensitive information and simulates poisoning attacks to evaluate the robustness of the global model against malicious clients.

---

##  Features
- Federated Learning-based collaborative model training
- Differential Privacy for secure model updates
- Poisoning attack simulation to evaluate robustness
- FedAvg algorithm for global model aggregation
- Streamlit dashboard for real-time visualization
- Performance evaluation using accuracy and loss metrics

---

##  Technologies Used
- Python
- TensorFlow
- Flower (Federated Learning Framework)
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## Dataset
- **UNSW-NB15** Network Intrusion Detection Dataset

---

## Workflow

1. Load and preprocess the UNSW-NB15 dataset.
2. Build the neural network intrusion detection model.
3. Train local models on multiple federated clients.
4. Apply Differential Privacy to model updates.
5. Simulate poisoning attacks from malicious clients.
6. Aggregate client models using the FedAvg algorithm.
7. Visualize model performance and security metrics using the Streamlit dashboard.

---

## Key Features
- Privacy-preserving distributed learning
- Robustness evaluation against malicious clients
- Federated model aggregation
- Interactive dashboard for monitoring training progress
- Secure IoT intrusion detection

---

## My Contribution
This was a **team project consisting of four members**. My primary contributions included:
- Implementing the **server-side FedAvg aggregation** module.
- Developing the **poisoning attack simulation** module.
- Assisting in system integration, testing, and performance evaluation.

---

##  Future Enhancements
- Support for additional federated aggregation algorithms.
- Integration with real-world IoT devices.
- Advanced Byzantine-resilient aggregation techniques.
- Deployment on cloud-based federated environments.

---

##  License
This project was developed for academic and educational purposes.
