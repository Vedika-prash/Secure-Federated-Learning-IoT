import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="IoT Federated Learning Security Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background-color: #0F172A;
    }
    .sidebar .sidebar-content {
        background-color: #1E293B;
    }
    h1, h2, h3 {
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
    }
    .card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .viva-title {
        color: #F43F5E;
        font-weight: bold;
    }
    .concept-card {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .success-alert {
        background-color: #064E3B;
        color: #34D399;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #059669;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate SVG Network Diagram
def render_network_svg(malicious_client_id=0, attack_type=""):
    # Node colors
    server_color = "#818CF8"  # Indigo
    c1_color = "#34D399" if malicious_client_id != 1 else "#F87171"
    c2_color = "#34D399" if malicious_client_id != 2 else "#F87171"
    c3_color = "#34D399" if malicious_client_id != 3 else "#F87171"
    
    # Text tags for status
    c1_status = "Smart Camera (Normal)" if malicious_client_id != 1 else f"Smart Camera (ATTACK: {attack_type.upper()})"
    c2_status = "Smart Bulb (Normal)" if malicious_client_id != 2 else f"Smart Bulb (ATTACK: {attack_type.upper()})"
    c3_status = "Smart Thermostat (Normal)" if malicious_client_id != 3 else f"Smart Thermostat (ATTACK: {attack_type.upper()})"
    
    # Link animations (green/blue for normal, red for malicious)
    flow_1 = "flow-malicious" if malicious_client_id == 1 else "flow-normal"
    flow_2 = "flow-malicious" if malicious_client_id == 2 else "flow-normal"
    flow_3 = "flow-malicious" if malicious_client_id == 3 else "flow-normal"
    
    svg = f"""
    <svg width="100%" height="280" viewBox="0 0 800 280" style="background:#1E293B; border-radius:12px; border:1px solid #334155;">
        <defs>
            <style>
                .flow-normal {{
                    stroke: #34D399;
                    stroke-width: 2.5;
                    stroke-dasharray: 8, 8;
                    animation: dash 20s linear infinite;
                }}
                .flow-malicious {{
                    stroke: #F87171;
                    stroke-width: 3.5;
                    stroke-dasharray: 8, 8;
                    animation: dash 10s linear infinite;
                }}
                .node {{
                    transition: all 0.3s ease;
                }}
                .node:hover {{
                    filter: brightness(1.2);
                    cursor: pointer;
                }}
                @keyframes dash {{
                    to {{
                        stroke-dashoffset: -1000;
                    }}
                }}
                .pulse {{
                    animation: pulse-glow 2s infinite alternate;
                }}
                @keyframes pulse-glow {{
                    0% {{ filter: drop-shadow(0 0 2px rgba(99, 102, 241, 0.5)); }}
                    100% {{ filter: drop-shadow(0 0 15px rgba(99, 102, 241, 0.9)); }}
                }}
            </style>
            <linearGradient id="normalGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#059669" />
                <stop offset="100%" stop-color="#34D399" />
            </linearGradient>
            <linearGradient id="maliciousGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#DC2626" />
                <stop offset="100%" stop-color="#F87171" />
            </linearGradient>
            <linearGradient id="serverGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#4F46E5" />
                <stop offset="100%" stop-color="#818CF8" />
            </linearGradient>
        </defs>
        
        <!-- Connection Lines -->
        <path d="M 200 200 L 400 65" class="{flow_1}" />
        <path d="M 400 200 L 400 65" class="{flow_2}" />
        <path d="M 600 200 L 400 65" class="{flow_3}" />
        
        <!-- Central Server Node -->
        <g class="node pulse" transform="translate(400, 65)">
            <circle r="30" fill="url(#serverGrad)" />
            <text y="-40" text-anchor="middle" fill="#E2E8F0" font-family="sans-serif" font-weight="bold" font-size="14">Aggregation Server</text>
            <text y="5" text-anchor="middle" fill="#FFFFFF" font-family="sans-serif" font-weight="bold" font-size="12">FedAvg</text>
        </g>
        
        <!-- Client 1 (Smart Camera) -->
        <g class="node" transform="translate(200, 200)">
            <circle r="25" fill="{ 'url(#maliciousGrad)' if malicious_client_id == 1 else 'url(#normalGrad)' }" />
            <text y="45" text-anchor="middle" fill="#94A3B8" font-family="sans-serif" font-size="12" font-weight="bold">{c1_status}</text>
            <text y="5" text-anchor="middle" fill="#FFFFFF" font-family="sans-serif" font-size="10">Camera</text>
        </g>
        
        <!-- Client 2 (Smart Bulb) -->
        <g class="node" transform="translate(400, 200)">
            <circle r="25" fill="{ 'url(#maliciousGrad)' if malicious_client_id == 2 else 'url(#normalGrad)' }" />
            <text y="45" text-anchor="middle" fill="#94A3B8" font-family="sans-serif" font-size="12" font-weight="bold">{c2_status}</text>
            <text y="5" text-anchor="middle" fill="#FFFFFF" font-family="sans-serif" font-size="10">Bulb</text>
        </g>
        
        <!-- Client 3 (Smart Thermostat) -->
        <g class="node" transform="translate(600, 200)">
            <circle r="25" fill="{ 'url(#maliciousGrad)' if malicious_client_id == 3 else 'url(#normalGrad)' }" />
            <text y="45" text-anchor="middle" fill="#94A3B8" font-family="sans-serif" font-size="12" font-weight="bold">{c3_status}</text>
            <text y="5" text-anchor="middle" fill="#FFFFFF" font-family="sans-serif" font-size="10">Thermostat</text>
        </g>
    </svg>
    """
    st.components.v1.html(svg, height=295)

# Page Title Layout
st.markdown("<h1>🛡️ Secure Federated Learning for IoT Security</h1>", unsafe_allow_html=True)
st.markdown("### Privacy-Preserving Collaborative Cyber Intrusion Detection in Smart Homes", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("## ⚙️ Simulation Settings")

dataset_type = st.sidebar.selectbox("Dataset Size", ["Subsampled (20%)", "Full (100%)"])
use_full_dataset = dataset_type == "Full (100%)"

rounds = st.sidebar.slider("Federated Communication Rounds", min_value=1, max_value=15, value=5)
epochs = st.sidebar.slider("Local Epochs (per round)", min_value=1, max_value=5, value=3)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
non_iid = st.sidebar.checkbox("Non-I.I.D. Data Partitioning", value=True, 
                             help="Distributes attack and normal traffic unevenly among devices.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔒 Security & Privacy Layer")

privacy_noise = st.sidebar.slider(
    "Differential Privacy Noise (Std Dev)", 
    min_value=0.0, 
    max_value=0.1, 
    value=0.0, 
    step=0.01,
    help="Adds Gaussian noise to model weights to prevent leakage of private training data."
)

malicious_client = st.sidebar.selectbox(
    "Malicious Client (Attack Simulation)",
    ["None", "Client 1 (Smart Camera)", "Client 2 (Smart Bulb)", "Client 3 (Smart Thermostat)"]
)
malicious_client_id = 0
if malicious_client != "None":
    malicious_client_id = int(malicious_client.split()[1])

attack_type = st.sidebar.selectbox(
    "Attack Vector (Weight/Label Poisoning)",
    ["Sign Flip", "Label Flip", "Zero Out", "Random Noise"],
    disabled=(malicious_client_id == 0)
)

st.sidebar.markdown("---")
run_sim = st.sidebar.button("🚀 Run FL Simulation", use_container_width=True)

# ----------------- MAIN LAYOUT -----------------
col_main, col_edu = st.columns([7, 3])

with col_main:
    # Render SVG Flow Diagram
    st.markdown("### 🕸️ Live Network Node Visualization")
    render_network_svg(malicious_client_id, attack_type if malicious_client_id != 0 else "")
    
    # Simulation Run Execution block
    if run_sim:
        st.markdown("### 🖥️ Simulation execution console")
        log_placeholder = st.empty()
        
        # Build subprocess arguments
        cmd = [
            sys.executable, "run_simulation.py",
            "--rounds", str(rounds),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--privacy-noise", str(privacy_noise),
            "--malicious-client", str(malicious_client_id)
        ]
        
        if malicious_client_id > 0:
            # Map clean name back to code command string
            cmd.extend(["--attack-type", attack_type.lower().replace(" ", "_")])
            
        if non_iid:
            cmd.append("--non-iid")
        if use_full_dataset:
            cmd.append("--full-dataset")
            
        # Spawn execution process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        log_text = ""
        while True:
            line = process.stdout.readline()
            if not line:
                break
            log_text += line
            # Keep log readable, trim older lines if log gets extremely long
            log_lines = log_text.split('\n')
            if len(log_lines) > 25:
                log_display = "...\n" + "\n".join(log_lines[-25:])
            else:
                log_display = log_text
            log_placeholder.code(log_display)
            
        process.wait()
        
        st.markdown('<div class="success-alert">✅ Federated Learning Simulation Finished successfully! Loading results...</div>', unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()

    # Load Simulation Data
    clean_history_path = "results/history_clean.json"
    attack_history_path = "results/history_attack.json"
    
    clean_history = None
    attack_history = None
    
    if os.path.exists(clean_history_path):
        with open(clean_history_path, "r") as f:
            clean_history = json.load(f)
            
    if os.path.exists(attack_history_path):
        with open(attack_history_path, "r") as f:
            attack_history = json.load(f)


    # ----------------- VISUALIZATIONS -----------------

    st.markdown("""
<style>
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color:white;'>== Evaluation Metrics ==</h3>", unsafe_allow_html=True)
    
    if clean_history is None and attack_history is None:
        st.info("💡 No simulation history found. Click 'Run FL Simulation' on the sidebar to execute the training workflow.")
    else:
        # Display Key Performance Indicators (KPIs)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        latest_clean_acc = clean_history["accuracy"][-1] if clean_history else None
        latest_attack_acc = attack_history["accuracy"][-1] if attack_history else None
        
        with kpi_col1:
            val = f"{latest_clean_acc * 100:.2f}%" if latest_clean_acc else "N/A"
            st.metric("Clean Global Accuracy", val)
            
        with kpi_col2:
            val = f"{latest_attack_acc * 100:.2f}%" if latest_attack_acc else "N/A"
            st.metric("Poisoned Global Accuracy", val, 
                      delta=f"-{(latest_clean_acc - latest_attack_acc)*100:.2f}%" if (latest_clean_acc and latest_attack_acc) else None,
                      delta_color="inverse")
            
        with kpi_col3:
            r = clean_history["rounds"][-1] if clean_history else (attack_history["rounds"][-1] if attack_history else 0)
            st.metric("Total Rounds Run", r)
            
        with kpi_col4:
            # Simple threat assessment
            status = "HEALTHY"
            color = "green"
            if latest_attack_acc and latest_clean_acc:
                diff = latest_clean_acc - latest_attack_acc
                if diff > 0.15:
                    status = "CRITICAL COMPROMISE"
                    color = "red"
                elif diff > 0.05:
                    status = "WARN: ANOMALOUS UPDATE"
                    color = "orange"
            elif latest_attack_acc:
                status = "POTENTIAL POISONING ACTIVE"
                color = "orange"
            st.metric("IoT Network Security Status", status)


        # Plot Graphs
        st.markdown("<h3 style='color:white;'>Model Performance Charts</h3>", unsafe_allow_html=True)
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Accuracy Graph
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1E293B')
            ax.set_facecolor('#0F172A')
            
            if clean_history:
                ax.plot(clean_history["rounds"], clean_history["accuracy"], marker='o', color='#10B981', linewidth=2.5, label="Clean Federated Learning")
            if attack_history:
                ax.plot(attack_history["rounds"], attack_history["accuracy"], marker='s', color='#EF4444', linewidth=2.5, linestyle='--', label=f"Attacked (Client {malicious_client_id})")
                
            ax.set_title("Global Model Accuracy vs. Communication Rounds", color='#F8FAFC', fontsize=11, fontweight='bold')
            ax.set_xlabel("Rounds", color='#94A3B8', fontsize=9)
            ax.set_ylabel("Accuracy", color='#94A3B8', fontsize=9)
            ax.tick_params(colors='#94A3B8', labelsize=8)
            ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
            ax.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
            st.pyplot(fig)
            
        with fig_col2:
            # Loss Graph
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1E293B')
            ax.set_facecolor('#0F172A')
            
            if clean_history:
                ax.plot(clean_history["rounds"], clean_history["loss"], marker='o', color='#38BDF8', linewidth=2.5, label="Clean FL Loss")
            if attack_history:
                ax.plot(attack_history["rounds"], attack_history["loss"], marker='s', color='#F59E0B', linewidth=2.5, linestyle='--', label="Attacked FL Loss")
                
            ax.set_title("Global Model Loss vs. Communication Rounds", color='#F8FAFC', fontsize=11, fontweight='bold')
            ax.set_xlabel("Rounds", color='#94A3B8', fontsize=9)
            ax.set_ylabel("Loss", color='#94A3B8', fontsize=9)
            ax.tick_params(colors='#94A3B8', labelsize=8)
            ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
            ax.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
            st.pyplot(fig)

        # Clear Results Button
        if st.button("🧹 Clear Simulation Records"):
            for path in [clean_history_path, attack_history_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            st.rerun()



#               streamlit run dashboard.py
