import torch
import torch.nn as nn
import sys
import os
import copy
import numpy as np

# Ensure src is discoverable
sys.path.append(os.getcwd())

from src.datasets.sentiment140 import Sentiment140Dataset
from src.models.nlp import SentimentLSTM
from src.fl.server import FedAvgAggregator
from src.fl.client import BenignClient

# Path to your 81% accuracy model
PRETRAINED_PATH = "debug_centralized_model.pth"

def test_fl_sanity():
    print("==================================================")
    print("   FL PIPELINE SANITY CHECK (PRE-TRAINED)")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. SETUP DATA (Using Natural Partition)
    print("\n[1] Loading Dataset...")
    dataset = Sentiment140Dataset(root="data/sentiment140")
    # This might take a minute to generate partitions again
    dataset.get_client_loaders(num_clients=10, strategy="natural")
    
    # Get the global test loader
    test_loader = dataset.get_test_loader(batch_size=128)

    # 2. INITIALIZE MODEL (MUST MATCH TRAINED ARCHITECTURE)
    print("\n[2] Initializing Model & Loading Weights...")
    model = SentimentLSTM(
        vocab_size=len(dataset.word2idx),
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        bidirectional=True,  # <--- CRITICAL: Must match the trained model!
        dropout=0.5
    ).to(device)

    try:
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"   ✅ Successfully loaded weights from {PRETRAINED_PATH}")
    except Exception as e:
        print(f"   ❌ FAILED to load weights: {e}")
        print("      (Did you change bidirectional=True/False between training and FL?)")
        return

    # 3. TEST SERVER EVALUATION
    print("\n[3] Testing Server Evaluation Logic...")
    server = FedAvgAggregator(model=copy.deepcopy(model), testloader=test_loader, device=device)
    
    server_metrics = server.evaluate()
    acc = server_metrics['metrics'].get('main_accuracy', server_metrics['metrics'].get('accuracy', 0.0)) * 100
    
    print(f"   Server Global Accuracy: {acc:.2f}%")
    if acc < 70:
        print("   ❌ FAIL: Server evaluation is broken (or model mismatch).")
    else:
        print("   ✅ PASS: Server correctly evaluates the good model.")

    # 4. TEST CLIENT EVALUATION
    print("\n[4] Testing Client Local Evaluation...")
    # Get a client with actual data
    client_id = 0
    client_loader = dataset.get_client_loaders(num_clients=1, strategy="natural")[0]
    
    client = BenignClient(
        id=client_id,
        trainloader=client_loader,
        testloader=None, # Uses trainloader for eval
        model=copy.deepcopy(model),
        lr=0.001,
        optimizer='adam',
        weight_decay=0.0001,
        device=device
    )
    
    # Force set parameters to the good global model
    client.set_params(server.get_params())
    
    client_metrics = client.local_evaluate()['metrics']
    c_acc = client_metrics['accuracy'] * 100
    print(f"   Client {client_id} Local Accuracy: {c_acc:.2f}%")
    
    # Note: Client accuracy varies wildly on natural partitions (some are easy, some hard)
    # But usually it should be better than 50%
    if c_acc > 55:
        print("   ✅ PASS: Client correctly evaluates local data.")
    else:
        print("   ⚠️ WARN: Client accuracy low. (Could be a hard user, or client eval bug).")

    # 5. TEST TRAINING & AGGREGATION INTEGRITY
    print("\n[5] Testing Training Integrity (1 Round)...")
    print("    Running 1 epoch of training. If accuracy crashes to 50%, training is broken.")
    
    # Train
    update = client.local_train(epochs=1, round_idx=1)
    
    # Send update to server
    server.receive_update(client_id=0, params=update['weights'], length=update['num_samples'])
    
    # Aggregate (With just 1 client, this effectively copies the trained weights)
    server.aggregate()
    
    # Re-evaluate Global Model
    post_metrics = server.evaluate()
    post_acc = post_metrics['metrics'].get('accuracy', 0.0) * 100
    
    print(f"   Post-Training Global Accuracy: {post_acc:.2f}%")
    
    drop = acc - post_acc
    if post_acc < 60:
         print(f"   ❌ CRITICAL FAIL: Accuracy crashed from {acc:.2f}% to {post_acc:.2f}%!")
         print("      Root Cause Likely: Optimizer reset, Learning Rate too high, or Loss Function mismatch.")
    elif drop > 5:
         print(f"   ⚠️ WARN: Accuracy dropped by {drop:.2f}%. Check Learning Rate.")
    else:
         print("   ✅ PASS: FL Pipeline preserves model knowledge.")

if __name__ == "__main__":
    test_fl_sanity()