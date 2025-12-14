import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Ensure src is discoverable
sys.path.append(os.getcwd())

from src.datasets.sentiment140 import Sentiment140Dataset
from src.models.nlp import SentimentLSTM
from src.fl.server import FedAvgAggregator

MODEL_PATH = "debug_centralized_model.pth"

def test_server_eval():
    print("==================================================")
    print("   DEBUG: TESTING SERVER EVALUATION LOGIC")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup Data (Test Set Only)
    print("\n[1] Loading Test Data...")
    dataset = Sentiment140Dataset(root="data/sentiment140")
    dataset.load_datasets()
    
    # Get the same test loader used in centralized training
    test_loader = dataset.get_test_loader(batch_size=128)
    print(f"Test Set Size: {len(test_loader.dataset)}")

    # 2. Initialize Model Architecture
    print("\n[2] Initializing Model & Loading Trained Weights...")
    vocab_size = len(dataset.word_to_int)
    
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,      
        n_layers=2,
        dropout=0.5
    )
    
    # Load the trained weights from the previous step
    if not os.path.exists(MODEL_PATH):
        print(f"❌ CRITICAL: {MODEL_PATH} not found.")
        print("   Run 'debug_centralized_fl_model.py' first to generate it.")
        return

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("   ✅ Weights loaded.")

    # 3. Initialize FL Server
    print("\n[3] Initializing FedAvgAggregator...")
    # The server takes the model and the test loader
    server = FedAvgAggregator(
        model=model, 
        testloader=test_loader, 
        device=device
    )

    # 4. Run Evaluate
    print("\n[4] Running server.evaluate()...")
    results = server.evaluate()
    
    # 5. Report Results
    metrics = results['metrics']
    acc = metrics.get('main_accuracy', metrics.get('accuracy', -1))
    loss = metrics.get('loss', -1)
    
    print("\n--------------------------------------------------")
    print(f"SERVER EVALUATION RESULT:")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   Loss:     {loss:.4f}")
    print("--------------------------------------------------")

    # Final Verdict
    if acc > 0.70:
        print("\n✅ PASS: Server evaluation matches centralized performance.")
        print("         The issue is DEFINITELY in the Client Training loop (Optimizer/LR).")
    elif acc > 0.45 and acc < 0.60:
        print("\n❌ FAIL: Server reports random guessing (~50%).")
        print("         Possible causes:")
        print("         1. Server uses wrong Loss (BCE vs CrossEntropy).")
        print("         2. Server post-processing (Sigmoid vs Argmax) is wrong.")
        print("         3. Data collator in server is incompatible.")
    else:
        print("\n⚠️ WARN: Accuracy is unexpected (neither good nor random). Check data.")

if __name__ == "__main__":
    test_server_eval()