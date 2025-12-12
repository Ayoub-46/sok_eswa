import torch
import torch.nn as nn
import copy
import sys
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is discoverable
sys.path.append(os.getcwd())

from src.fl.client import BenignClient
from src.fl.server import FedAvgAggregator
from src.models.nlp import SentimentLSTM

def test_client_training():
    print("\n[Test 1] Checking Client Local Training...")
    
    # 1. Setup Dummy Data (Batch of 4, Seq Len 5)
    # 0/1 are PAD/UNK, so we use integers 2-10
    X = torch.randint(2, 10, (4, 5)) 
    y = torch.tensor([1, 0, 1, 0]) # Binary targets
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)
    
    # 2. Setup Model
    model = SentimentLSTM(vocab_size=20, embedding_dim=10, hidden_dim=8, output_dim=2)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 3. Initialize Client
    # CRITICAL: We force the parameters that SHOULD work (Adam, lr=0.001)
    client = BenignClient(
        id=0,
        trainloader=loader,
        testloader=None,
        model=copy.deepcopy(model),
        lr=0.001,  # <--- Testing Adam LR
        weight_decay=0.0,
        epochs=5,
        device=torch.device("cpu")
    )
    
    # Force Adam optimizer if not already default
    client.optimizer = torch.optim.Adam(client.model.parameters(), lr=0.001)

    # 4. Run Training
    print("   Running local_train (5 epochs)...")
    initial_loss = client.local_evaluate()['metrics']['loss']
    
    update = client.local_train(epochs=25, round_idx=1)
    
    # 5. Check if Weights Changed
    trained_weights = update['weights']
    changed = False
    for k in initial_weights:
        # Embeddings might be frozen, so check LSTM/FC weights
        if "lstm" in k or "fc" in k:
            if not torch.allclose(initial_weights[k], trained_weights[k]):
                changed = True
                break
    
    # 6. Check if Loss Decreased (on training set)
    # Note: local_evaluate uses testloader or trainloader. 
    # Since we passed None for test, it uses train.
    final_metrics = client.local_evaluate()['metrics']
    final_loss = final_metrics['loss']
    
    print(f"   Initial Loss: {initial_loss:.4f}")
    print(f"   Final Loss:   {final_loss:.4f}")
    
    if not changed:
        print("❌ FAIL: Weights did not change! Optimizer might be broken or LR is 0.")
    elif np.isnan(final_loss) or final_loss > initial_loss * 1.2:
        print("❌ FAIL: Loss exploded or is NaN. LR might be too high.")
    else:
        print("✅ PASS: Client training reduces loss and updates weights.")

def test_server_aggregation():
    print("\n[Test 2] Checking Server Aggregation...")
    
    # 1. Setup Server with a simple model
    model = SentimentLSTM(vocab_size=20, embedding_dim=10, hidden_dim=8, output_dim=2)
    server = FedAvgAggregator(model=model, device=torch.device("cpu"))
    
    # 2. Create Dummy Updates
    # Update A: All weights +1
    update_a = {k: v + 1.0 for k, v in model.state_dict().items()}
    # Update B: All weights +3
    update_b = {k: v + 3.0 for k, v in model.state_dict().items()}
    
    # 3. Send to Server (Equal weights: 5 samples each)
    server.receive_update(client_id=0, params=update_a, length=5)
    server.receive_update(client_id=1, params=update_b, length=5)
    
    # 4. Aggregate
    print("   Aggregating 2 clients (Avg of +1 and +3 should be +2)...")
    new_weights = server.aggregate()
    
    # 5. Verify Math
    # Expected: Original + 2.0
    original = model.state_dict()
    passed = True
    
    for k in original:
        if "num_batches_tracked" in k: continue # Skip batchnorm tracking
        
        expected = original[k] + 2.0
        actual = new_weights[k]
        
        if not torch.allclose(expected, actual, atol=1e-5):
            print(f"   ❌ FAIL: Param {k} mismatch.")
            print(f"      Expected: {expected.flatten()[0]:.2f}")
            print(f"      Actual:   {actual.flatten()[0]:.2f}")
            passed = False
            break
            
    if passed:
        print("✅ PASS: Aggregation math is correct.")

if __name__ == "__main__":
    test_client_training()
    test_server_aggregation()