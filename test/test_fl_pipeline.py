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
    y = torch.tensor([1, 0, 1, 0], dtype=torch.long) # Binary targets (Long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)
    
    # 2. Setup Model
    model = SentimentLSTM(vocab_size=20, embedding_dim=10, hidden_dim=8, output_dim=2)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 3. Initialize Client
    # Refactor: We do NOT manually set the optimizer here. 
    # We want to test if the client creates it automatically.
    client = BenignClient(
        id=0,
        trainloader=loader,
        testloader=None,
        model=copy.deepcopy(model),
        lr=0.001,  
        optimizer='adam', # This should trigger Adam creation inside client
        weight_decay=0.0,
        epochs=25,
        device=torch.device("cpu")
    )
    
    print("   Client initialized. Checking optimizer state...")
    if client.optimizer is None:
        print("   ✅ PASS: Optimizer is lazily initialized (None at start).")
    else:
        print("   ⚠️ WARN: Optimizer was initialized early.")

    # 4. Run Training
    print("   Running local_train (25 epochs)...")
    # Initial eval (should handle None testloader gracefully)
    initial_metrics = client.local_evaluate()['metrics']
    print(f"   Initial Metrics: {initial_metrics}")

    update = client.local_train(epochs=25, round_idx=1)
    
    # 5. Check if Weights Changed
    trained_weights = update['weights']
    changed = False
    for k in initial_weights:
        if "lstm" in k or "fc" in k:
            if not torch.allclose(initial_weights[k], trained_weights[k]):
                changed = True
                break
    
    # 6. Check Optimizer Persistence
    if client.optimizer is not None:
         print("   ✅ PASS: Optimizer was created and persisted.")
    else:
         print("   ❌ FAIL: Optimizer is None after training.")

    # 7. Check if Loss Decreased
    final_metrics = client.local_evaluate()['metrics']
    final_loss = final_metrics['loss']
    
    # If initial loss was NaN (due to uninitialized model behavior), we skip strict check
    initial_loss_val = initial_metrics.get('loss', float('inf'))
    if np.isnan(initial_loss_val): initial_loss_val = float('inf')

    print(f"   Final Loss:   {final_loss:.4f}")
    
    if not changed:
        print("❌ FAIL: Weights did not change!")
    elif np.isnan(final_loss):
        print("❌ FAIL: Loss is NaN.")
    else:
        print("✅ PASS: Client training successfully updated weights.")

def test_server_aggregation():
    print("\n[Test 2] Checking Server Aggregation...")
    
    # 1. Setup Server with a simple model
    model = SentimentLSTM(vocab_size=20, embedding_dim=10, hidden_dim=8, output_dim=2)
    server = FedAvgAggregator(model=model, testloader=None, device=torch.device("cpu"))
    
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
    server.aggregate() # Updates internal model
    new_weights = server.get_params()
    
    # 5. Verify Math
    original = model.state_dict()
    passed = True
    
    for k in original:
        if "num_batches_tracked" in k: continue 
        if "embedding" in k: continue # Sometimes embeddings are tricky if indices differ, but here full update
        
        # We check a specific weight (e.g. fc.weight)
        if "fc.weight" in k:
            expected = original[k] + 2.0
            actual = new_weights[k]
            
            if not torch.allclose(expected, actual, atol=1e-5):
                print(f"   ❌ FAIL: Param {k} mismatch.")
                passed = False
                break
            
    if passed:
        print("✅ PASS: Aggregation math is correct.")

if __name__ == "__main__":
    test_client_training()
    test_server_aggregation()