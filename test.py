import torch
from src.datasets.sentiment140 import Sentiment140Dataset

def verify():
    # 1. Initialize Adapter
    print("--- Initializing Adapter ---")
    adapter = Sentiment140Dataset(root="data/sentiment140")
    adapter.load_datasets()
    
    # 2. Check Test Loader
    print("\n--- Checking Test Loader ---")
    test_loader = adapter.get_test_loader(batch_size=20)
    
    # Get first batch
    x, y = next(iter(test_loader))
    
    print(f"Batch X Shape: {x.shape}") # Should be [20, 25]
    print(f"Batch Y Shape: {y.shape}") # Should be [20]
    
    print(f"\nSample Labels (First 20): {y.tolist()}")
    
    # 3. Verify Balance
    ones = (y == 1).sum().item()
    zeros = (y == 0).sum().item()
    print(f"Positives (1): {ones}")
    print(f"Negatives (0): {zeros}")
    
    if zeros == 20 or ones == 20:
        print("❌ CRITICAL: Batch is not mixed! It contains only one label.")
    else:
        print("✅ Success: Batch contains mixed labels.")

if __name__ == "__main__":
    verify()