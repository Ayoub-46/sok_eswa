import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

# Import the EXACT classes used in FL
from src.datasets.sentiment140 import Sentiment140Dataset
from src.models.nlp import SentimentLSTM

OUTPUT_MODEL_PATH = "debug_centralized_model.pth"

def train_centralized():
    print("==================================================")
    print("   DEBUG: CENTRALIZED TRAINING WITH FL MODEL")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. LOAD DATA 
    print("\n[1] Loading Dataset...")
    dataset = Sentiment140Dataset(root="data/sentiment140")
    dataset.load_datasets()
    
    # Merge all partitions for centralized training
    all_indices = []
    for user_indices in dataset.train_partitions.values():
        all_indices.extend(user_indices)
    
    # Train on a substantial subset (e.g. 20k samples) to ensure convergence
    subset_size = min(len(all_indices), 20000) 
    central_indices = np.random.choice(all_indices, subset_size, replace=False)
    
    central_dataset = Subset(dataset.hf_dataset, central_indices)
    
    train_loader = DataLoader(
        central_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=dataset._collate_fn
    )
    
    # Test on full test set
    test_loader = dataset.get_test_loader(batch_size=128)

    # 2. INITIALIZE FL MODEL
    print("\n[2] Initializing FL Model...")
    vocab_size = len(dataset.word_to_int)
    embedding_weights = dataset.get_embedding_weights()
    
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,      
        n_layers=2,
        dropout=0.5
    ).to(device)
    
    model.load_pretrained_embeddings(embedding_weights, freeze=False)
    
    # 3. SETUP TRAINING
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam is the key!

    print("\n[3] Starting Training Loop (5 Epochs)...")
    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(0)
        
        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    # 4. SAVE MODEL
    print(f"\n[4] Saving Model to {OUTPUT_MODEL_PATH}...")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print("    ✅ Model Saved.")

if __name__ == "__main__":
    train_centralized()