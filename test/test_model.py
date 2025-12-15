import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

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
    
    # 2. PREPARE CENTRALIZED SPLIT
    # We combine all "natural" client partitions into one big training set
    # (or just use the raw train_dataset if available)
    
    print("   Creating training subset...")
    # Use indices from natural partitions if generated, else use all
    if not dataset.natural_train_partitions:
        # Trigger partition generation if empty
        dataset.get_client_loaders(num_clients=10, strategy="natural")

    all_indices = []
    for user_indices in dataset.natural_train_partitions.values():
        all_indices.extend(user_indices)
    
    # Train on a smaller subset (e.g. 10k) for rapid debugging, 
    # or all_indices for full performance.
    subset_size = min(len(all_indices), 10000) 
    central_indices = np.random.choice(all_indices, subset_size, replace=False)
    
    # Access .train_dataset (TensorDataset) directly
    central_dataset = Subset(dataset.train_dataset, central_indices)
    
    train_loader = DataLoader(
        central_dataset, 
        batch_size=32, 
        shuffle=True
        # No collate_fn needed for TensorDataset (already padded)
    )
    
    # Test on full test set
    test_loader = dataset.get_test_loader(batch_size=128)

    # 3. INITIALIZE FL MODEL
    print("\n[2] Initializing FL Model...")
    
    # FIX: Use 'word2idx' (from provided Dataset code) not 'word_to_int'
    vocab_size = len(dataset.word2idx) 
    print(f"   Vocabulary Size: {vocab_size}")
    
    embedding_weights = dataset.get_embedding_weights()
    
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,      
        n_layers=2,
        bidirectional=True,  # <--- CRITICAL FIX: Enables backward pass signal
        dropout=0.5
    ).to(device)
    
    if embedding_weights is not None:
        print("   Loading GloVe embeddings...")
        model.load_pretrained_embeddings(embedding_weights, freeze=False)
    
    # 4. SETUP TRAINING
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    print("\n[3] Starting Training Loop (3 Epochs)...")
    
    for epoch in range(1, 4):
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

    # 5. SAVE MODEL
    print(f"\n[4] Saving Model to {OUTPUT_MODEL_PATH}...")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print("    ✅ Model Saved.")

if __name__ == "__main__":
    train_centralized()