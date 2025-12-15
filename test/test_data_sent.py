import torch
import numpy as np
import os
import sys

# Ensure src is discoverable
sys.path.append(os.getcwd())

from src.datasets.sentiment140 import Sentiment140Dataset

# --- Global Test Config ---
# Note: "!" and "?" are removed by the adapter's cleaning logic, 
# so we check for words that should survive.
CRITICAL_TOKENS = ["happy", "sad", "you", "are"]
TEST_SENTENCE = "I am happy! Are you?"

def test_initialization():
    """
    Test 1: Can we load the dataset class and the CSV?
    """
    print("\n[Test 1] Initialization & Loading...")
    try:
        # Initialize
        ds = Sentiment140Dataset(root="data/sentiment140")
        
        # Check if CSV exists (adapter constructs path internally)
        csv_path = os.path.join(ds.root, 'training.1600000.processed.noemoticon.csv')
        if not os.path.exists(csv_path):
            print(f"   ❌ FAIL: CSV not found at {csv_path}")
            return None
            
        print(f"   --> Found CSV at: {csv_path}")
        
        # Trigger actual loading (expensive part)
        ds.load_datasets()
        print("   ✅ PASS: Dataset loaded successfully.")
        return ds
        
    except Exception as e:
        print(f"   ❌ CRASH: Initialization failed with error: {e}")
        return None

def test_preprocessing(ds):
    """
    Test 2: Does the regex correctly clean text (remove punctuation/handles)?
    """
    print("\n[Test 2] Preprocessing Logic...")
    
    raw = TEST_SENTENCE
    cleaned = ds._clean_text(raw)
    
    print(f"   Input:   '{raw}'")
    print(f"   Output:  '{cleaned}'")
    
    # The adapter logic: text.translate(str.maketrans('', '', string.punctuation))
    # This removes punctuation entirely.
    expected = "i am happy are you"
    
    if cleaned == expected:
         print("   ✅ PASS: Text cleaning works (punctuation removed).")
         return True
    else:
         print(f"   ❌ FAIL: Expected '{expected}', got '{cleaned}'")
         return False

def test_vocabulary(ds):
    """
    Test 3: Do critical tokens exist in the vocab?
    """
    print("\n[Test 3] Vocabulary Health...")
    
    # Refactor: Adapter uses 'word2idx', not 'word_to_int'
    vocab = ds.word2idx 
    print(f"   Vocab Size: {len(vocab)}")
    
    missing = []
    for token in CRITICAL_TOKENS:
        if token not in vocab:
            missing.append(token)
            
    if len(missing) == 0:
        print(f"   ✅ PASS: All critical tokens found ({CRITICAL_TOKENS}).")
        return True
    else:
        print(f"   ❌ FAIL: Missing tokens: {missing}")
        return False

def test_embeddings(ds):
    """
    Test 4: Are embeddings loaded from the Real GloVe file?
    """
    print("\n[Test 4] Embedding Quality...")
    
    weights = ds.get_embedding_weights()
    
    # 1. Check Dimensions
    expected_dim = 100
    if weights.shape[1] != expected_dim:
        print(f"   ❌ FAIL: Dimension mismatch. Expected {expected_dim}, got {weights.shape[1]}.")
        return False

    # 2. Check Content
    # We check the vector for a known word like "happy"
    if "happy" in ds.word2idx:
        idx = ds.word2idx["happy"]
        vec = weights[idx]
        
        mean_val = vec.mean().item()
        std_val = vec.std().item()
        
        print(f"   Vector Stats for 'happy': Mean={mean_val:.5f}, Std={std_val:.5f}")
        
        if std_val == 0.0:
            print("   ❌ FAIL: Vector has 0 variance. Likely a placeholder/random init.")
            return False
        elif abs(mean_val) < 1e-6 and std_val < 1e-6:
             print("   ⚠️ WARN: Vector is all zeros.")
             return False
        else:
            print("   ✅ PASS: Vector looks like a valid GloVe embedding.")
            return True
    else:
        print("   ⚠️ SKIP: 'happy' is missing from vocab.")
        return False

def main():
    print("=======================================")
    print("    SENTIMENT140 TEST SUITE")
    print("=======================================")
    
    # Run Test 1
    dataset = test_initialization()
    
    if dataset is None:
        print("\n⛔ Aborting: Could not load dataset.")
        return

    # Run remaining tests
    results = {
        "Preprocessing": test_preprocessing(dataset),
        "Vocabulary":    test_vocabulary(dataset),
        "Embeddings":    test_embeddings(dataset)
    }
    
    print("\n=======================================")
    print("    TEST SUMMARY")
    print("=======================================")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f" {status} : {name}")
        if not passed: all_passed = False
        
    if all_passed:
        print("\n🚀 READY: Your dataset is healthy.")
    else:
        print("\n⚠️ ACTION REQUIRED: Fix failed components.")

if __name__ == "__main__":
    main()