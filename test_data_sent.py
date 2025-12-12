import torch
import numpy as np
import os
import sys

# Ensure src is discoverable
sys.path.append(os.getcwd())

from src.datasets.sentiment140 import Sentiment140Dataset

# --- Global Test Config ---
CRITICAL_TOKENS = ["!", "?", "happy", "sad"]
TEST_SENTENCE = "I am happy! Are you?"

def test_initialization():
    """
    Test 1: Can we load the dataset class and the CSV?
    """
    print("\n[Test 1] Initialization & Loading...")
    try:
        # Initialize
        ds = Sentiment140Dataset(root="data/sentiment140")
        
        # Check if CSV exists before trying to load (avoids hard crash)
        if not os.path.exists(ds.csv_path):
            print(f"   ❌ FAIL: CSV not found at {ds.csv_path}")
            return None
            
        print(f"   --> Found CSV at: {ds.csv_path}")
        
        # Trigger actual loading (expensive part)
        ds.load_datasets()
        print("   ✅ PASS: Dataset loaded successfully.")
        return ds
        
    except Exception as e:
        print(f"   ❌ CRASH: Initialization failed with error: {e}")
        return None

def test_preprocessing(ds):
    """
    Test 2: Does the regex correctly separate punctuation?
    """
    print("\n[Test 2] Preprocessing Logic...")
    
    raw = TEST_SENTENCE
    cleaned = ds._clean_text(raw)
    
    print(f"   Input:   '{raw}'")
    print(f"   Output:  '{cleaned}'")
    
    # We expect punctuation to be space-separated
    if "happy !" in cleaned or "happy  !" in cleaned:
         print("   ✅ PASS: Punctuation separation works.")
         return True
    else:
         print("   ❌ FAIL: Punctuation is 'glued' (e.g., 'happy!'). Fix regex in dataset.py.")
         return False

def test_vocabulary(ds):
    """
    Test 3: Do critical tokens exist in the vocab?
    """
    print("\n[Test 3] Vocabulary Health...")
    
    vocab = ds.word_to_int
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
        print("            (This usually means preprocessing is stripping them or gluing them).")
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

    # 2. Check Content (The "Fake File" Check)
    # We check the vector for a known word like "happy"
    if "happy" in ds.word_to_int:
        idx = ds.word_to_int["happy"]
        vec = weights[idx]
        
        mean_val = vec.mean().item()
        std_val = vec.std().item()
        
        print(f"   Vector Stats for 'happy': Mean={mean_val:.5f}, Std={std_val:.5f}")
        
        if std_val == 0.0:
            print("   ❌ FAIL: Vector has 0 variance (Values are identical).")
            print("            This indicates a FAKE/DUMMY GloVe file is being used.")
            return False
        elif abs(mean_val) < 1e-6 and std_val < 1e-6:
             print("   ⚠️ WARN: Vector is all zeros. 'happy' might not be in your GloVe file.")
             return False
        else:
            print("   ✅ PASS: Vector looks like a valid GloVe embedding.")
            return True
    else:
        print("   ⚠️ SKIP: Cannot check embedding values because 'happy' is missing from vocab.")
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
        print("\n🚀 READY: Your dataset is healthy. You can start FL training.")
    else:
        print("\n⚠️ ACTION REQUIRED: Fix the failed components before training.")

if __name__ == "__main__":
    main()