import torch
import pandas as pd
import numpy as np
import re
import string
import os
from collections import Counter
from typing import Dict, List, Optional
from torch.utils.data import TensorDataset, DataLoader, Subset

# Ensure this matches your project structure
from .adapter import DatasetAdapter

class Sentiment140Dataset(DatasetAdapter):
    """
    Adapter for Sentiment140 dataset with support for "Natural" (User-based) partitioning.
    
    Features:
    - Loads raw CSV (latin-1 encoding).
    - Cleans text (removes handles, URLs, punctuation).
    - Maps users to specific data indices for realistic Non-IID FL.
    - Loads pre-trained GloVe embeddings.
    """
    def __init__(self, root: str = "data/sentiment140", download: bool = True):
        super().__init__(root, download, None, None)
        
        # Configuration
        self.max_vocab_size = 50000  
        self.max_seq_len = 100
        self.embedding_dim = 100
        self.min_samples_per_user = 30 # Only users with >30 tweets are valid clients
        
        # Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.embedding_weights = None
        
        # Store natural partition indices: { client_id_0: [idx1, idx2...], ... }
        self.natural_train_partitions: Dict[int, List[int]] = {}
        self._is_loaded = False
        
        # Path construction
        self.csv_path = os.path.join(self.root, 'training.1600000.processed.noemoticon.csv')

    def load_datasets(self) -> None:
        if self._is_loaded: return
        print("--- Loading Sentiment140 Data (preserving natural partitions) ---")

        if not os.path.exists(self.csv_path):
             raise FileNotFoundError(f"Sentiment140 file not found at {self.csv_path}. Please download it.")

        cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
        # Read full dataset
        print("Reading CSV...")
        df = pd.read_csv(self.csv_path, encoding='latin-1', names=cols)
        
        # Map target: 0=Negative, 4=Positive -> 0=Negative, 1=Positive
        df['target'] = df['target'].replace(4, 1)

        print("Cleaning text...")
        df['clean_text'] = df['text'].apply(self._clean_text)

        # --- 1. Natural User Partitioning Logic ---
        print("Grouping by user (Natural Partitioning)...")
        
        # Filter for active users only
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= self.min_samples_per_user].index.tolist()
        df = df[df['user'].isin(valid_users)]
        
        # Prepare lists to hold the centralized tensors
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []
        
        current_train_idx = 0
        client_id_counter = 0
        
        # Group by user and split their data
        grouped = df.groupby('user')
        
        for user, group in grouped:
            user_texts = group['clean_text'].values
            user_targets = group['target'].values
            
            # Simple 80/20 split per user
            n_samples = len(user_texts)
            n_train = int(0.8 * n_samples)
            
            u_train_txt = user_texts[:n_train]
            u_train_y = user_targets[:n_train]
            u_test_txt = user_texts[n_train:]
            u_test_y = user_targets[n_train:]
            
            # Record indices for this user (Client) relative to the global list
            # range(start, end)
            indices = list(range(current_train_idx, current_train_idx + len(u_train_txt)))
            self.natural_train_partitions[client_id_counter] = indices
            
            # Append to global lists
            train_texts.extend(u_train_txt)
            train_labels.extend(u_train_y)
            test_texts.extend(u_test_txt)
            test_labels.extend(u_test_y)
            
            current_train_idx += len(u_train_txt)
            client_id_counter += 1

        print(f"Processed {client_id_counter} natural clients from {len(df)} tweets.")

        # --- 2. Build Vocabulary (Training data only) ---
        self._build_vocab(train_texts)

        # --- 3. Load GloVe Embeddings ---
        self._load_glove_embeddings(dim=self.embedding_dim)

        # --- 4. Process text to Integers (Padding/Truncating) ---
        print("Tokenizing and creating tensors...")
        x_train, y_train = self._process_text_to_tensor(train_texts, train_labels)
        x_test, y_test = self._process_text_to_tensor(test_texts, test_labels)

        # --- 5. Create TensorDatasets ---
        self._train_dataset = TensorDataset(x_train, y_train)
        # Helper for efficient label extraction in adapter (if needed)
        self._train_dataset.targets = y_train.numpy()
        
        self._test_dataset = TensorDataset(x_test, y_test)
        self._test_dataset.targets = y_test.numpy()

        self._is_loaded = True
        print("Sentiment140 preparation complete.")

    def get_client_loaders(self, num_clients: int, batch_size: int = 32, strategy: str = "iid", seed: int = 0, **strategy_args) -> Dict[int, DataLoader]:
        """
        Override to support 'natural' strategy which uses the user groupings.
        """
        self.setup()
        
        if strategy == "natural":
            print(f"--- Generating {num_clients} Client Loaders using Natural User Partition ---")
            loaders = {}
            
            # Available real users
            available_clients = list(self.natural_train_partitions.keys())
            
            if num_clients > len(available_clients):
                print(f"Warning: Requested {num_clients} clients, but only {len(available_clients)} valid users found. Using all available.")
                num_clients = len(available_clients)
            
            # Randomly select 'num_clients' users from the available pool
            rng = np.random.RandomState(seed)
            selected_user_ids = rng.choice(available_clients, num_clients, replace=False)

            for i, original_user_id in enumerate(selected_user_ids):
                # map logic client_id (0..N) -> real user data
                indices = self.natural_train_partitions[original_user_id]
                
                # Create Subset based on indices
                subset = Subset(self.train_dataset, indices)
                loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
                
            return loaders

        else:
            # Fallback to standard IID/Dirichlet logic (from Adapter)
            return super().get_client_loaders(num_clients, batch_size, strategy, seed, **strategy_args)

    # --- Helpers ---

    def _clean_text(self, text):
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def _build_vocab(self, texts):
        print(f"Building Vocabulary (Max: {self.max_vocab_size})...")
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        most_common = counter.most_common(self.max_vocab_size - 2)
        
        # Start at 2 (0=PAD, 1=UNK)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            
        print(f"Vocabulary size: {len(self.word2idx)}")

    def _process_text_to_tensor(self, texts, labels):
        x_list = []
        for text in texts:
            words = text.split()
            indices = [self.word2idx.get(w, self.unk_idx) for w in words]
            
            # Padding / Truncating
            if len(indices) < self.max_seq_len:
                indices += [self.pad_idx] * (self.max_seq_len - len(indices))
            else:
                indices = indices[:self.max_seq_len]
            x_list.append(indices)
            
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _load_glove_embeddings(self, dim=100):
        # Assumes data/glove/glove.6B.100d.txt structure
        # self.root is usually data/sentiment140, so we go up one level
        glove_path = os.path.join(os.path.dirname(self.root), 'glove', f'glove.6B.{dim}d.txt')
        
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}. Initializing random embeddings.")
            self.embedding_weights = torch.randn(len(self.word2idx), dim)
            return

        print(f"Loading GloVe embeddings from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError:
                    continue
        
        # Create matrix
        matrix = np.zeros((len(self.word2idx), dim))
        hits = 0
        
        for word, idx in self.word2idx.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                matrix[idx] = vec
                hits += 1
            elif word == self.unk_token:
                # Initialize UNK with random normal
                matrix[idx] = np.random.normal(scale=0.6, size=(dim,))
                
        self.embedding_weights = torch.tensor(matrix, dtype=torch.float32)
        print(f"GloVe loaded. Coverage: {hits}/{len(self.word2idx)} words.")

    def get_embedding_weights(self):
        self.setup()
        return self.embedding_weights
    
    def get_vocab_size(self):
        self.setup()
        return len(self.word2idx)