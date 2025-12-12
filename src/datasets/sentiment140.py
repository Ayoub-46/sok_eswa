import torch
import numpy as np
import os
import re
import pandas as pd
from collections import Counter
from datasets import Dataset 
from torch.utils.data import DataLoader, Subset
from .adapter import DatasetAdapter

class Sentiment140Dataset(DatasetAdapter):
    """
    Standardized Sentiment140 Adapter (Local CSV Version).
    
    1. Source: Local CSV (training.1600000.processed.noemoticon.csv)
    2. Cleaning: Regex removal of URLs, Mentions, Special Chars.
    3. Vocab: Top-K (50k).
    4. Embeddings: GloVe.
    """
    def __init__(self, root: str = "data/sentiment140", csv_name: str = "training.1600000.processed.noemoticon.csv"):
        super().__init__(root, False, None, None)
        
        self.csv_path = os.path.join(root, csv_name)
        
        # --- Research Constants ---
        self.max_vocab_size = 50000 
        self.seq_len = 25
        self.embedding_dim = 100
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.word_to_int = {self.pad_token: 0, self.unk_token: 1}
        self.embedding_weights = None
        
        self.train_partitions = {}
        self.test_partitions = {}
        self._is_loaded = False 

    def load_datasets(self) -> None:
        if self._is_loaded: return

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Sentiment140 CSV not found at {self.csv_path}. Please download it from Kaggle.")

        print(f"--- Loading Sentiment140 from Local CSV: {self.csv_path} ---")
        
        # 1. Read CSV with Pandas
        # Kaggle Dataset has no headers. Columns: [target, id, date, flag, user, text]
        # Encoding is usually Latin-1 (ISO-8859-1)
        df = pd.read_csv(
            self.csv_path, 
            encoding='latin-1', 
            header=None, 
            names=['sentiment', 'id', 'date', 'flag', 'user', 'text']
        )
        
        # 2. Manual Shuffle (Critical for this sorted dataset)
        print("Shuffling dataset...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 3. Take Subset (e.g., 100k)
        subset_size = 100000
        df_small = df.iloc[:subset_size]
        
        # 4. Convert to HF Dataset for efficiency (keeps downstream code compatible)
        # This wrapper is lightweight and allows us to use .select() if needed later
        self.hf_dataset = Dataset.from_pandas(df_small)

        # 5. Build Vocabulary
        print(f"Building Top-{self.max_vocab_size} Vocabulary...")
        self._build_vocab(self.hf_dataset)
        
        # 6. Load GloVe
        print(f"Loading GloVe (Dim {self.embedding_dim})...")
        self._load_glove_embeddings(dim=self.embedding_dim)
        
        # 7. Partition by User
        print("Partitioning data by user...")
        grouped = df_small.groupby('user')
        
        min_samples = 5
        valid_users = [u for u, g in grouped if len(g) >= min_samples]
        indices_map = grouped.indices
        
        for user in valid_users:
            indices = indices_map[user]
            np.random.shuffle(indices) # Randomize local split
            
            split = int(len(indices) * 0.8)
            if split == 0 and len(indices) > 0: split = 1
                
            self.train_partitions[user] = indices[:split]
            self.test_partitions[user] = indices[split:]
            
        print(f"Loaded {len(self.train_partitions)} clients.")
        
        # 8. Centralized Test Set
        all_test_indices = []
        for u in self.test_partitions:
            all_test_indices.extend(self.test_partitions[u])
        
        np.random.shuffle(all_test_indices)
        
        # We wrap the HF dataset in a Subset for the test loader
        self._test_dataset = Subset(self.hf_dataset, all_test_indices)
        self.raw_dataset = self.hf_dataset
        
        self._is_loaded = True

    def _clean_text(self, text: str) -> str:
        text = str(text).lower() # Ensure string
        text = re.sub(r'@[a-z0-9_]+', '', text)  # Remove mentions
        text = re.sub(r'https?://[^\s]+', '', text) # Remove URLs
        text = re.sub(r'[^a-z0-9 ]', '', text)   # Keep alpha-numeric + space
        return text

    def _build_vocab(self, dataset):
        # We iterate over the first 50k rows of the HF dataset object
        samples = dataset.select(range(min(50000, len(dataset))))['text']
        all_tokens = []
        
        for text in samples:
            clean = self._clean_text(text)
            all_tokens.extend(clean.split())
            
        counts = Counter(all_tokens)
        common = counts.most_common(self.max_vocab_size - 2)
        
        for i, (word, c) in enumerate(common):
            self.word_to_int[word] = i + 2 

    def _load_glove_embeddings(self, dim=100):
        glove_path = f"data/glove/glove.6B.{dim}d.txt"
        
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}. Using random initialization.")
            return

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

        vocab_size = len(self.word_to_int)
        embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, dim))
        embedding_matrix[self.pad_idx] = np.zeros(dim)
        
        hits = 0
        for word, i in self.word_to_int.items():
            if i < 2: continue 
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
                
        self.embedding_weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        print(f"GloVe loaded. Coverage: {hits}/{vocab_size} ({hits/vocab_size:.1%})")

    def _collate_fn(self, batch):
        x_batch = []
        y_batch = []
        
        for item in batch:
            clean_text = self._clean_text(item['text'])
            words = clean_text.split()
            
            indices = [self.word_to_int.get(w, self.unk_idx) for w in words]
            
            if len(indices) < self.seq_len:
                indices += [self.pad_idx] * (self.seq_len - len(indices))
            else:
                indices = indices[:self.seq_len]
                
            x_batch.append(torch.tensor(indices, dtype=torch.long))
            
            # Label Mapping: CSV has 0 (Neg) and 4 (Pos) -> Map 4 to 1
            label = 1 if item['sentiment'] == 4 else 0
            y_batch.append(torch.tensor(label, dtype=torch.long))
            
        return torch.stack(x_batch), torch.stack(y_batch)

    def get_client_loaders(self, num_clients: int, batch_size: int = 32, seed: int = 0, **kwargs):
        self.setup()
        users = sorted(list(self.train_partitions.keys()))
        np.random.seed(seed)
        selected = np.random.choice(users, min(num_clients, len(users)), replace=False)
        
        loaders = {}
        for i, user in enumerate(selected):
            indices = self.train_partitions[user]
            # Subset of the HF dataset object
            subset = Subset(self.raw_dataset, indices)
            loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            
        return loaders

    def get_test_loader(self, batch_size: int = 128):
        self.setup()
        return DataLoader(self._test_dataset, batch_size=batch_size, collate_fn=self._collate_fn)