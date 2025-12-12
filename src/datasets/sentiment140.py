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
    Standardized Sentiment140 Adapter (Corrected for Research Consistency).
    
    1. Source: Local CSV.
    2. Cleaning: Preserves some punctuation for emoticons; tokenizes URLs/Mentions.
    3. Vocab: Top-K (50k), built STRICTLY on training data to prevent leakage.
    4. Embeddings: GloVe.
    """
    def __init__(self, root: str = "data/sentiment140", csv_name: str = "training.1600000.processed.noemoticon.csv"):
        super().__init__(root, False, None, None)
        
        self.csv_path = os.path.join(root, csv_name)
        
        # --- Research Constants ---
        self.max_vocab_size = 50000 
        self.seq_len = 25
        self.embedding_dim = 100
        
        # Special Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.user_token = "<USER>" # Token for @mentions
        self.url_token = "<URL>"   # Token for http://...
        
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
            raise FileNotFoundError(f"Sentiment140 CSV not found at {self.csv_path}")

        print(f"--- Loading Sentiment140 from Local CSV: {self.csv_path} ---")
        
        # 1. Read CSV
        # Using latin-1 as per standard Sentiment140 format
        df = pd.read_csv(
            self.csv_path, 
            encoding='latin-1', 
            header=None, 
            names=['sentiment', 'id', 'date', 'flag', 'user', 'text']
        )
        
        print("Grouping by user to filter active clients...")
        
        user_counts = df['user'].value_counts()
        
        min_samples = 5
        active_users = user_counts[user_counts >= min_samples].index.tolist()
        
        # 2. Select Subset of Users
        # Randomly select 'target_subset_users' to keep dataset manageable while preserving FL structure
        target_subset_users = 2000 
        
        np.random.seed(42)
        selected_users = np.random.choice(active_users, min(len(active_users), target_subset_users), replace=False)
        
        # 3. Filter dataframe
        df_small = df[df['user'].isin(selected_users)].copy()
        
        # Shuffle rows within the subset
        df_small = df_small.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Subset created: {len(df_small)} samples from {len(selected_users)} users.")
        
        # 4. Convert to HF Dataset
        self.hf_dataset = Dataset.from_pandas(df_small)
        
        # 5. Partition by User FIRST (Crucial for correct Vocab building)
        print("Partitioning data by user...")
        grouped = df_small.groupby('user')
        indices_map = grouped.indices # Dictionary of {user: [row_indices]}
        
        train_indices_buffer = [] # We will use this to build vocab
        
        # We iterate over selected users to ensure stable order
        valid_users = [u for u in selected_users if u in indices_map]

        for user in valid_users:
            indices = indices_map[user]
            np.random.shuffle(indices) # Randomize local split
            
            # Standard 80/20 split per client
            split = int(len(indices) * 0.8)
            if split == 0 and len(indices) > 0: split = 1
                
            self.train_partitions[user] = indices[:split]
            self.test_partitions[user] = indices[split:]
            
            # Collect ONLY training indices for vocab building
            train_indices_buffer.extend(indices[:split])
            
        print(f"Partitions created for {len(self.train_partitions)} clients.")

        # 6. Build Vocabulary (Strictly on Training Data)
        print(f"Building Top-{self.max_vocab_size} Vocabulary on Training Data only...")
        # Create a temporary subset view of only training data
        train_subset_for_vocab = Subset(self.hf_dataset, train_indices_buffer)
        self._build_vocab(train_subset_for_vocab)

        # 7. Load GloVe
        print(f"Loading GloVe (Dim {self.embedding_dim})...")
        self._load_glove_embeddings(dim=self.embedding_dim)
        
        # 8. Centralized Test Set (Optional in FL, but good for global eval)
        all_test_indices = []
        for u in self.test_partitions:
            all_test_indices.extend(self.test_partitions[u])
        
        np.random.shuffle(all_test_indices)
        self._test_dataset = Subset(self.hf_dataset, all_test_indices)
        self.raw_dataset = self.hf_dataset
        
        self._is_loaded = True

    def _clean_text(self, text: str) -> str:
        text = str(text).lower()
        
        # 1. Replace mentions/URLs first (Preserve these specific tokens)
        text = re.sub(r'@[a-z0-9_]+', f' {self.user_token} ', text) 
        text = re.sub(r'https?://[^\s]+', f' {self.url_token} ', text)
        
        # 2. Add spaces around crucial punctuation (FIXED)
        # This ensures "happy!" becomes "happy !" so both are found in GloVe
        text = re.sub(r'([!?:\)\(\-])', r' \1 ', text)
        
        # 3. Remove illegal characters (allow the ones we spaced out)
        text = re.sub(r'[^a-z0-9 !?:\)\(\-]', '', text)
        
        # 4. Collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _build_vocab(self, dataset_subset):
        """
        Builds vocab from a Subset of the dataset (Training data only).
        """
        # Subset doesn't allow direct column access like dataset['text']
        # We must access the underlying dataset via indices
        
        all_tokens = []
        
        # Optimization: Access underlying huge array once, filter by indices
        # (This is faster than iterating one by one)
        source_data = dataset_subset.dataset['text']
        indices = dataset_subset.indices
        
        for idx in indices:
            text = source_data[idx]
            clean = self._clean_text(text)
            all_tokens.extend(clean.split())
            
        counts = Counter(all_tokens)
        
        # Reserve spots for PAD and UNK
        common = counts.most_common(self.max_vocab_size - 2)
        
        for i, (word, c) in enumerate(common):
            self.word_to_int[word] = i + 2 

        print(f"Vocabulary built. Size: {len(self.word_to_int)}")

    def _load_glove_embeddings(self, dim=100):
        glove_path = f"data/glove/glove.6B.{dim}d.txt"
        
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}. Using random initialization.")
            vocab_size = len(self.word_to_int)
            self.embedding_weights = torch.randn(vocab_size, dim)
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
        
        # Initialize with random normal (scale=0.6 is a heuristic for GloVe var)
        embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, dim))
        
        # Explicitly zero out padding
        embedding_matrix[self.pad_idx] = np.zeros(dim)
        
        hits = 0
        for word, i in self.word_to_int.items():
            if i < 2: continue # Skip PAD/UNK
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
                
        self.embedding_weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        print(f"GloVe loaded. Coverage: {hits}/{vocab_size} ({hits/vocab_size if vocab_size>0 else 0:.1%})")

    def _collate_fn(self, batch):
        x_batch = []
        y_batch = []
        
        for item in batch:
            clean_text = self._clean_text(item['text'])
            words = clean_text.split()
            
            indices = [self.word_to_int.get(w, self.unk_idx) for w in words]
            
            # Padding / Truncating
            if len(indices) < self.seq_len:
                indices += [self.pad_idx] * (self.seq_len - len(indices))
            else:
                indices = indices[:self.seq_len]
                
            x_batch.append(torch.tensor(indices, dtype=torch.long))
            
            # Label Mapping: 
            # 0 = Negative, 2 = Neutral, 4 = Positive
            # Map 4 -> 1 (Pos), 0 -> 0 (Neg).
            raw_label = item['sentiment']
            label = 1 if raw_label == 4 else 0
            
            y_batch.append(torch.tensor(label, dtype=torch.long))            
        return torch.stack(x_batch), torch.stack(y_batch)

    def get_client_loaders(self, num_clients: int, batch_size: int = 32, seed: int = 0, **kwargs):
        self.setup() # Ensure loaded
        
        # Deterministic client selection
        users = sorted(list(self.train_partitions.keys()))
        
        np.random.seed(seed)
        if num_clients > len(users):
            print(f"Warning: Requested {num_clients} clients but only have {len(users)}. Returning all.")
            selected = users
        else:
            selected = np.random.choice(users, num_clients, replace=False)
        
        loaders = {}
        for i, user in enumerate(selected):
            indices = self.train_partitions[user]
            subset = Subset(self.raw_dataset, indices)
            loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            
        return loaders

    def get_test_loader(self, batch_size: int = 128):
        self.setup()
        return DataLoader(self._test_dataset, batch_size=batch_size, collate_fn=self._collate_fn)
    
    def get_embedding_weights(self):
        self.setup()
        return self.embedding_weights.clone().detach()
    def get_vocab_size(self):
        self.setup()
        return len(self.word_to_int)