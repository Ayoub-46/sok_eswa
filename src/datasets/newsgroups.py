import torch
import numpy as np
import os
import re
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import TensorDataset
from .adapter import DatasetAdapter

class NewsgroupsDataset(DatasetAdapter):
    """
    Adapter for the 20 Newsgroups text classification dataset.
    
    1. Source: sklearn.datasets
    2. Processing: Tokenization, GloVe embeddings (optional), Padding/Truncating.
    3. Task: 20-class classification.
    """
    def __init__(self, root: str = "data/newsgroups", download: bool = True):
        super().__init__(root, download, None, None)
        
        # --- Constants ---
        self.max_vocab_size = 30000
        self.seq_len = 200  # Longer sequence length for news articles
        self.embedding_dim = 100
        
        # Special Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.word_to_int = {self.pad_token: 0, self.unk_token: 1}
        self.embedding_weights = None
        
        self._is_loaded = False 

    def load_datasets(self) -> None:
        if self._is_loaded: return
        
        print("--- Loading 20 Newsgroups Data ---")
        
        # 1. Fetch Data
        # Remove headers/footers to prevent model from learning metadata shortcuts
        remove = ('headers', 'footers', 'quotes')
        train_source = fetch_20newsgroups(subset='train', remove=remove, data_home=self.root)
        test_source = fetch_20newsgroups(subset='test', remove=remove, data_home=self.root)
        
        print(f"Loaded {len(train_source.data)} training and {len(test_source.data)} test samples.")

        # 2. Build Vocabulary (Training data only)
        self._build_vocab(train_source.data)
        
        # 3. Load GloVe (Optional)
        self._load_glove_embeddings(dim=self.embedding_dim)
        
        # 4. Tokenize and Pad Data (Pre-process into Tensors)
        # We pre-process into fixed tensors so we can use the default DatasetAdapter
        # partitioning logic which doesn't support custom collate_fns easily.
        x_train, y_train = self._process_text_to_tensor(train_source.data, train_source.target)
        x_test, y_test = self._process_text_to_tensor(test_source.data, test_source.target)
        
        # 5. Create TensorDatasets
        self._train_dataset = TensorDataset(x_train, y_train)
        # Helper for adapter to extract labels efficiently without iteration
        self._train_dataset.targets = y_train.numpy() 
        
        self._test_dataset = TensorDataset(x_test, y_test)
        self._test_dataset.targets = y_test.numpy()
        
        self._is_loaded = True
        print("20 Newsgroups Dataset preparation complete.")

    def _clean_text(self, text: str) -> str:
        text = str(text).lower()
        # Basic cleaning
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_vocab(self, texts):
        print(f"Building Top-{self.max_vocab_size} Vocabulary...")
        all_tokens = []
        for text in texts:
            clean = self._clean_text(text)
            all_tokens.extend(clean.split())
            
        counts = Counter(all_tokens)
        common = counts.most_common(self.max_vocab_size - 2)
        
        for i, (word, c) in enumerate(common):
            self.word_to_int[word] = i + 2 
        
        print(f"Vocabulary size: {len(self.word_to_int)}")

    def _process_text_to_tensor(self, texts, labels):
        x_list = []
        for text in texts:
            clean = self._clean_text(text)
            words = clean.split()
            indices = [self.word_to_int.get(w, self.unk_idx) for w in words]
            
            # Padding / Truncating
            if len(indices) < self.seq_len:
                indices += [self.pad_idx] * (self.seq_len - len(indices))
            else:
                indices = indices[:self.seq_len]
            
            x_list.append(indices)
            
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _load_glove_embeddings(self, dim=100):
        # Similar logic to Sentiment140, assuming GloVe exists at standard path
        glove_path = f"data/glove/glove.6B.{dim}d.txt"
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}. Initializing random.")
            self.embedding_weights = torch.randn(len(self.word_to_int), dim)
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
        
        vocab_size = len(self.word_to_int)
        embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, dim))
        embedding_matrix[self.pad_idx] = np.zeros(dim)
        
        hits = 0
        for word, i in self.word_to_int.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
                hits += 1
                
        self.embedding_weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        print(f"GloVe coverage: {hits}/{vocab_size}")

    def get_embedding_weights(self):
        self.setup()
        return self.embedding_weights
    
    def get_vocab_size(self):
        self.setup()
        return len(self.word_to_int)