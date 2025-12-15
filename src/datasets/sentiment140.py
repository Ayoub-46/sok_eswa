import torch
import pandas as pd
import numpy as np
import re
import string
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from .adapter import DatasetAdapter

class Sentiment140Dataset(DatasetAdapter):
    """
    Adapter for Sentiment140 dataset.
    Handles loading, cleaning, tokenization, padding, and GloVe embedding loading.
    """
    def __init__(self, root: str = "data/sentiment140", download: bool = True):
        super().__init__(root, download, None, None)
        
        # Configuration matches your centralized script
        self.max_vocab_size = 50000  
        self.max_seq_len = 100
        self.embedding_dim = 100
        self.sample_size = None # Set to e.g., 50000 for debugging
        
        # Special Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.embedding_weights = None
        self._is_loaded = False

    def load_datasets(self) -> None:
        if self._is_loaded: return
        print("--- Loading Sentiment140 Data ---")

        # 1. Load Data
        data_path = os.path.join(self.root, 'training.1600000.processed.noemoticon.csv')
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Sentiment140 file not found at {data_path}. Please download it.")

        cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(data_path, encoding='latin-1', names=cols)
        
        # Map target: 0=Negative, 4=Positive -> 0, 1
        df['target'] = df['target'].replace(4, 1)

        # Subsample if configured
        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)

        # 2. Clean Text
        print("Cleaning text...")
        df['clean_text'] = df['text'].apply(self._clean_text)
        
        # 3. Split Data (80/20 split)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['clean_text'].values, df['target'].values, 
            test_size=0.2, random_state=42, stratify=df['target'].values
        )

        # 4. Build Vocabulary (on Train only)
        self._build_vocab(train_texts)

        # 5. Load GloVe Embeddings
        self._load_glove_embeddings(dim=self.embedding_dim)

        # 6. Process into Tensors
        print("Tokenizing and padding...")
        x_train, y_train = self._process_text_to_tensor(train_texts, train_labels)
        x_test, y_test = self._process_text_to_tensor(test_texts, test_labels)

        # 7. Create Datasets
        self._train_dataset = TensorDataset(x_train, y_train)
        # Helper for efficient label extraction in the adapter
        self._train_dataset.targets = y_train.numpy()

        self._test_dataset = TensorDataset(x_test, y_test)
        self._test_dataset.targets = y_test.numpy()

        self._is_loaded = True
        print("Sentiment140 preparation complete.")

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def _build_vocab(self, texts):
        print(f"Building Vocabulary (Max: {self.max_vocab_size})...")
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        most_common = counter.most_common(self.max_vocab_size - 2)
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
        # Assumes standard path structure: data/glove/glove.6B.100d.txt
        glove_path = os.path.join(os.path.dirname(self.root), 'glove', f'glove.6B.{dim}d.txt')
        
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe file not found at {glove_path}. Initializing random.")
            self.embedding_weights = torch.randn(len(self.word2idx), dim)
            return

        print(f"Loading GloVe embeddings from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        # Create matrix
        matrix = np.zeros((len(self.word2idx), dim))
        hits = 0
        for word, idx in self.word2idx.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                matrix[idx] = vec
                hits += 1
            elif word == self.unk_token:
                matrix[idx] = np.random.normal(scale=0.6, size=(dim,))
                
        self.embedding_weights = torch.tensor(matrix, dtype=torch.float32)
        print(f"GloVe loaded. Hits: {hits}/{len(self.word2idx)}")

    def get_vocab_size(self):
        self.setup()
        return len(self.word2idx)