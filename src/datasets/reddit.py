import torch
from torch.utils.data import Dataset, TensorDataset
import os
from .adapter import DatasetAdapter
from ..defenses.const import NUM_CLASSES

class RedditDataset(DatasetAdapter):
    """
    Adapter for the Reddit dataset (LEAF benchmark).
    Expects pre-processed .pt files for efficiency, or generates synthetic data for testing.
    """
    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(root, download, None, None)
        self.vocab_size = NUM_CLASSES.get("REDDIT", 50000)
        self.seq_len = 50 # Standard sequence length for Reddit in LEAF

    def load_datasets(self) -> None:
        train_path = os.path.join(self.root, "reddit_train.pt")
        test_path = os.path.join(self.root, "reddit_test.pt")

        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading Reddit data from {train_path}...")
            self._train_dataset = torch.load(train_path)
            self._test_dataset = torch.load(test_path)
        else:
            print(f"Warning: Reddit data not found at {train_path}. Generating DUMMY data for testing.")
            self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generates random integers to simulate tokenized text."""
        # Simulate 1000 samples, sequence length 50
        train_x = torch.randint(0, self.vocab_size, (1000, self.seq_len))
        # Targets are usually the sequence shifted by one or next token. 
        # Here we simulate arbitrary targets.
        train_y = torch.randint(0, self.vocab_size, (1000, self.seq_len))
        
        test_x = torch.randint(0, self.vocab_size, (200, self.seq_len))
        test_y = torch.randint(0, self.vocab_size, (200, self.seq_len))

        self._train_dataset = TensorDataset(train_x, train_y)
        self._test_dataset = TensorDataset(test_x, test_y)