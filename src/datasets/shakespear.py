import torch
import os
from torch.utils.data import TensorDataset
from .adapter import DatasetAdapter
from ..defenses.const import NUM_CLASSES

class ShakespeareDataset(DatasetAdapter):
    """
    Adapter for the Shakespeare dataset (LEAF benchmark).
    Task: Next-Character Prediction.
    """
    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(root, download, None, None)
        self.vocab_size = NUM_CLASSES["SHAKESPEARE"]
        self.seq_len = 80 # Standard sequence length for Shakespeare in LEAF

    def load_datasets(self) -> None:
        train_path = os.path.join(self.root, "shakespeare_train.pt")
        test_path = os.path.join(self.root, "shakespeare_test.pt")

        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading Shakespeare data from {train_path}...")
            self._train_dataset = torch.load(train_path)
            self._test_dataset = torch.load(test_path)
        else:
            print(f"Warning: Shakespeare data not found at {train_path}. Generating DUMMY data.")
            self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generates random integer sequences for testing."""
        # 1000 samples, sequence length 80, values 0-79
        train_x = torch.randint(0, self.vocab_size, (1000, self.seq_len))
        train_y = torch.randint(0, self.vocab_size, (1000, self.seq_len))
        
        test_x = torch.randint(0, self.vocab_size, (200, self.seq_len))
        test_y = torch.randint(0, self.vocab_size, (200, self.seq_len))

        self._train_dataset = TensorDataset(train_x, train_y)
        self._test_dataset = TensorDataset(test_x, test_y)