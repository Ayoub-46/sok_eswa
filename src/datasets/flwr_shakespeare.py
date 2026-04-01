import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from .adapter import DatasetAdapter

class FlwrShakespeareDataset(DatasetAdapter):
    """
    Adapter for Hugging Face 'flwrlabs/shakespeare'.
    Manually implements the standard 80/20 LEAF split per user.
    """
    def __init__(self, root: str = "data/flwr_shakespeare", download: bool = True):
        super().__init__(root, download, None, None)
        
        # Standard Q1 Vocabulary (80 chars + 1 PAD)
        self.leaf_chars = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.pad_token = "<PAD>"
        self.vocab = [self.pad_token] + list(self.leaf_chars)
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        
        self.seq_len = 80
        
        # Caches
        self.raw_dataset = None
        self.train_partitions = {} # {user_id: [indices]}
        self.test_partitions = {}  # {user_id: [indices]}

    def load_datasets(self) -> None:
        print("--- Loading flwrlabs/shakespeare from Hugging Face ---")
        # Load the single 'train' split
        full_dataset = load_dataset("flwrlabs/shakespeare")
        self.raw_dataset = full_dataset['train']
        
        print("Partitioning data by character_id...")
        all_user_indices = self._partition_by_char(self.raw_dataset)
        
        # Perform Standard LEAF Split (80% Train, 20% Test PER CLIENT)
        print("Performing 80/20 split per client...")
        for user, indices in all_user_indices.items():
            n_samples = len(indices)
            if n_samples < 2: continue # Skip users with insufficient data
            
            # Strict 80% split
            split_idx = int(n_samples * 0.8)
            
            # Ensure at least 1 sample in train if possible, but 80/20 rule is priority
            if split_idx == 0 and n_samples > 0: split_idx = 1
                
            self.train_partitions[user] = indices[:split_idx]
            self.test_partitions[user] = indices[split_idx:]
            
        print(f"Loaded {len(self.train_partitions)} clients.")

        # Create Centralized Test Set from ALL test partitions
        all_test_indices = []
        for user in self.test_partitions:
            all_test_indices.extend(self.test_partitions[user])
            
        # We define _test_dataset as a Subset of the main raw dataset
        self._test_dataset = Subset(self.raw_dataset, all_test_indices)
        print(f"Centralized Test Set: {len(self._test_dataset)} samples.")

        # We don't store _train_dataset to save RAM (clients load on demand)
        self._train_dataset = None 

    def _partition_by_char(self, hf_split):
        """Groups indices by character_id using Pandas for speed."""
        try:
            # Fast grouping
            df = hf_split.select_columns(["character_id"]).to_pandas()
            return df.groupby("character_id").indices.to_dict()
        except Exception as e:
            print(f"Warning: Pandas grouping failed ({e}). Falling back to slow loop.")
            indices = {}
            for i, item in enumerate(hf_split):
                char = item['character_id']
                if char not in indices: indices[char] = []
                indices[char].append(i)
            return indices

    def _collate_fn(self, batch):
        """
        Converts batch of raw items to Tensors.
        Handles Many-to-One format (Input: 80 chars, Target: 1 char).
        """
        x_batch = []
        y_batch = []
        
        default_idx = self.char_to_int.get(' ', 1)
        
        for item in batch:
            # x is string of 80 chars
            idx_x = [self.char_to_int.get(c, default_idx) for c in item['x']]
            
            # y is single char string
            idx_y = self.char_to_int.get(item['y'], default_idx)
            
            x_batch.append(torch.tensor(idx_x, dtype=torch.long))
            y_batch.append(torch.tensor(idx_y, dtype=torch.long))
            
        return torch.stack(x_batch), torch.stack(y_batch)

    def get_client_loaders(self, num_clients: int, batch_size: int = 64, seed: int = 0, **kwargs) -> dict:
        self.setup()
        all_users = sorted(list(self.train_partitions.keys()))
        
        np.random.seed(seed)
        selected_users = all_users
        if num_clients < len(all_users):
            selected_users = np.random.choice(all_users, num_clients, replace=False)
            
        loaders = {}
        for i, user_id in enumerate(selected_users):
            indices = self.train_partitions[user_id]
            if len(indices) == 0: continue
            
            subset = Subset(self.raw_dataset, indices)
            loaders[i] = DataLoader(
                subset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn
            )
            
        print(f"Generated {len(loaders)} clients from Hugging Face data.")
        return loaders

    def get_test_loader(self, batch_size: int = 256) -> DataLoader:
        self.setup()
        return DataLoader(
            self._test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn
        )