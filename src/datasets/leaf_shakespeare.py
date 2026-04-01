import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .adapter import DatasetAdapter

class LEAFShakespeareDataset(DatasetAdapter):
    def __init__(self, root: str = "data/leaf_shakespeare", download: bool = False):
        super().__init__(root, download, None, None)
        
        # Standard LEAF characters
        self.vocab = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.pad_char = ' '  # Use space for padding
        self.pad_idx = self.char_to_int[self.pad_char]
        
        self.seq_len = 80
        self.train_data = {}
        self.test_data = {}

    def load_datasets(self) -> None:
        """Loads all JSON files from train/ and test/ directories."""
        # Adjust paths to match your structure
        train_dir = os.path.join(self.root, 'train')
        test_dir = os.path.join(self.root, 'test')

        print(f"--- Loading LEAF Shakespeare from {self.root} ---")
        
        self.train_data = self._load_json_dir(train_dir)
        self.test_data = self._load_json_dir(test_dir)
        
        print(f"Loaded {len(self.train_data)} training users.")
        print(f"Loaded {len(self.test_data)} testing users.")

        # Create Centralized Test Set
        all_test_x, all_test_y = [], []
        for user_id, data in self.test_data.items():
            ux, uy = self._process_user_data(data['x'], data['y'])
            if len(ux) > 0:
                all_test_x.append(ux)
                all_test_y.append(uy)
            
        if all_test_x:
            combined_x = torch.cat(all_test_x)
            combined_y = torch.cat(all_test_y)
            self._test_dataset = TensorDataset(combined_x, combined_y)
            print(f"Centralized Test Set: {len(self._test_dataset)} samples.")
        else:
            print("Warning: No LEAF test data found (Check paths or JSON content).")
            self._test_dataset = TensorDataset(torch.empty(0, 80), torch.empty(0, 80))

        self._train_dataset = None 

    def _load_json_dir(self, dir_path):
        data_map = {}
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} not found.")
            return data_map
            
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        if not json_files:
            print(f"Warning: No .json files found in {dir_path}")
            
        for fname in json_files:
            path = os.path.join(dir_path, fname)
            try:
                with open(path, 'r') as f:
                    file_data = json.load(f)
                    
                # Handle 'user_data' key
                if 'user_data' in file_data:
                    for user, udata in file_data['user_data'].items():
                        data_map[user] = udata
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                
        return data_map

    def _process_user_data(self, raw_x, raw_y):
        """Converts text to indices with Padding/Truncation."""
        x_tensors = []
        y_tensors = []
        
        for seq_x, seq_y in zip(raw_x, raw_y):
            # 1. Convert to indices
            idx_x = [self.char_to_int.get(c, 0) for c in seq_x]
            idx_y = [self.char_to_int.get(c, 0) for c in seq_y]
            
            # 2. Pad or Truncate X
            if len(idx_x) < self.seq_len:
                # Pad with spaces
                idx_x += [self.pad_idx] * (self.seq_len - len(idx_x))
            elif len(idx_x) > self.seq_len:
                # Truncate
                idx_x = idx_x[:self.seq_len]

            # 3. Pad or Truncate Y
            if len(idx_y) < self.seq_len:
                idx_y += [self.pad_idx] * (self.seq_len - len(idx_y))
            elif len(idx_y) > self.seq_len:
                idx_y = idx_y[:self.seq_len]

            x_tensors.append(torch.tensor(idx_x, dtype=torch.long))
            y_tensors.append(torch.tensor(idx_y, dtype=torch.long))
            
        if not x_tensors:
             return torch.empty(0, self.seq_len, dtype=torch.long), torch.empty(0, self.seq_len, dtype=torch.long)
             
        return torch.stack(x_tensors), torch.stack(y_tensors)

    def get_client_loaders(self, num_clients: int, batch_size: int = 64, strategy: str = "natural", seed: int = 0, **kwargs) -> dict:
        self.setup()
        
        all_users = list(self.train_data.keys())
        # Sort for deterministic behavior
        all_users.sort() 
        
        total_available = len(all_users)
        
        np.random.seed(seed)
        if num_clients < total_available:
            selected_users = np.random.choice(all_users, num_clients, replace=False)
        else:
            selected_users = all_users
            if num_clients > total_available:
                print(f"Warning: Requested {num_clients} clients but dataset only has {total_available}. Using {total_available}.")

        loaders = {}
        for i, user_id in enumerate(selected_users):
            user_raw = self.train_data[user_id]
            tx, ty = self._process_user_data(user_raw['x'], user_raw['y'])
            
            if len(tx) > 0:
                ds = TensorDataset(tx, ty)
                loaders[i] = DataLoader(ds, batch_size=batch_size, shuffle=True)
            
        print(f"Generated {len(loaders)} clients (Natural Partitioning).")
        return loaders  