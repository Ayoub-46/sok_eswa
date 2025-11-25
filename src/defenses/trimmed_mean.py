import torch
from typing import Dict
from ..fl.server import FedAvgAggregator

class TrimmedMeanServer(FedAvgAggregator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = kwargs.get('defense_cfg', {}).get('beta', 0.1) # Trim fraction (e.g., 10%)

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_updates: return self.get_params()
        
        # Stack updates: {param_name: Tensor[num_clients, *param_shape]}
        stacked_updates = {}
        client_ids = list(self.received_updates.keys())
        first_id = client_ids[0]
        
        for name in self.received_updates[first_id]['params'].keys():
            # Stack tensors from all clients for this parameter
            stacked = torch.stack([self.received_updates[cid]['params'][name] for cid in client_ids])
            stacked_updates[name] = stacked

        num_clients = len(client_ids)
        num_trim = int(self.beta * num_clients)
        
        averaged_params = {}
        for name, tensor_stack in stacked_updates.items():
            if 'num_batches_tracked' in name:
                averaged_params[name] = tensor_stack[0] # Aggregation logic for buffers
                continue
            
            # Sort along the client dimension (dim=0)
            sorted_stack, _ = torch.sort(tensor_stack, dim=0)
            
            # Trim the top and bottom k updates
            if 2 * num_trim < num_clients:
                trimmed_stack = sorted_stack[num_trim : num_clients - num_trim]
            else:
                trimmed_stack = sorted_stack # Fallback if beta is too high
            
            averaged_params[name] = torch.mean(trimmed_stack, dim=0)

        # Update model
        self.set_params({k: v.to(self.device) for k, v in averaged_params.items()})
        self.received_updates = {}
        return averaged_params


class MedianServer(FedAvgAggregator):
    """Coordinate-wise Median Aggregation"""
    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_updates: return self.get_params()
        
        client_ids = list(self.received_updates.keys())
        
        averaged_params = {}
        # Iterate over one client's keys to get parameter names
        for name in self.received_updates[client_ids[0]]['params'].keys():
            # Stack all client updates for this parameter: (Num_Clients, Param_Shape)
            stacked = torch.stack([self.received_updates[cid]['params'][name] for cid in client_ids])
            
            # Compute median along dimension 0 (clients)
            # torch.median returns (values, indices)
            median_val, _ = torch.median(stacked, dim=0)
            averaged_params[name] = median_val

        self.set_params({k: v.to(self.device) for k, v in averaged_params.items()})
        self.received_updates = {}
        return averaged_params