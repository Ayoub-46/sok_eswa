from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import copy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseServer(ABC):
    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass

class FedAvgAggregator(BaseServer):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, *args, **kwargs):
        self.device = device if device is not None else torch.device("cpu")
        self.model = cp.deepcopy(model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.testloader = testloader  
        self.received_updates: Dict[int, Dict[str, Any]] = {}

    def load_testdata(self, testloader):
        self.testloader = testloader

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in params.items()})

    def receive_update(self, client_id: int, params: Dict[str, torch.Tensor], length: int) -> None:
        params_cpu = {k: v.cpu().clone().float() for k, v in params.items()}
        self.received_updates[client_id] = {
            'params': params_cpu,
            'length': int(length)
        }

    def evaluate(self, valloader=None) -> Dict[str, object]:
        valloader = valloader or self.testloader
        self.model.eval()

        if valloader is None:
            return {'num_samples': 0, 'metrics': {'loss': float('nan'), 'main_accuracy': float('nan')}}
        
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        
        # Define Binary Loss separately just in case (for output_dim=1)
        bce_loss = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in valloader:
                # 1. Unpack Batch (Handle optional lengths)
                if len(batch) == 3:
                    inputs, targets, lengths = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # Move lengths to device for "Manual Indexing" trick
                    lengths = lengths.to(self.device) 
                else:
                    # Fallback for datasets without lengths (e.g. Shakespeare)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    lengths = None

                # 2. Forward Pass (Pass lengths if available)
                # Ensure your Model's forward() signature accepts lengths!
                if lengths is not None:
                    outputs = self.model(inputs, lengths)
                else:
                    outputs = self.model(inputs)
                

                # 3. Calculate Metrics
                # Case A: NLP Sequence (Shakespeare/Next Character) [Batch, Seq, Vocab]
                if outputs.ndim == 3: 
                    # print("Evaluating NLP Sequence Model") 
                    loss_sum += nn.functional.cross_entropy(outputs, targets, ignore_index=0).item()
                    _, preds = torch.max(outputs.data, 1)
                    mask = (targets != 0)
                    correct += (preds == targets)[mask].sum().item()
                    total += mask.sum().item()
                    
                # Case B: Classification (Sentiment140) [Batch, Classes]
                else:
                    if outputs.size(1) == 1:
                        # Binary Classification Logic (Sigmoid)
                        outputs = outputs.squeeze(1) # [Batch, 1] -> [Batch]
                        
                        loss_sum += bce_loss(outputs, targets.float()).item()
                        
                        # Threshold at 0.5
                        preds = (torch.sigmoid(outputs) > 0.5).long()
                        correct += (preds == targets).sum().item()
                    else:
                        # Multi-class Logic (Softmax/CrossEntropy)
                        # Assumes self.loss_fn is CrossEntropyLoss
                        loss_sum += self.loss_fn(outputs, targets).item()
                        _, preds = torch.max(outputs.data, 1)
                        correct += (preds == targets).sum().item()
                    
                    total += targets.size(0)
                
                iters += 1
                
        avg_loss = (loss_sum / iters) if iters > 0 else float('nan')
        accuracy = (correct / total) if total > 0 else float('nan')
        
        return {'num_samples': total, 'metrics': {'loss': avg_loss, 'main_accuracy': accuracy}}
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_updates) > 0, "No client updates received."
        total_samples = sum(data['length'] for data in self.received_updates.values())

        averaged = {}
        first_client_data = next(iter(self.received_updates.values()))
        first_params = first_client_data['params']

        for k in first_params.keys():
            acc = torch.zeros_like(first_params[k], dtype=torch.float32)
            for client_data in self.received_updates.values():
                client_params = client_data['params']
                client_len = client_data['length']
                weight = float(client_len) / float(total_samples) if total_samples > 0 else 0.0
                if k in client_params:
                   acc += client_params[k] * weight
            averaged[k] = acc

        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_updates = {}

        return {k: v.cpu().clone() for k, v in averaged.items()}

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)


class FedOptAggregator(FedAvgAggregator):
    """
    Implements Federated Optimization (FedAdam, FedYogi, FedAdagrad).
    """
    def __init__(self, model: torch.nn.Module, testloader=None, device=None, 
                 opt_method='adam', server_lr=0.01, betas=(0.9, 0.99), tau=1e-3, **kwargs):
        super().__init__(model, testloader, device, **kwargs)
        
        self.opt_method = opt_method.lower()
        self.server_lr = server_lr
        self.betas = betas
        self.tau = tau
        
        # Initialize Momentums (m) and Velocities (v) for server optimizer
        self.m_t = {k: torch.zeros_like(p) for k, p in self.model.named_parameters()}
        self.v_t = {k: torch.zeros_like(p) + tau**2 for k, p in self.model.named_parameters()}
        

    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_updates) > 0, "No client updates received."
        
        # 1. Calculate Standard Weighted Average (Pseudogradient)
        total_samples = sum(data['length'] for data in self.received_updates.values())
        weighted_avg = {}
        
        first_key = next(iter(self.received_updates.keys()))
        param_keys = self.received_updates[first_key]['params'].keys()

        for k in param_keys:
            acc = torch.zeros_like(self.received_updates[first_key]['params'][k])
            for client_data in self.received_updates.values():
                weight = client_data['length'] / total_samples
                acc += client_data['params'][k] * weight
            weighted_avg[k] = acc.to(self.device)

        # 2. Compute Delta (Update Direction)
        current_params = {k: p for k, p in self.model.named_parameters()}
        updates = {}
        
        for k, new_p in weighted_avg.items():
            if k in current_params:
                updates[k] = new_p - current_params[k].data
        
        # 3. Apply Server Optimizer Step
        self._server_opt_step(current_params, updates)
        
        # 4. Cleanup
        self.received_updates = {}
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def _server_opt_step(self, params, pseudo_grads):
        beta1, beta2 = self.betas
        
        for k, grad in pseudo_grads.items():
            if k not in self.m_t: continue 
            
            # --- Momentum ---
            self.m_t[k] = beta1 * self.m_t[k] + (1 - beta1) * grad
            
            # --- Velocity ---
            grad_sq = grad**2
            if self.opt_method == 'adam':
                self.v_t[k] = beta2 * self.v_t[k] + (1 - beta2) * grad_sq
            elif self.opt_method == 'yogi':
                diff = self.v_t[k] - grad_sq
                self.v_t[k] = self.v_t[k] - (1 - beta2) * torch.sign(diff) * grad_sq
            elif self.opt_method == 'adagrad':
                self.v_t[k] = self.v_t[k] + grad_sq

            # --- Update ---
            step = self.server_lr * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
            params[k].data.add_(step)