import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..fl.client import BenignClient

class LeadFLClient(BenignClient):
    """
    Implements LeadFL: Client Self-Defense against Model Poisoning.
    
    Reference: Zhu et al., "LeadFL: Client Self-Defense against Model Poisoning in Federated Learning", ICML 2023.
    
    The defense adds a regularization term to the local loss function to "nullify the Hessian" 
    (minimize curvature). The faithful implementation uses the Squared Gradient Norm as the 
    computationally efficient proxy for this objective.
    
    Loss = Task_Loss + (mu / 2) * || \nabla Task_Loss ||^2
    """
    def __init__(self, defense_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 'mu' is the regularization strength (Default ~0.05 - 0.1 in paper)
        self.mu = defense_config.get('mu', 0.1) 
        print(f"Initialized LeadFL Client {self.id} with mu={self.mu}")

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        self.model.train()
        self._create_optimizer()
        
        train_loss = 0.0
        grad_norm_avg = 0.0
        correct = 0
        total = 0
        
        run_epochs = epochs if epochs is not None else self.epochs_default

        for _ in range(run_epochs):
            if self.trainloader is None: break
            
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 1. Forward Pass
                outputs = self.model(inputs)
                task_loss = self.loss_fn(outputs, targets)
                
                # 2. LeadFL Regularization
                if self.mu > 0.0:
                    # Filter for trainable parameters
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    
                    # Compute gradients of the task loss w.r.t parameters
                    # create_graph=True is ESSENTIAL to allow backprop through the gradient norm
                    grads = torch.autograd.grad(
                        task_loss, 
                        params, 
                        create_graph=True, 
                        retain_graph=True,
                        only_inputs=True
                    )
                    
                    # Compute Squared L2 Norm of the gradient vector
                    grad_norm_sq = 0.0
                    for grad in grads:
                        grad_norm_sq += grad.pow(2).sum()
                    
                    # Total Loss with regularization
                    total_loss = task_loss + 0.5 * self.mu * grad_norm_sq
                    
                    # Logging
                    grad_norm_avg += grad_norm_sq.item()
                else:
                    total_loss = task_loss

                # 3. Backward & Step
                total_loss.backward()
                self.optimizer.step()

                # Metrics
                train_loss += task_loss.item() 
                if outputs.ndim == 3: # Handle sequence output (NLP)
                     # Flatten for accuracy: [Batch*Seq, Vocab]
                     outputs_flat = outputs.transpose(1, 2).reshape(-1, outputs.shape[1])
                     targets_flat = targets.reshape(-1)
                     _, predicted = torch.max(outputs_flat.data, 1)
                     total += targets_flat.size(0)
                     correct += (predicted == targets_flat).sum().item()
                else:
                     _, predicted = torch.max(outputs.data, 1)
                     total += targets.size(0)
                     correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        total_steps = num_batches * run_epochs
        
        metrics = {
            'loss': train_loss / total_steps if total_steps > 0 else 0.0, 
            'accuracy': correct / total if total > 0 else 0.0,
            'avg_grad_norm_sq': grad_norm_avg / total_steps if total_steps > 0 else 0.0
        }
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }