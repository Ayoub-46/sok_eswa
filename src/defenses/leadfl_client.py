import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..fl.client import BenignClient

class LeadFLClient(BenignClient):
    """
    Implements LeadFL: Client Self-Defense against Model Poisoning.
    
    Reference: Zhu et al., "LeadFL: Client Self-Defense against Model Poisoning in Federated Learning", ICML 2023.
    Repository: https://github.com/chaoyitud/LeadFL
    
    Mechanism:
    Adds a regularization term to the local loss function:
        Loss_total = Loss_task + (mu / 2) * || \nabla Loss_task ||^2
        
    Minimizing the gradient norm ||g||^2 implicitly penalizes the curvature (Hessian),
    guiding the optimization to flatter, more robust minima.
    """
    def __init__(self, defense_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 'mu' is the regularization coefficient. 
        # Paper typically uses 0.05 - 0.1. 
        # NOTE: For pre-trained models, start lower (e.g., 0.01) to prevent weight explosion.
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
                
                # 1. Forward Pass & Task Loss
                outputs = self.model(inputs)
                task_loss = self.loss_fn(outputs, targets)
                
                # 2. LeadFL Regularization (Gradient Penalty)
                if self.mu > 0.0:
                    # Filter for trainable parameters
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    
                    # Compute first-order gradients
                    # create_graph=True is REQUIRED to backpropagate through the gradients themselves
                    grads = torch.autograd.grad(
                        task_loss, 
                        params, 
                        create_graph=True, 
                        retain_graph=True,
                        only_inputs=True
                    )
                    
                    # Compute Squared L2 Norm: ||g||^2
                    grad_norm_sq = 0.0
                    for grad in grads:
                        grad_norm_sq += grad.pow(2).sum()
                    
                    # Total Loss = Task Loss + (mu/2) * ||g||^2
                    # Differentiating this term effectively computes Hessian-Vector Products
                    total_loss = task_loss + 0.5 * self.mu * grad_norm_sq
                    
                    grad_norm_avg += grad_norm_sq.item()
                else:
                    total_loss = task_loss

                # 3. Backward Pass
                total_loss.backward()
                
                # --- Stability Fix ---
                # Hessian-based updates can be large, especially for pre-trained models in sharp minima.
                # Clipping prevents numerical explosion (NaNs).
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()

                # 4. Metrics (Dynamic for Image vs NLP)
                train_loss += task_loss.item() 
                
                if outputs.ndim == 3: # NLP Sequence: [Batch, Class, Seq_Len]
                     # Max over class dimension (dim=1)
                     _, predicted = torch.max(outputs.data, dim=1) 
                     
                     # Flatten for correct token-level accuracy
                     flat_targets = targets.reshape(-1)
                     flat_predicted = predicted.reshape(-1)
                     
                     current_total = flat_targets.numel()
                     correct += (flat_predicted == flat_targets).sum().item()
                     
                else: # Image/Classification: [Batch, Class]
                     _, predicted = torch.max(outputs.data, 1)
                     current_total = targets.size(0)
                     correct += (predicted == targets).sum().item()
                     
                total += current_total

        if self.scheduler:
            self.scheduler.step()

        # Normalize metrics
        # If 'total' is 0 (empty loader), handle safely
        num_batches = len(self.trainloader) if self.trainloader else 1
        total_steps = num_batches * run_epochs
        
        # Avoid division by zero
        avg_loss = train_loss / total_steps if total_steps > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0
        avg_grad_norm = grad_norm_avg / total_steps if total_steps > 0 else 0.0
        
        metrics = {
            'loss': avg_loss, 
            'accuracy': avg_acc,
            'avg_grad_norm_sq': avg_grad_norm
        }
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }