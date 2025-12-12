import torch
from typing import Dict, Any
from ..fl.client import BenignClient

class LeadFLClient(BenignClient):
    """
    Implements LeadFL: Client Self-Defense against Model Poisoning.
    
    Reference: Zhu et al., "LeadFL: Client Self-Defense against Model Poisoning in Federated Learning", ICML 2023.
    
    Faithful Logic:
    1. Compute Task Gradients: g_task = \nabla L_task
    2. Compute Regularization Gradients: g_reg = \nabla (alpha * ||g_task||^2)
    3. Clip ONLY g_reg to norm 'q'.
    4. Final Update: g_final = g_task + clip(g_reg, q)
    """
    def __init__(self, defense_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Regularization strength (alpha)
        self.alpha = defense_config.get('alpha', defense_config.get('mu', 0.25))
        
        # Clipping norm for the regularization term (q)
        # Paper uses q=0.2
        self.q = defense_config.get('q', defense_config.get('clipping_norm', 0.2))
        
        print(f"Initialized LeadFL Client {self.id}: alpha={self.alpha}, q={self.q}")

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        self.model.train()
        
        train_loss = 0.0
        grad_norm_avg = 0.0
        correct = 0
        total = 0
        
        run_epochs = epochs if epochs is not None else self.epochs_default

        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(run_epochs):
            if self.trainloader is None: break
            
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- 1. Compute Task Gradients ---
                outputs = self.model(inputs)
                task_loss = self.loss_fn(outputs, targets)
                
                # Retain graph for second-order derivative calculation
                task_grads = torch.autograd.grad(
                    task_loss, 
                    params, 
                    create_graph=True, 
                    retain_graph=True
                )
                
                # --- 2. Compute Regularization Gradients ---
                if self.alpha > 0.0:
                    # Squared L2 Norm of Task Gradients
                    grad_norm_sq = 0.0
                    for g in task_grads:
                        grad_norm_sq += g.pow(2).sum()
                    
                    # Reg Term = alpha * ||g||^2 (No 0.5 factor)
                    reg_term = self.alpha * grad_norm_sq
                    
                    # Compute gradients of the regularization term (Hessian-vector product)
                    # We can discard the graph here as we don't need 3rd derivatives
                    reg_grads = torch.autograd.grad(reg_term, params, retain_graph=False)
                    
                    # --- 3. Clip ONLY Regularization Gradients (q) ---
                    if self.q > 0:
                        # Calculate global norm of reg_grads
                        reg_grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in reg_grads))
                        
                        # Apply clipping factor
                        if reg_grad_norm > self.q:
                            scale = self.q / (reg_grad_norm + 1e-6)
                            reg_grads = [g * scale for g in reg_grads]
                    
                    # --- 4. Combine and Assign to p.grad ---
                    # We manually set .grad so the optimizer can perform the step
                    for p, g_task, g_reg in zip(params, task_grads, reg_grads):
                        # Detach to ensure we don't leak graph memory
                        p.grad = g_task.detach() + g_reg.detach()
                        
                    grad_norm_avg += grad_norm_sq.item()
                    
                else:
                    # Fallback to standard training if alpha=0
                    for p, g_task in zip(params, task_grads):
                        p.grad = g_task.detach()

                self.optimizer.step()

                # --- 5. Metrics ---
                train_loss += task_loss.item() 
                if outputs.ndim == 3: # NLP
                     _, predicted = torch.max(outputs.data, dim=1) 
                     flat_targets = targets.reshape(-1)
                     flat_pred = predicted.reshape(-1)
                     total += flat_targets.numel()
                     correct += (flat_pred == flat_targets).sum().item()
                else: # Image
                     _, predicted = torch.max(outputs.data, 1)
                     total += targets.size(0)
                     correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        total_steps = (len(self.trainloader) if self.trainloader else 1) * run_epochs
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {
                'loss': train_loss / total_steps if total_steps > 0 else 0.0, 
                'accuracy': correct / total if total > 0 else 0.0,
                'avg_grad_norm_sq': grad_norm_avg / total_steps if total_steps > 0 else 0.0
            },
            'round_idx': round_idx
        }