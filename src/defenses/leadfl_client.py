import torch
from typing import Dict, Any
from ..fl.client import BenignClient, move_optimizer_state

import torch
from typing import Dict, Any
# Ensure you import move_optimizer_state from where BenignClient is defined
# If you cannot import it, copy the definition from BenignClient file to here.
from ..fl.client import BenignClient, move_optimizer_state

class LeadFLClient(BenignClient):
    """
    Implements LeadFL: Client Self-Defense against Model Poisoning.
    """
    def __init__(self, defense_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = defense_config.get('alpha', defense_config.get('mu', 0.25))
        self.q = defense_config.get('q', defense_config.get('clipping_norm', 0.2))
        # print(f"Initialized LeadFL Client {self.id}: alpha={self.alpha}, q={self.q}")

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        # --- FIX 1: Move Model to Device ---
        self.model.to(self.device)
        
        # --- FIX 2: Ensure Optimizer Exists ---
        self._ensure_optimizer()
        
        # --- FIX 3: Move Optimizer State Correctly (No .to() method) ---
        move_optimizer_state(self.optimizer, self.device)
        
        self.model.train()

        train_loss = 0.0
        grad_norm_avg = 0.0
        correct = 0
        total = 0
        
        run_epochs = epochs if epochs is not None else self.epochs_default

        # Filter params once
        params = [p for p in self.model.parameters() if p.requires_grad]

        for _ in range(run_epochs):
            if self.trainloader is None: break
            
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- 1. Compute Task Gradients ---
                outputs = self.model(inputs)
                task_loss = self.loss_fn(outputs, targets)
                
                # Retain graph for second-order derivative (Hessian-vector product)
                task_grads = torch.autograd.grad(
                    task_loss, 
                    params, 
                    create_graph=True, 
                    retain_graph=True
                )
                
                # --- 2. Compute Regularization Gradients ---
                if self.alpha > 0.0:
                    grad_norm_sq = 0.0
                    for g in task_grads:
                        grad_norm_sq += g.pow(2).sum()
                    
                    reg_term = self.alpha * grad_norm_sq
                    
                    # Compute gradients of reg_term; no graph needed after this
                    reg_grads = torch.autograd.grad(reg_term, params, retain_graph=False)
                    
                    # --- 3. Clip ONLY Regularization Gradients (q) ---
                    if self.q > 0:
                        reg_grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in reg_grads))
                        if reg_grad_norm > self.q:
                            scale = self.q / (reg_grad_norm + 1e-6)
                            reg_grads = [g * scale for g in reg_grads]
                    
                    # --- 4. Combine and Assign ---
                    for p, g_task, g_reg in zip(params, task_grads, reg_grads):
                        # Detach to free graph memory
                        p.grad = g_task.detach() + g_reg.detach()
                        
                    grad_norm_avg += grad_norm_sq.item()
                    
                else:
                    # Fallback (Alpha=0)
                    for p, g_task in zip(params, task_grads):
                        p.grad = g_task.detach()

                self.optimizer.step()

                # --- Metrics ---
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

        # --- FIX 4: Cleanup to prevent Memory Leaks / GPU OOM ---
        self.model.to("cpu")
        move_optimizer_state(self.optimizer, "cpu")
        torch.cuda.empty_cache()

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