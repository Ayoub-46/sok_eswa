from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy as cp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def local_evaluate(self) -> Dict[str, Any]:
        pass

class BenignClient(BaseClient):
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        optimizer: str = 'sgd',
        epochs: int = 1,
        device: Optional[torch.device] = None,
        ignore_index: int = -100 
    ):
        self.id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device is not None else torch.device("cpu")
        self.epochs_default = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self._model = model.to("cpu")
        self.dataset_len = len(trainloader.dataset) if trainloader is not None else 0
        
        self.optimizer = None
        self.scheduler = None
        self.optimizer_name = optimizer
        # self._create_optimizer(self.optimizer_name)
        
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def _create_optimizer(self, optimizer: str) -> None:
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = None
        else:  # Default to SGD
            self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

    def get_id(self) -> int:
        return self.id

    def num_samples(self) -> int:
        return self.dataset_len

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(params)
        
        
    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        self.model.to(self.device)
        self._create_optimizer(self.optimizer_name)

        self.model.train()
        train_loss, correct, total = 0.0, 0, 0
        run_epochs = epochs if epochs is not None else self.epochs_default

        for _ in range(run_epochs):
            if self.trainloader is None: break
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Handle Hugging Face Format (Many-to-One)
                if targets.ndim == 1 and outputs.ndim == 3:
                    outputs = outputs[:, :, -1] 

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                # Calculate Accuracy
                if outputs.ndim == 3: # NLP Sequence (Many-to-Many)
                     _, predicted = torch.max(outputs.data, 1)
                     mask = (targets != self.ignore_index)
                     correct += (predicted == targets)[mask].sum().item()
                     total += mask.sum().item()
                else: # Classification / Many-to-One
                     _, predicted = torch.max(outputs.data, 1)
                     correct += (predicted == targets).sum().item()
                     total += targets.size(0)

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        avg_loss = train_loss / (num_batches * run_epochs) if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        self.model.to("cpu")
        self.optimizer = None
        self.scheduler = None
        torch.cuda.empty_cache()

        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }

    def local_evaluate(self) -> Dict[str, Any]:
        self.model.to(self.device)
        self.model.eval()
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        valloader = self.testloader or self.trainloader
        
        if valloader is None:
            return {'metrics': {'loss': float('nan'), 'accuracy': float('nan')}}

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                if targets.ndim == 1 and outputs.ndim == 3:
                    outputs = outputs[:, :, -1]

                loss_sum += self.loss_fn(outputs, targets).item()
                iters += 1
                
                if outputs.ndim == 3: 
                     _, predicted = torch.max(outputs.data, 1)
                     mask = (targets != self.ignore_index)
                     correct += (predicted == targets)[mask].sum().item()
                     total += mask.sum().item()
                else: 
                     _, predicted = torch.max(outputs.data, 1)
                     correct += (predicted == targets).sum().item()
                     total += targets.size(0)
                
        loss_avg = (loss_sum / iters) if iters > 0 else float('nan')
        acc = (correct / total) if total > 0 else float('nan')
        
        self.model.to("cpu")
        
        return {
            'client_id': self.get_id(), 
            'num_samples': total, 
            'metrics': {'loss': loss_avg, 'accuracy': acc}
        }