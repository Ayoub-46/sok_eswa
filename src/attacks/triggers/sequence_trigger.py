import torch
from typing import Union, Tuple
from .base import BaseTrigger

Tensor = torch.Tensor

class SequenceTrigger(BaseTrigger):
    """
    A trigger designed for sequence data (NLP tasks), which replaces a slice
    of the input token sequence with a predefined token pattern.
    
    The 'position' and 'size' are single integers for 1D sequence data.
    'pattern' is a 1D tensor of token IDs.
    'alpha' is effectively ignored for discrete token replacement.
    """
    def __init__(self, 
                 start_index: int, 
                 trigger_tokens: Union[int, list, torch.Tensor], 
                 # BaseTrigger compatibility
                 position: Union[int, Tuple[int]] = None,
                 size: Union[int, Tuple[int]] = None,
                 pattern: Union[int, list, torch.Tensor] = None,
                 alpha: float = 1.0):
        
        # --- NLP-Specific Init ---
        if not isinstance(trigger_tokens, torch.Tensor):
            trigger_tokens = torch.tensor(trigger_tokens, dtype=torch.long)
            
        self.start_index = start_index
        self.trigger_tokens = trigger_tokens
        self.trigger_len = len(trigger_tokens)
        
        # --- BaseTrigger Init for compatibility ---
        if position is None: position = start_index 
        if size is None: size = self.trigger_len
        if pattern is None: pattern = trigger_tokens
            
        super().__init__(position, size, pattern, alpha)


    def apply(self, sequence: Tensor) -> Tensor:
        """Applies the trigger to a given sequence tensor (1D)."""
        if sequence.dim() != 1:
            # Handle the case where the input might be [C, H, W] for a CNN/Image 
            # or has a channel dimension. For sequence, we expect [L] or [L, C] for embedded.
            # Assuming [L] for raw tokens.
            if sequence.dim() == 2 and sequence.shape[1] == 1:
                sequence = sequence.squeeze(-1) # If shape is [L, 1]
            elif sequence.dim() > 1:
                # Log a warning or raise error if input shape is unexpected
                print(f"Warning: SequenceTrigger received tensor with unexpected shape {sequence.shape}. Expecting 1D.")
                return sequence
                
        # Ensure sequence is on the correct device as the trigger_tokens
        device = sequence.device
        trigger = self.trigger_tokens.to(device)
        
        L = sequence.shape[0]
        start = self.start_index
        end = min(start + self.trigger_len, L) # Ensure end does not exceed sequence length
        
        if end > start:
            # Replace the slice of the sequence with the trigger tokens
            sequence[start:end] = trigger[:(end - start)]
            
        return sequence