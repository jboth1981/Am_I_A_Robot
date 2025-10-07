"""Binary Sequence Dataset for Transformer training."""

from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import torch
import random

class BinarySequenceDataset(Dataset):
    """Dataset for binary sequence prediction."""
    
    def __init__(self, sequences: List[str], max_length: int = 100):
        """
        Args:
            sequences: List of binary sequences (e.g., ["010101", "110011"])
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.max_length = max_length
        
        # Vocabulary mapping
        self.vocab = {'0': 0, '1': 1, '<PAD>': 2}
        self.idx_to_token = {0: '0', 1: '1', 2: '<PAD>'}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence and create input/target tensors."""
        sequence = self.sequences[idx]
        
        # Convert to integers
        tokens = [int(c) for c in sequence]
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([2] * (self.max_length - len(tokens)))  # Pad with 2
        
        # Create input and target
        input_tokens = tokens[:-1]  # All but last
        target_tokens = tokens[1:]  # All but first
        
        # Convert targets to only 0/1 (ignore padding in loss)
        target_tokens = [t if t < 2 else -100 for t in target_tokens]  # -100 is ignored in CrossEntropyLoss
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'labels': torch.tensor(target_tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t < 2 else 0 for t in input_tokens], dtype=torch.long)
        }
