"""Binary Sequence Dataset for Transformer training."""

from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import torch
import random

# Vocabulary constants
START_TOKEN = 2
PAD_TOKEN = 3
VOCAB_SIZE = 4  # 0, 1, START_TOKEN, PAD_TOKEN

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
        
        # Vocabulary mapping - using integers directly
        self.START_TOKEN = START_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.vocab_size = VOCAB_SIZE
    
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
            tokens.extend([self.PAD_TOKEN] * (self.max_length - len(tokens)))  # Pad with PAD_TOKEN
        
        # Create input and target with START token
        # Input: [START, digit1, digit2, ...] -> Target: [digit1, digit2, digit3, ...]
        input_tokens = [self.START_TOKEN] + tokens[:-1]  # START + all but last
        target_tokens = tokens  # All digits including first
        
        # Convert targets to only 0/1 (ignore padding in loss)
        target_tokens = [t if t < self.PAD_TOKEN else -100 for t in target_tokens]  # -100 is ignored in CrossEntropyLoss
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'labels': torch.tensor(target_tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t < self.PAD_TOKEN else 0 for t in input_tokens], dtype=torch.long)
        }
