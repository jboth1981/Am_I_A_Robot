"""Binary Sequence Dataset for Transformer training."""

from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import torch
import random

class BinarySequenceDataset(Dataset):
    """Dataset for binary sequence prediction."""
    
    def __init__(self, sequences: List[str], targets: List[str], vocab_size: int = 3):
        """
        Args:
            sequences: List of binary sequences (e.g., ["010101", "110011"])
            targets: List of target next digits (e.g., ["0", "1"])
            vocab_size: Size of vocabulary (3 for binary: 0, 1, <PAD>)
        """
        self.sequences = sequences
        self.targets = targets
        self.vocab_size = vocab_size
        
        # Vocabulary mapping
        self.vocab = {'0': 0, '1': 1, '<PAD>': 2}
        self.idx_to_token = {0: '0', 1: '1', 2: '<PAD>'}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get a sequence and target."""
        return self.sequences[idx], self.targets[idx]
    
    def encode_sequence(self, sequence: str, max_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sequence to tensor representation.
        
        Args:
            sequence: Binary sequence string
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_tensor, attention_mask)
        """
        # Convert to token IDs
        tokens = [self.vocab.get(char, 0) for char in sequence]
        
        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]  # Take last max_length tokens
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.vocab['<PAD>'])
            attention_mask.append(0)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.bool)
    
    def create_data_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def split_train_val(self, val_ratio: float = 0.2) -> Tuple['BinarySequenceDataset', 'BinarySequenceDataset']:
        """Split dataset into training and validation sets."""
        total_size = len(self.sequences)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        # Create index shuffle
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split sequences and targets
        train_sequences = [self.sequences[i] for i in train_indices]
        train_targets = [self.targets[i] for i in train_indices]
        
        val_sequences = [self.sequences[i] for i in val_indices]
        val_targets = [self.targets[i] for i in val_indices]
        
        train_dataset = BinarySequenceDataset(train_sequences, train_targets, self.vocab_size)
        val_dataset = BinarySequenceDataset(val_sequences, val_targets, self.vocab_size)
        
        return train_dataset, val_dataset
