from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Iterator, Tuple

class BaseBinarySequenceDataset(Dataset):
    """Abstract base class for binary sequence datasets."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Get a sequence and target from the dataset.
        Returns:
            Tuple of (sequence, target_next_digit)
        """
        pass
    
    @abstractmethod
    def split_train_val(self, val_ratio: float = 0.2) -> Tuple['BaseBinarySequenceDataset', 'BaseBinarySequenceDataset']:
        """Split dataset into training and validation sets."""
        pass
