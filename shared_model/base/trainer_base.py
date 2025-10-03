from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""
    
    @abstractmethod
    def train(self, num_epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        Returns:
            Training metrics and results
        """
        pass
    
    @abstractmethod
    def validate(self, validation_data: Any) -> Dict[str, float]:
        """Validate the model on validation data."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        pass
