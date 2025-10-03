from abc import ABC, abstractmethod
from typing import Tuple

class BaseModel(ABC):
    """Abstract base class for all binary sequence prediction models."""
    
    @abstractmethod
    def predict_next(self, sequence: str, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Predict the next digit in a binary sequence.
        Args:
            sequence: Binary sequence (e.g., "010101")
            temperature: Sampling temperature for generation
        Returns:
            Tuple of (prediction, confidence)
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        pass
    
    @abstractmethod
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        pass
    
    @abstractmethod
    def train_mode(self) -> None:
        """Set model to training mode."""
        pass
