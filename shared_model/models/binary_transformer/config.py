"""Configuration for BinaryTransformer model."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BinaryTransformerConfig:
    """Configuration class for BinaryTransformer model."""
    
    # Model architecture
    vocab_size: int = 3  # 0, 1, and padding token
    d_model: int = 128  # Model dimension
    nhead: int = 8      # Number of attention heads
    num_layers: int = 6 # Number of transformer layers
    dim_feedforward: int = 512  # Feedforward dimension
    max_seq_length: int = 100   # Maximum sequence length
    dropout: float = 0.1        # Dropout rate
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    val_ratio: float = 0.2
    
    # Data preprocessing
    min_length: int = 10
    max_length: int = 100
    augment: int = 1
    
    # Device
    device: str = 'auto'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'val_ratio': self.val_ratio,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'augment': self.augment,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BinaryTransformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def default(cls) -> 'BinaryTransformerConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def small_model(cls) -> 'BinaryTransformerConfig':
        """Create configuration for smaller, faster model."""
        return cls(
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            max_seq_length=50,
            batch_size=16,
            num_epochs=50
        )
    
    @classmethod
    def large_model(cls) -> 'BinaryTransformerConfig':
        """Create configuration for larger, more powerful model."""
        return cls(
            d_model=256,
            nhead=16,
            num_layers=12,
            dim_feedforward=1024,
            max_seq_length=200,
            batch_size=64,
            num_epochs=200
        )
