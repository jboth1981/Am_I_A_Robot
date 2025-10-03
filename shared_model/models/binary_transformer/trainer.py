"""Binary Transformer Trainer."""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
from shared_model.base.trainer_base import BaseTrainer
from .model import BinaryTransformer
from .dataset import BinarySequenceDataset

class BinaryTransformerTrainer(BaseTrainer):
    """Trainer for BinaryTransformer model."""
    
    def __init__(
        self,
        model: BinaryTransformer,
        train_dataset: BinarySequenceDataset,
        val_dataset: BinarySequenceDataset = None,
        learning_rate: float = 1e-4,
        device: str = 'auto'
    ):
        """
        Args:
            model: BinaryTransformer instance
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            learning_rate: Learning rate for optimizer
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Loss function (CrossEntropyLoss for classification)
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding token
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
    
    def train(self, num_epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training metrics and results
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        
        # Create data loaders
        train_loader = self.train_dataset.create_data_loader(batch_size=batch_size, shuffle=True)
        val_loader = None
        if self.val_dataset:
            val_loader = self.val_dataset.create_data_loader(batch_size=batch_size, shuffle=False)
            print(f"Validation samples: {len(self.val_dataset)}")
        
        # Training loop
        self.model.train_mode()
        
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['epochs'].append(epoch)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
            
            # Log progress
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
                if val_loss:
                    log_msg += f" | Val Loss: {val_loss:.4f}"
                print(log_msg)
        
        print("Training completed!")
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'history': self.history,
            'device': str(self.device),
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        
        for sequences, targets in train_loader:
            self.optimizer.zero_grad()
            
            # Encode sequences
            loss = 0.0
            for seq, target in zip(sequences, targets):
                # Convert target to index
                target_idx = int(target)
                
                # Get prediction
                _, confidence = self.model.predict_next(seq)
                pred_idx = int(_)
                
                # For simplicity, use cross-entropy-like loss
                loss += self.criterion(
                    torch.tensor([[confidence, 1-confidence]], device=self.device),
                    torch.tensor([target_idx], device=self.device)
                )
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval_mode()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                loss = 0.0
                for seq, target in zip(sequences, targets):
                    target_idx = int(target)
                    _, confidence = self.model.predict_next(seq)
                    pred_idx = int(_)
                    
                    loss += self.criterion(
                        torch.tensor([[confidence, 1-confidence]], device=self.device),
                        torch.tensor([target_idx], device=self.device)
                    )
                
                total_loss += loss.item()
        
        self.model.train_mode()
        return total_loss / len(val_loader)
    
    def validate(self, validation_data) -> Dict[str, float]:
        """Validate the model on validation data."""
        if isinstance(validation_data, BinarySequenceDataset):
            val_loader = validation_data.create_data_loader(batch_size=32, shuffle=False)
            val_loss = self._validate_epoch(val_loader)
            return {'val_loss': val_loss}
        else:
            raise ValueError("Validation data must be a BinarySequenceDataset")
    
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
