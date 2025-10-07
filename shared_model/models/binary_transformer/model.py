"""
Binary Sequence Transformer Predictor
A transformer-based model for predicting the next binary digit (0 or 1) in a sequence.
This model treats binary sequences like language, where each position can be 0 or 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
import json
import os
from datetime import datetime
from shared_model.base.model_base import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class BinaryTransformer(BaseModel, nn.Module):
    """
    Transformer model for binary sequence prediction.
    Vocabulary: {0: '0', 1: '1', 2: '<PAD>'}
    """
    
    def __init__(
        self,
        vocab_size: int = 3,  # 0, 1, and padding token
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_seq_length: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 2)  # Only predict 0 or 1
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def create_padding_mask(self, x, pad_token=2):
        """Create mask for padded positions"""
        return (x == pad_token)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask to prevent looking at future tokens"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, src, src_key_padding_mask=None):
        """
        Forward pass
        Args:
            src: Input tensor of shape (batch_size, seq_len)
            src_key_padding_mask: Mask for padded positions
        Returns:
            Output logits of shape (batch_size, seq_len, 2)
        """
        seq_len = src.size(1)
        
        # Embeddings and positional encoding
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(src.device)
        
        # Transformer forward pass
        output = self.transformer(
            embedded,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def predict_next(self, sequence: str, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Predict the next digit in a binary sequence
        Args:
            sequence: String of 0s and 1s
            temperature: Sampling temperature (1.0 = normal, <1.0 = more confident)
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        self.eval()
        
        # Convert sequence to tensor
        tokens = [int(c) for c in sequence]
        if len(tokens) > self.max_seq_length - 1:
            tokens = tokens[-(self.max_seq_length - 1):]  # Keep last N-1 tokens
        
        # Add to batch dimension
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            # Forward pass
            logits = self.forward(input_tensor)
            
            # Get logits for the last position
            last_logits = logits[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            return str(predicted_class), confidence
    
    def generate_sequence(self, start_sequence: str = "", length: int = 10, temperature: float = 1.0) -> str:
        """
        Generate a sequence of given length
        Args:
            start_sequence: Initial sequence to start with
            length: Total length of sequence to generate
            temperature: Sampling temperature
        Returns:
            Generated binary sequence
        """
        self.eval()
        sequence = start_sequence
        
        for _ in range(length - len(start_sequence)):
            next_digit, _ = self.predict_next(sequence, temperature)
            sequence += next_digit
            
            if len(sequence) >= self.max_seq_length:
                break
        
        return sequence
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'max_seq_length': self.max_seq_length
            },
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        # Get device for loading
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)

        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.eval()
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        self.train()
    
    @classmethod
    def load_model_from_file(cls, path: str, device: str = 'auto') -> 'BinaryTransformer':
        """Load a saved model from file (class method version)"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = BinaryTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_length=config['max_seq_length']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class BinarySequenceDataset(torch.utils.data.Dataset):
    """Dataset for binary sequences"""
    
    def __init__(self, sequences: List[str], max_length: int = 100):
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
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


class BinaryTransformerTrainer:
    """Trainer class for the binary transformer model"""
    
    def __init__(
        self,
        model: BinaryTransformer,
        train_dataset: BinarySequenceDataset,
        val_dataset: Optional[BinarySequenceDataset] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = 'auto'
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_dataset:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Create padding mask (True for padded positions)
            padding_mask = (attention_mask == 0)
            
            # Forward pass
            logits = self.model(input_ids, src_key_padding_mask=padding_mask)
            
            # Reshape for loss calculation
            logits = logits.view(-1, 2)
            labels = labels.view(-1)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        if not self.val_dataset:
            return None, None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                padding_mask = (attention_mask == 0)
                
                # Forward pass
                logits = self.model(input_ids, src_key_padding_mask=padding_mask)
                
                # Reshape for loss calculation
                logits_flat = logits.view(-1, 2)
                labels_flat = labels.view(-1)
                
                # Calculate loss
                loss = self.criterion(logits_flat, labels_flat)
                total_loss += loss.item()
                
                # Calculate accuracy (only for non-ignored tokens)
                mask = labels_flat != -100
                if mask.sum() > 0:
                    predictions = torch.argmax(logits_flat[mask], dim=1)
                    correct += (predictions == labels_flat[mask]).sum().item()
                    total += mask.sum().item()
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, save_path: str = "binary_transformer.pth", print_every: int = 10):
        """Train the model"""
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate()
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Accuracy: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
        
        print("Training completed!")
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            },
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        # Get device for loading
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)

        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.eval()
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        self.train()
    
    @classmethod
    def load_model_from_file(cls, path: str, device: str = 'auto') -> 'BinaryTransformer':
        """Load a saved model from file (class method version)"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = BinaryTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_length=config['max_seq_length']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    # Example usage
    print("Binary Transformer Predictor")
    print("=" * 40)
    
    # Create some example data
    example_sequences = [
        "0101010101",
        "1100110011",
        "0011001100",
        "1010101010",
        "0000111100",
        "1111000011"
    ]
    
    # Create model
    model = BinaryTransformer(
        d_model=64,
        nhead=4,
        num_layers=3,
        max_seq_length=20
    )
    
    # Test prediction
    test_sequence = "010101"
    next_digit, confidence = model.predict_next(test_sequence)
    print(f"Sequence: {test_sequence}")
    print(f"Predicted next digit: {next_digit} (confidence: {confidence:.3f})")
