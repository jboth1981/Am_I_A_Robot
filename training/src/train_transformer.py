#!/usr/bin/env python3
"""
Training Script for Binary Sequence Transformer
This script trains a transformer model on binary sequences extracted from the database.
"""

import os
import sys
import json
import argparse
from typing import List, Optional
import torch
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_model.config.model_registry import get_model_class, get_trainer_class, get_dataset_class
from shared_model.models.binary_transformer.dataset import BinarySequenceDataset
from .data_extractor import DataExtractor


def load_sequences_from_database(
    database_url: Optional[str] = None,
    min_unpredictability: float = 0.0,
    method_filter: Optional[str] = None
) -> List[str]:
    """Load sequences directly from database"""
    extractor = DataExtractor(database_url)
    
    try:
        if method_filter:
            sequences = extractor.get_sequences_by_method(method_filter)
        else:
            sequences = extractor.get_sequences_by_performance(min_unpredictability)
        
        # Show database stats
        stats = extractor.get_database_stats()
        print(f"Database contains {stats['total_submissions']} total submissions")
        
        return sequences
    finally:
        extractor.close()


def load_sequences_from_file(file_path: str) -> List[str]:
    """Load sequences from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    if 'sequences' in data:
        # Format from data_extractor.py
        sequences = [item['sequence'] for item in data['sequences']]
    else:
        # Assume it's a simple list
        sequences = data
    
    print(f"Loaded {len(sequences)} sequences from {file_path}")
    return sequences


def filter_sequences(sequences: List[str], min_length: int = 10, max_length: int = 100) -> List[str]:
    """Filter sequences by length"""
    filtered = [seq for seq in sequences if min_length <= len(seq) <= max_length]
    print(f"Filtered {len(sequences)} -> {len(filtered)} sequences (length {min_length}-{max_length})")
    return filtered


def duplicate_sequences(sequences: List[str], duplicate_factor: int = 1) -> List[str]:
    """
    Simple duplication: repeat each sequence N times
    This preserves original human behavior patterns without introducing artificial variations
    """
    duplicated = []
    for seq in sequences:
        for _ in range(duplicate_factor):
            duplicated.append(seq)
    
    print(f"Duplicated {len(sequences)} -> {len(duplicated)} sequences ({duplicate_factor}x)")
    return duplicated


def augment_sequences(sequences: List[str], augment_factor: int = 15) -> List[str]:
    """
    Aggressively augment sequences by creating many variations
    - Reverse sequences
    - Take subsequences of different lengths
    - Create pattern variations (flip every other, every 2nd, etc.)
    - Add synthetic alternating patterns
    - Create mathematical sequences
    - Add noise variations
    """
    augmented = sequences.copy()
    
    for seq in sequences:
        # Add reversed sequence
        if len(augmented) < len(sequences) * augment_factor:
            augmented.append(seq[::-1])
        
        # Add pattern variations (flip every other digit)
        if len(augmented) < len(sequences) * augment_factor:
            flipped = ''.join('1' if c == '0' else '0' for c in seq[::2]) + seq[1::2]
            if len(flipped) >= 10:
                augmented.append(flipped)
        
        # Add pattern variations (flip every 2nd digit)
        if len(augmented) < len(sequences) * augment_factor:
            flipped = seq[0] + ''.join('1' if c == '0' else '0' for c in seq[1::2]) + seq[2::2]
            if len(flipped) >= 10:
                augmented.append(flipped)
        
        # Add pattern variations (flip every 3rd digit)
        if len(augmented) < len(sequences) * augment_factor:
            flipped = seq[:2] + ''.join('1' if c == '0' else '0' for c in seq[2::3]) + seq[3::3]
            if len(flipped) >= 10:
                augmented.append(flipped)
        
        # Add random subsequences
        if len(seq) > 20 and len(augmented) < len(sequences) * augment_factor:
            import random
            start = random.randint(0, len(seq) - 20)
            end = random.randint(start + 10, len(seq))
            augmented.append(seq[start:end])
        
        # Add synthetic alternating patterns
        if len(augmented) < len(sequences) * augment_factor:
            import random
            pattern_length = random.randint(2, 8)
            synthetic = ''
            for i in range(min(60, len(seq))):
                synthetic += str(i % pattern_length % 2)
            if len(synthetic) >= 10:
                augmented.append(synthetic)
        
        # Add mathematical sequences (Fibonacci-like patterns)
        if len(augmented) < len(sequences) * augment_factor:
            import random
            a, b = random.randint(0, 1), random.randint(0, 1)
            synthetic = str(a) + str(b)
            for i in range(min(50, len(seq))):
                c = (a + b) % 2
                synthetic += str(c)
                a, b = b, c
            if len(synthetic) >= 10:
                augmented.append(synthetic)
        
        # Add noise variations (randomly flip some digits)
        if len(augmented) < len(sequences) * augment_factor:
            import random
            noisy = list(seq)
            for i in range(len(noisy)):
                if random.random() < 0.1:  # 10% chance to flip each digit
                    noisy[i] = '1' if noisy[i] == '0' else '0'
            if len(noisy) >= 10:
                augmented.append(''.join(noisy))
        
        # Add block patterns (repeating blocks)
        if len(augmented) < len(sequences) * augment_factor:
            import random
            block_size = random.randint(2, 6)
            block = ''.join(str(random.randint(0, 1)) for _ in range(block_size))
            synthetic = ''
            for i in range(min(50, len(seq))):
                synthetic += block[i % block_size]
            if len(synthetic) >= 10:
                augmented.append(synthetic)
        
        # Add sequences with specific patterns (0101, 1010, etc.)
        if len(augmented) < len(sequences) * augment_factor:
            patterns = ['0101', '1010', '0011', '1100', '0110', '1001']
            pattern = random.choice(patterns)
            synthetic = ''
            for i in range(min(50, len(seq))):
                synthetic += pattern[i % len(pattern)]
            if len(synthetic) >= 10:
                augmented.append(synthetic)
    
    print(f"Augmented {len(sequences)} -> {len(augmented)} sequences")
    return augmented


def create_model_config(args) -> dict:
    """Create model configuration from arguments"""
    return {
        'vocab_size': 4,  # 0, 1, START_TOKEN, PAD_TOKEN
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'max_seq_length': args.max_seq_length,
        'dropout': args.dropout
    }


def main():
    parser = argparse.ArgumentParser(description="Train Binary Sequence Transformer")
    
    # Data arguments
    parser.add_argument("--data-source", choices=['database', 'file'], default='database',
                       help="Source of training data")
    parser.add_argument("--data-file", help="JSON file containing sequences (if using file source)")
    parser.add_argument("--database-url", help="Database connection string")
    parser.add_argument("--min-unpredictability", type=float, default=0.0,
                       help="Minimum unpredictability threshold for database extraction")
    parser.add_argument("--method-filter", choices=['frequency', 'pattern'],
                       help="Filter sequences by prediction method")
    
    # Data processing arguments
    parser.add_argument("--min-length", type=int, default=10, help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--augment", type=int, default=1, help="Data augmentation factor")
    parser.add_argument("--duplicate", type=int, default=1, help="Simple duplication factor (each sequence repeated N times)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation data ratio")
    
    # Model arguments
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=512, help="Feedforward dimension")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default='auto', help="Device to use (auto, cpu, cuda)")
    
    # Output arguments
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument("--model-name", default="binary_transformer", help="Model name")
    parser.add_argument("--save-every", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N epochs")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without training")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Binary Sequence Transformer Training")
    print("=" * 50)
    
    # Load sequences
    print("Loading sequences...")
    if args.data_source == 'database':
        sequences = load_sequences_from_database(
            args.database_url, 
            args.min_unpredictability,
            args.method_filter
        )
    else:
        if not args.data_file:
            print("Error: --data-file required when using file source")
            return
        sequences = load_sequences_from_file(args.data_file)
    
    if not sequences:
        print("No sequences found! Check your data source and filters.")
        return
    
    # Filter sequences
    sequences = filter_sequences(sequences, args.min_length, args.max_length)
    
    if not sequences:
        print("No sequences remain after filtering!")
        return
    
    # Duplicate sequences if requested (simpler than augmentation)
    if args.duplicate > 1:
        sequences = duplicate_sequences(sequences, args.duplicate)
    
    # Augment data if requested (alternative to duplication)
    if args.augment > 1:
        sequences = augment_sequences(sequences, args.augment)
    
    # Create train/validation split
    from .data_extractor import DataExtractor
    extractor = DataExtractor()  # Just for the split method
    train_sequences, val_sequences = extractor.create_train_val_split(
        sequences, args.val_ratio, args.seed
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BinarySequenceDataset(train_sequences, args.max_seq_length)
    val_dataset = BinarySequenceDataset(val_sequences, args.max_seq_length) if val_sequences else None
    
    # Create model
    print("Creating model...")
    model_config = create_model_config(args)
    BinaryTransformer = get_model_class("binary_transformer")
    model = BinaryTransformer(**model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Show configuration
    print("\nConfiguration:")
    print(f"  Data source: {args.data_source}")
    print(f"  Duplication factor: {args.duplicate}x")
    print(f"  Augmentation factor: {args.augment}x")
    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Validation sequences: {len(val_sequences) if val_sequences else 0}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Transformer layers: {args.num_layers}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    
    if args.dry_run:
        print("\nDry run completed. No training performed.")
        return
    
    # Create trainer
    print("\nInitializing trainer...")
    BinaryTransformerTrainer = get_trainer_class("binary_transformer")
    trainer = BinaryTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Train model
    print("\nStarting training...")
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pth")
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_path=model_path,
        print_every=args.print_every
    )
    
    # Save final model and configuration
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}_final.pth")
    trainer.save_model(final_model_path)
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, f"{args.model_name}_config.json")
    config = {
        'model_config': model_config,
        'training_args': vars(args),
        'training_history': history,
        'data_stats': {
            'total_sequences': len(sequences),
            'train_sequences': len(train_sequences),
            'val_sequences': len(val_sequences) if val_sequences else 0,
            'avg_sequence_length': sum(len(s) for s in sequences) / len(sequences)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {final_model_path}")
    print(f"Configuration saved to: {config_path}")
    
    # Test the model
    print("\nTesting model...")
    test_sequences = ["0101", "1100", "0011"]
    
    for test_seq in test_sequences:
        next_digit, confidence = model.predict_next(test_seq)
        print(f"  {test_seq} -> {next_digit} (confidence: {confidence:.3f})")
    
    # Auto-deploy to backend models directory
    print("\nDeploying model to backend...")
    backend_models_dir = "/app/backend_models"
    if os.path.exists(backend_models_dir):
        import shutil
        backend_model_path = os.path.join(backend_models_dir, f"{args.model_name}_final.pth")
        shutil.copy2(final_model_path, backend_model_path)
        print(f"✓ Model deployed to: {backend_model_path}")
        print("✓ Backend will automatically load the new model on next restart")
    else:
        print("⚠ Backend models directory not found - manual deployment required")


if __name__ == "__main__":
    main()
