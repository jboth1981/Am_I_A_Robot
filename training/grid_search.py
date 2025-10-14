#!/usr/bin/env python3
"""
Grid Search Script for Transformer Training
Tests different epoch counts with fixed architecture parameters
"""

import os
import sys
import json
import subprocess
import time
from typing import List, Dict, Tuple
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_model.config.model_registry import get_model_class, get_trainer_class, get_dataset_class
from shared_model.models.binary_transformer.dataset import BinarySequenceDataset
from src.data_extractor import DataExtractor

def run_training_experiment(
    epochs: int,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    d_model: int = 16,
    nhead: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 64,
    model_name: str = None
) -> Dict:
    """Run a single training experiment and return results"""
    
    if model_name is None:
        model_name = f"grid_search_epochs_{epochs}"
    
    print(f"\n{'='*60}")
    print(f"TRAINING EXPERIMENT: {epochs} epochs")
    print(f"{'='*60}")
    
    # Load data
    extractor = DataExtractor()
    sequences = extractor.get_all_sequences()
    valid_sequences = [s for s in sequences if 10 <= len(s) <= 100]
    
    # Split data (same split as main training)
    train_sequences = valid_sequences[:-10]  # All but last 10
    val_sequences = valid_sequences[-10:]    # Last 10 for validation
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create datasets
    train_dataset = BinarySequenceDataset(train_sequences, max_length=100)
    val_dataset = BinarySequenceDataset(val_sequences, max_length=100)
    
    # Create model
    model_config = {
        'vocab_size': 4,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'max_seq_length': 100,
        'dropout': 0.1
    }
    
    model_class = get_model_class('binary_transformer')
    model = model_class(**model_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer_class = get_trainer_class('binary_transformer')
    trainer = trainer_class(
        model=model,
        learning_rate=learning_rate,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train model
    start_time = time.time()
    training_results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_every=epochs,  # Save at the end
        print_every=max(1, epochs // 20)  # Print ~20 times during training
    )
    training_time = time.time() - start_time
    
    # Get final validation accuracy
    final_val_accuracy = training_results['val_accuracies'][-1] if training_results['val_accuracies'] else 0.0
    final_train_loss = training_results['train_losses'][-1] if training_results['train_losses'] else float('inf')
    final_val_loss = training_results['val_losses'][-1] if training_results['val_losses'] else float('inf')
    
    # Save model
    model_path = f"models/{model_name}_final.pth"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    
    # Test on validation sequences
    model.eval()
    val_predictions = []
    val_actuals = []
    
    for seq in val_sequences:
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            input_seq = seq[:i+1]
            actual_next = seq[i+1]
            try:
                pred, _ = model.predict_next(input_seq)
                val_predictions.append(int(pred))
                val_actuals.append(int(actual_next))
            except Exception as e:
                print(f"Error predicting: {e}")
                continue
    
    # Calculate validation accuracy
    if val_predictions and val_actuals:
        val_accuracy = sum(1 for p, a in zip(val_predictions, val_actuals) if p == a) / len(val_predictions)
    else:
        val_accuracy = 0.0
    
    results = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'model_name': model_name,
        'training_time': training_time,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'validation_accuracy': val_accuracy,
        'total_predictions': len(val_predictions),
        'model_path': model_path
    }
    
    print(f"\nRESULTS:")
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Final val accuracy: {final_val_accuracy:.4f}")
    print(f"  Validation accuracy: {val_accuracy:.4f}")
    print(f"  Total predictions: {len(val_predictions)}")
    
    return results

def run_grid_search():
    """Run grid search over epoch counts"""
    
    print("GRID SEARCH: Transformer Training")
    print("="*60)
    print("Fixed parameters:")
    print("  batch_size: 4")
    print("  learning_rate: 0.001")
    print("  d_model: 16")
    print("  nhead: 4")
    print("  num_layers: 1")
    print("  dim_feedforward: 64")
    print()
    print("Variable parameter: epochs (100 to 10,000)")
    print("="*60)
    
    # Define epoch values (10 cuts between 100 and 10,000)
    epoch_values = [int(x) for x in np.linspace(100, 10000, 10)]
    
    print(f"Epoch values to test: {epoch_values}")
    print()
    
    all_results = []
    
    for i, epochs in enumerate(epoch_values):
        print(f"\nEXPERIMENT {i+1}/10")
        try:
            result = run_training_experiment(
                epochs=epochs,
                batch_size=4,
                learning_rate=0.001,
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                model_name=f"grid_search_{epochs}_epochs"
            )
            all_results.append(result)
            
            # Save intermediate results
            with open('grid_search_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"ERROR in experiment {i+1}: {e}")
            continue
    
    # Print final summary
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    print(f"{'Epochs':<8} {'Val Acc':<8} {'Train Loss':<12} {'Val Loss':<12} {'Time (s)':<10}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['epochs']:<8} {result['validation_accuracy']:<8.4f} "
              f"{result['final_train_loss']:<12.4f} {result['final_val_loss']:<12.4f} "
              f"{result['training_time']:<10.1f}")
    
    # Find best result
    if all_results:
        best_result = max(all_results, key=lambda x: x['validation_accuracy'])
        print(f"\nBEST RESULT:")
        print(f"  Epochs: {best_result['epochs']}")
        print(f"  Validation Accuracy: {best_result['validation_accuracy']:.4f}")
        print(f"  Model: {best_result['model_name']}")
        print(f"  Training Time: {best_result['training_time']:.1f} seconds")
    
    # Save final results
    with open('grid_search_final_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: grid_search_final_results.json")
    
    return all_results

if __name__ == "__main__":
    import torch
    run_grid_search()
