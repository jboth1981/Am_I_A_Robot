#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates transformer models on binary sequence prediction task
"""

import sys
import os
sys.path.append('/app/src')
sys.path.append('/app/shared_model')

from data_extractor import DataExtractor
from shared_model.models.binary_transformer.model import BinaryTransformer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from collections import defaultdict
import argparse

def evaluate_model(model_path, sequences=None):
    """Comprehensive evaluation of a transformer model"""
    
    print(f"Evaluating model: {os.path.basename(model_path)}")
    print("=" * 60)
    
    # Load model
    model = BinaryTransformer.load_model_from_file(model_path)
    
    # Load data if not provided
    if sequences is None:
        extractor = DataExtractor()
        sequences = extractor.get_all_sequences()
        sequences = [s for s in sequences if 10 <= len(s) <= 100]
    
    print(f"Testing on {len(sequences)} sequences")
    
    # Collect predictions
    predictions = []
    actuals = []
    confidences = []
    sequence_lengths = []
    position_metrics = defaultdict(list)  # Track performance by sequence position
    
    for seq_idx, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
            
        seq_length = len(seq)
        sequence_lengths.append(seq_length)
        
        # Test each position in the sequence
        for i in range(len(seq) - 1):
            input_seq = seq[:i+1]
            actual_next = seq[i+1]
            
            try:
                pred, conf = model.predict_next(input_seq)
                predictions.append(int(pred))
                actuals.append(int(actual_next))
                confidences.append(conf)
                
                # Track by position in sequence
                position_metrics[i].append({
                    'pred': int(pred),
                    'actual': int(actual_next),
                    'confidence': conf,
                    'correct': int(pred) == int(actual_next)
                })
                
            except Exception as e:
                print(f"Error predicting for sequence {seq_idx} at position {i}: {e}")
                continue
    
    # Basic metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='binary')
    recall = recall_score(actuals, predictions, average='binary')
    f1 = f1_score(actuals, predictions, average='binary')
    
    print(f"\n=== BASIC METRICS ===")
    print(f"Total predictions: {len(predictions):,}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(actuals, predictions)
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"                 Predicted")
    print(f"Actual    0     1")
    print(f"    0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"    1   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    actual_0s = actuals.count(0)
    actual_1s = actuals.count(1)
    pred_0s = predictions.count(0)
    pred_1s = predictions.count(1)
    
    print(f"Actual 0s: {actual_0s:,} ({actual_0s/len(actuals)*100:.1f}%)")
    print(f"Actual 1s: {actual_1s:,} ({actual_1s/len(actuals)*100:.1f}%)")
    print(f"Predicted 0s: {pred_0s:,} ({pred_0s/len(predictions)*100:.1f}%)")
    print(f"Predicted 1s: {pred_1s:,} ({pred_1s/len(predictions)*100:.1f}%)")
    
    # Position-based analysis
    print(f"\n=== POSITION-BASED ANALYSIS ===")
    position_accuracies = []
    position_confidences = []
    
    for pos in sorted(position_metrics.keys()):
        pos_data = position_metrics[pos]
        if len(pos_data) > 0:
            pos_accuracy = sum(d['correct'] for d in pos_data) / len(pos_data)
            pos_conf = np.mean([d['confidence'] for d in pos_data])
            position_accuracies.append(pos_accuracy)
            position_confidences.append(pos_conf)
            
            if pos < 10 or pos % 10 == 0:  # Show first 10 and every 10th position
                print(f"Position {pos:2d}: Accuracy {pos_accuracy:.3f}, Confidence {pos_conf:.3f}, Samples {len(pos_data):3d}")
    
    # Sequence length analysis
    print(f"\n=== SEQUENCE LENGTH ANALYSIS ===")
    length_groups = {
        'Short (10-30)': [s for s in sequences if 10 <= len(s) <= 30],
        'Medium (31-70)': [s for s in sequences if 31 <= len(s) <= 70],
        'Long (71-100)': [s for s in sequences if 71 <= len(s) <= 100]
    }
    
    for group_name, group_seqs in length_groups.items():
        if group_seqs:
            print(f"{group_name}: {len(group_seqs)} sequences")
    
    # Pattern analysis
    print(f"\n=== PATTERN ANALYSIS ===")
    
    # Test specific patterns
    test_patterns = {
        'All zeros': '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000',
        'All ones': '1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111',
        'Alternating': '0101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101',
        'Repeating pairs': '0011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011',
        'Random-like': '1011001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101'
    }
    
    print("Pattern prediction tests:")
    for pattern_name, pattern in test_patterns.items():
        try:
            pred, conf = model.predict_next(pattern)
            print(f"{pattern_name:15s}: Predicts '{pred}' (confidence: {conf:.3f})")
        except Exception as e:
            print(f"{pattern_name:15s}: Error - {e}")
    
    # Game-specific metrics
    print(f"\n=== GAME-SPECIFIC METRICS ===")
    
    # Calculate "unpredictability" - how often the model gets it wrong
    unpredictability = 1 - accuracy
    print(f"Model unpredictability: {unpredictability:.3f} ({unpredictability*100:.1f}%)")
    print(f"This means the model is wrong {unpredictability*100:.1f}% of the time")
    print(f"Higher unpredictability = more human-like behavior")
    
    # Confidence distribution
    high_conf = sum(1 for c in confidences if c > 0.8)
    med_conf = sum(1 for c in confidences if 0.5 <= c <= 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)
    
    print(f"\nConfidence distribution:")
    print(f"High confidence (>0.8): {high_conf:,} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"Medium confidence (0.5-0.8): {med_conf:,} ({med_conf/len(confidences)*100:.1f}%)")
    print(f"Low confidence (<0.5): {low_conf:,} ({low_conf/len(confidences)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'unpredictability': unpredictability,
        'avg_confidence': np.mean(confidences),
        'total_predictions': len(predictions),
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            'actual_0s': actual_0s,
            'actual_1s': actual_1s,
            'pred_0s': pred_0s,
            'pred_1s': pred_1s
        }
    }

def compare_models(model_paths):
    """Compare multiple models"""
    print("COMPARING MODELS")
    print("=" * 60)
    
    results = {}
    for model_path in model_paths:
        if os.path.exists(model_path):
            results[os.path.basename(model_path)] = evaluate_model(model_path)
            print("\n" + "="*60 + "\n")
    
    # Summary comparison
    print("=== MODEL COMPARISON SUMMARY ===")
    print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'Unpredictability':<15} {'Confidence':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['accuracy']:<10.3f} {metrics['f1']:<10.3f} {metrics['unpredictability']:<15.3f} {metrics['avg_confidence']:<12.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transformer models")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in models directory")
    
    args = parser.parse_args()
    
    if args.all:
        models_dir = "/app/models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith("_final.pth")]
        model_paths = [os.path.join(models_dir, f) for f in model_files]
        compare_models(model_paths)
    elif args.compare:
        compare_models(args.compare)
    elif args.model:
        evaluate_model(args.model)
    else:
        # Default: evaluate the tiny model
        evaluate_model("/app/models/tiny_gpu_model_final.pth")
