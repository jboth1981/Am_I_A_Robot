#!/usr/bin/env python3
"""
Standalone Inference Script for Binary Sequence Transformer
This script loads a trained transformer model and provides predictions for binary sequences.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Optional
import torch

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_model.models.binary_transformer.model import BinaryTransformer
from shared_model.models.binary_transformer.trainer import BinaryTransformerTrainer


class TransformerPredictor:
    """Standalone predictor using trained transformer model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the predictor
        Args:
            model_path: Path to the saved model file
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = BinaryTransformerTrainer.load_model(model_path, device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
        
        # Load configuration if available
        config_path = model_path.replace('.pth', '_config.json')
        self.config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {config_path}")
    
    def predict_next(self, sequence: str, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Predict the next digit in a sequence
        Args:
            sequence: Binary sequence string (e.g., "010101")
            temperature: Sampling temperature
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        if not all(c in '01' for c in sequence):
            raise ValueError("Sequence must contain only 0s and 1s")
        
        return self.model.predict_next(sequence, temperature)
    
    def predict_multiple(self, sequences: List[str], temperature: float = 1.0) -> List[Tuple[str, float]]:
        """Predict next digits for multiple sequences"""
        results = []
        for seq in sequences:
            try:
                result = self.predict_next(seq, temperature)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for sequence '{seq}': {e}")
                results.append(("?", 0.0))
        return results
    
    def generate_sequence(self, start: str = "", length: int = 10, temperature: float = 1.0) -> str:
        """Generate a complete sequence"""
        return self.model.generate_sequence(start, length, temperature)
    
    def evaluate_predictions(self, test_sequences: List[str]) -> dict:
        """
        Evaluate the model on test sequences
        Args:
            test_sequences: List of complete binary sequences
        Returns:
            Dictionary with evaluation metrics
        """
        total_predictions = 0
        correct_predictions = 0
        confidences = []
        
        for sequence in test_sequences:
            if len(sequence) < 2:
                continue
            
            # Test prediction for each position (except the first)
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                true_next = sequence[i]
                
                try:
                    predicted_next, confidence = self.predict_next(input_seq)
                    
                    total_predictions += 1
                    if predicted_next == true_next:
                        correct_predictions += 1
                    
                    confidences.append(confidence)
                    
                except Exception as e:
                    print(f"Error evaluating position {i} in sequence '{sequence}': {e}")
        
        if total_predictions == 0:
            return {'error': 'No valid predictions made'}
        
        accuracy = correct_predictions / total_predictions
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'unpredictability_rate': 1 - accuracy
        }
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\nInteractive Prediction Mode")
        print("=" * 40)
        print("Enter binary sequences (0s and 1s only)")
        print("Commands: 'quit' to exit, 'generate' to generate a sequence")
        print()
        
        while True:
            try:
                user_input = input("Enter sequence (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower().startswith('generate'):
                    # Parse generate command
                    parts = user_input.split()
                    length = 20
                    start = ""
                    temp = 1.0
                    
                    if len(parts) > 1:
                        try:
                            length = int(parts[1])
                        except ValueError:
                            pass
                    
                    if len(parts) > 2:
                        start = parts[2]
                    
                    if len(parts) > 3:
                        try:
                            temp = float(parts[3])
                        except ValueError:
                            pass
                    
                    generated = self.generate_sequence(start, length, temp)
                    print(f"Generated: {generated}")
                
                elif all(c in '01' for c in user_input) and user_input:
                    # Predict next digit
                    next_digit, confidence = self.predict_next(user_input)
                    print(f"Sequence: {user_input}")
                    print(f"Predicted next: {next_digit} (confidence: {confidence:.3f})")
                    
                    # Show what the sequence would look like
                    print(f"Full sequence: {user_input}{next_digit}")
                
                else:
                    print("Invalid input. Enter a binary sequence (0s and 1s only)")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    def benchmark_against_methods(self, test_sequences: List[str]) -> dict:
        """
        Benchmark transformer against simple methods
        Args:
            test_sequences: List of test sequences
        Returns:
            Comparison results
        """
        print("Benchmarking against simple methods...")
        
        # Evaluate transformer
        transformer_results = self.evaluate_predictions(test_sequences)
        
        # Evaluate frequency method
        freq_correct = 0
        freq_total = 0
        
        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                true_next = sequence[i]
                
                # Frequency prediction
                count_0 = input_seq.count('0')
                count_1 = input_seq.count('1')
                freq_pred = '1' if count_1 > count_0 else '0'
                
                freq_total += 1
                if freq_pred == true_next:
                    freq_correct += 1
        
        # Evaluate pattern method
        pattern_correct = 0
        pattern_total = 0
        
        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                true_next = sequence[i]
                
                # Pattern prediction (simplified)
                if len(input_seq) < 3:
                    pattern_pred = '0'
                else:
                    last_3 = input_seq[-3:]
                    if last_3 == "000":
                        pattern_pred = "0"
                    elif last_3 == "111":
                        pattern_pred = "1"
                    else:
                        pattern_pred = input_seq[-1]
                
                pattern_total += 1
                if pattern_pred == true_next:
                    pattern_correct += 1
        
        return {
            'transformer': {
                'accuracy': transformer_results.get('accuracy', 0),
                'unpredictability_rate': transformer_results.get('unpredictability_rate', 0)
            },
            'frequency': {
                'accuracy': freq_correct / freq_total if freq_total > 0 else 0,
                'unpredictability_rate': 1 - (freq_correct / freq_total) if freq_total > 0 else 0
            },
            'pattern': {
                'accuracy': pattern_correct / pattern_total if pattern_total > 0 else 0,
                'unpredictability_rate': 1 - (pattern_correct / pattern_total) if pattern_total > 0 else 0
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Binary Sequence Transformer Predictor")
    
    parser.add_argument("model_path", help="Path to the trained model file")
    parser.add_argument("--device", default='auto', help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true", help="Interactive prediction mode")
    parser.add_argument("--sequence", help="Single sequence to predict")
    parser.add_argument("--sequences-file", help="File containing sequences to predict")
    parser.add_argument("--generate", type=int, help="Generate sequence of given length")
    parser.add_argument("--generate-start", default="", help="Starting sequence for generation")
    parser.add_argument("--evaluate", help="Evaluate model on sequences from file")
    parser.add_argument("--benchmark", help="Benchmark against simple methods using sequences from file")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        return
    
    # Create predictor
    try:
        predictor = TransformerPredictor(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Execute based on mode
    if args.interactive:
        predictor.interactive_mode()
    
    elif args.sequence:
        try:
            next_digit, confidence = predictor.predict_next(args.sequence, args.temperature)
            print(f"Sequence: {args.sequence}")
            print(f"Predicted next: {next_digit} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.sequences_file:
        try:
            with open(args.sequences_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
            
            results = predictor.predict_multiple(sequences, args.temperature)
            
            print("Predictions:")
            for seq, (pred, conf) in zip(sequences, results):
                print(f"  {seq} -> {pred} (confidence: {conf:.3f})")
        
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.generate:
        try:
            generated = predictor.generate_sequence(
                args.generate_start, 
                args.generate, 
                args.temperature
            )
            print(f"Generated sequence: {generated}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.evaluate:
        try:
            with open(args.evaluate, 'r') as f:
                if args.evaluate.endswith('.json'):
                    data = json.load(f)
                    if 'sequences' in data:
                        sequences = [item['sequence'] for item in data['sequences']]
                    else:
                        sequences = data
                else:
                    sequences = [line.strip() for line in f if line.strip()]
            
            results = predictor.evaluate_predictions(sequences)
            
            print("Evaluation Results:")
            print(f"  Total predictions: {results.get('total_predictions', 0)}")
            print(f"  Correct predictions: {results.get('correct_predictions', 0)}")
            print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
            print(f"  Average confidence: {results.get('avg_confidence', 0):.3f}")
            print(f"  Unpredictability rate: {results.get('unpredictability_rate', 0):.3f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.benchmark:
        try:
            with open(args.benchmark, 'r') as f:
                if args.benchmark.endswith('.json'):
                    data = json.load(f)
                    if 'sequences' in data:
                        sequences = [item['sequence'] for item in data['sequences']]
                    else:
                        sequences = data
                else:
                    sequences = [line.strip() for line in f if line.strip()]
            
            results = predictor.benchmark_against_methods(sequences)
            
            print("Benchmark Results:")
            print("=" * 40)
            for method, metrics in results.items():
                print(f"{method.capitalize()}:")
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
                print(f"  Unpredictability rate: {metrics['unpredictability_rate']:.3f}")
                print()
        
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("No mode specified. Use --help for options.")
        print("Quick start: use --interactive for interactive mode")


if __name__ == "__main__":
    main()
