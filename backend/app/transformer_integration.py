"""
Integration module for using transformer model in FastAPI backend
This shows how to integrate the trained transformer model with your existing prediction system.
"""

import os
from typing import Optional, Tuple
import asyncio
from functools import lru_cache

try:
    from shared_model.config.model_registry import get_model_class
    from shared_model.models.binary_transformer.trainer import BinaryTransformerTrainer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: Transformer dependencies not available. Install torch to use transformer predictions.")


class TransformerPredictionService:
    """Service for managing transformer-based predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or os.path.join("models", "binary_transformer.pth")
        self.is_loaded = False
        
        if TRANSFORMER_AVAILABLE:
            self.load_model()
    
    def load_model(self) -> bool:
        """Load the transformer model if available"""
        if not TRANSFORMER_AVAILABLE:
            return False
        
        if not os.path.exists(self.model_path):
            print(f"Transformer model not found at {self.model_path}")
            return False
        
        try:
            BinaryTransformer = get_model_class("binary_transformer")
            self.model = BinaryTransformer.load_model(self.model_path, device='cpu')
            self.model.eval()
            self.is_loaded = True
            print(f"Transformer model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            return False
    
    def predict(self, history: str, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Make a prediction using the transformer model
        Args:
            history: Binary sequence history
            temperature: Sampling temperature
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_loaded or not self.model:
            raise RuntimeError("Transformer model not loaded")
        
        return self.model.predict_next(history, temperature)
    
    def is_available(self) -> bool:
        """Check if transformer predictions are available"""
        return TRANSFORMER_AVAILABLE and self.is_loaded


# Global transformer service instance
_transformer_service = None


def get_transformer_service() -> TransformerPredictionService:
    """Get the global transformer service instance"""
    global _transformer_service
    if _transformer_service is None:
        _transformer_service = TransformerPredictionService()
    return _transformer_service


# Enhanced prediction functions that include transformer option
def predict_frequency(history: str) -> str:
    """Simple frequency-based prediction: predict most frequent digit"""
    if not history:
        return "0"
    
    count_0 = history.count("0")
    count_1 = history.count("1")
    
    prediction = "1" if count_1 > count_0 else "0"
    
    return prediction


def predict_pattern(history: str) -> str:
    """Pattern-based prediction with special rules for 000 and 111"""
    if not history:
        return "0"
    
    # For first 3 characters, always predict 0
    if len(history) < 3:
        return "0"
    
    # Get the last 3 characters
    last_3 = history[-3:]
    
    # Special rules for 000 and 111
    if last_3 == "000":
        return "0"
    elif last_3 == "111":
        return "1"
    else:
        # Otherwise, predict the most recent character
        return history[-1]


def predict_transformer(history: str, temperature: float = 1.0) -> Tuple[str, float]:
    """
    Transformer-based prediction
    Args:
        history: Binary sequence history
        temperature: Sampling temperature
    Returns:
        Tuple of (prediction, confidence)
    """
    service = get_transformer_service()
    if not service.is_available():
        # Fallback to frequency method
        prediction = predict_frequency(history)
        return prediction, 0.5  # Default confidence
    
    return service.predict(history, temperature)


def predict_enhanced(history: str, method: str = "frequency", temperature: float = 1.0) -> dict:
    """
    Enhanced prediction function that supports all methods including transformer
    Args:
        history: Binary sequence history
        method: Prediction method ('frequency', 'pattern', 'transformer')
        temperature: Sampling temperature (for transformer only)
    Returns:
        Dictionary with prediction and metadata
    """
    if method == "frequency":
        prediction = predict_frequency(history)
        return {
            "prediction": prediction,
            "method": "frequency",
            "confidence": 0.5,  # Frequency method doesn't provide confidence
            "fallback": False
        }
    
    elif method == "pattern":
        prediction = predict_pattern(history)
        return {
            "prediction": prediction,
            "method": "pattern",
            "confidence": 0.5,  # Pattern method doesn't provide confidence
            "fallback": False
        }
    
    elif method == "transformer":
        try:
            prediction, confidence = predict_transformer(history, temperature)
            return {
                "prediction": prediction,
                "method": "transformer",
                "confidence": confidence,
                "fallback": False
            }
        except Exception as e:
            # Fallback to frequency method
            prediction = predict_frequency(history)
            return {
                "prediction": prediction,
                "method": "frequency",
                "confidence": 0.5,
                "fallback": True,
                "fallback_reason": str(e)
            }
    
    else:
        # Default to frequency method
        prediction = predict_frequency(history)
        return {
            "prediction": prediction,
            "method": "frequency",
            "confidence": 0.5,
            "fallback": True,
            "fallback_reason": f"Unknown method: {method}"
        }


# Example of how to modify your existing FastAPI endpoint
"""
To integrate with your existing FastAPI app, modify your prediction endpoint in main.py:

from app.transformer_integration import predict_enhanced

@app.post("/predict/")
def predict_next(data: InputData):
    '''Enhanced prediction endpoint with transformer support'''
    history = data.history
    method = data.method
    temperature = getattr(data, 'temperature', 1.0)  # Add temperature to InputData if needed
    
    result = predict_enhanced(history, method, temperature)
    
    return {
        "prediction": result["prediction"],
        "method": result["method"],
        "confidence": result["confidence"],
        "fallback": result.get("fallback", False)
    }

# You might also want to add a new InputData model that includes temperature:

class EnhancedInputData(BaseModel):
    history: str
    method: str = "frequency"  # "frequency", "pattern", or "transformer"
    temperature: float = 1.0  # For transformer sampling
"""


# Utility functions for model management
def retrain_model_from_database(
    database_url: Optional[str] = None,
    min_unpredictability: float = 0.0,
    epochs: int = 50,
    output_path: str = "models/binary_transformer_retrained.pth"
) -> bool:
    """
    Retrain the transformer model with fresh data from the database
    Args:
        database_url: Database connection string
        min_unpredictability: Minimum unpredictability threshold for training data
        epochs: Number of training epochs
        output_path: Where to save the retrained model
    Returns:
        True if retraining was successful
    """
    if not TRANSFORMER_AVAILABLE:
        print("Transformer dependencies not available")
        return False
    
    try:
        from .data_extractor import DataExtractor
        from .transformer_predictor import BinarySequenceDataset, BinaryTransformerTrainer
        
        # Extract training data
        extractor = DataExtractor(database_url)
        sequences = extractor.get_sequences_by_performance(min_unpredictability)
        extractor.close()
        
        if len(sequences) < 10:
            print(f"Not enough training data: {len(sequences)} sequences")
            return False
        
        # Create train/val split
        val_size = max(1, len(sequences) // 5)  # 20% validation
        train_sequences = sequences[val_size:]
        val_sequences = sequences[:val_size]
        
        # Create datasets
        train_dataset = BinarySequenceDataset(train_sequences)
        val_dataset = BinarySequenceDataset(val_sequences)
        
        # Create model
        model = BinaryTransformer(d_model=128, nhead=8, num_layers=6)
        
        # Create trainer
        trainer = BinaryTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=1e-4,
            batch_size=16
        )
        
        # Train
        print(f"Retraining model with {len(train_sequences)} sequences...")
        trainer.train(num_epochs=epochs, save_path=output_path, print_every=10)
        
        print(f"Model retrained and saved to {output_path}")
        
        # Reload the global service with the new model
        global _transformer_service
        _transformer_service = TransformerPredictionService(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False


async def async_retrain_model(
    database_url: Optional[str] = None,
    min_unpredictability: float = 0.0,
    epochs: int = 50
) -> bool:
    """Async version of model retraining"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        retrain_model_from_database,
        database_url,
        min_unpredictability,
        epochs
    )


if __name__ == "__main__":
    # Test the integration
    print("Testing Transformer Integration")
    print("=" * 40)
    
    # Test basic functionality
    service = get_transformer_service()
    print(f"Transformer available: {service.is_available()}")
    
    # Test predictions
    test_sequences = ["0101", "1100", "0011"]
    
    for seq in test_sequences:
        print(f"\nTesting sequence: {seq}")
        
        # Test all methods
        for method in ["frequency", "pattern", "transformer"]:
            result = predict_enhanced(seq, method)
            print(f"  {method}: {result['prediction']} (confidence: {result['confidence']:.3f})")
            if result.get('fallback'):
                print(f"    Fallback used: {result.get('fallback_reason')}")

