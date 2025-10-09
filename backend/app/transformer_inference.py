"""
Lightweight Transformer Inference Module
Only includes inference capabilities, no training dependencies
"""

import os
import sys
from typing import Dict, Any, Optional

# Add shared model to path
sys.path.append('/app/shared_model')

# Check if transformer dependencies are available
try:
    import torch
    from shared_model.models.binary_transformer.model import BinaryTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

class TransformerInferenceService:
    """Lightweight service for transformer inference only"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or os.path.join("models", "improved_transformer_final.pth")
        self.is_loaded = False
        
        if TRANSFORMER_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load the transformer model for inference"""
        try:
            if os.path.exists(self.model_path):
                self.model = BinaryTransformer.load_model_from_file(self.model_path)
                self.is_loaded = True
                print(f"✓ Transformer model loaded from {self.model_path}")
            else:
                print(f"⚠ Model not found at {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            print(f"✗ Failed to load transformer model: {e}")
            self.is_loaded = False
    
    def predict_next(self, sequence: str) -> tuple[int, float]:
        """Make a prediction using the transformer model"""
        if not self.is_loaded:
            raise RuntimeError("Transformer model not loaded")
        
        return self.model.predict_next(sequence)

# Global inference service
_inference_service = None

def get_inference_service() -> TransformerInferenceService:
    """Get or create the global inference service"""
    global _inference_service
    if _inference_service is None:
        _inference_service = TransformerInferenceService()
    return _inference_service

def predict_enhanced(history: str, method: str = "transformer", temperature: float = 1.0) -> Dict[str, Any]:
    """
    Enhanced prediction with transformer support
    Lightweight version - only inference, no training capabilities
    """
    if method == "transformer" and TRANSFORMER_AVAILABLE:
        try:
            service = get_inference_service()
            if service.is_loaded:
                prediction, confidence = service.predict_next(history)
                return {
                    "prediction": prediction,
                    "method": "transformer",
                    "confidence": confidence,
                    "fallback": False
                }
            else:
                # Fallback to simple frequency method
                prediction = predict_frequency_simple(history)
                return {
                    "prediction": prediction,
                    "method": "frequency",
                    "confidence": 0.5,
                    "fallback": True
                }
        except Exception as e:
            print(f"Transformer prediction failed: {e}")
            # Fallback to simple frequency method
            prediction = predict_frequency_simple(history)
            return {
                "prediction": prediction,
                "method": "frequency", 
                "confidence": 0.5,
                "fallback": True
            }
    else:
        # Use simple frequency method
        prediction = predict_frequency_simple(history)
        return {
            "prediction": prediction,
            "method": "frequency",
            "confidence": 0.5,
            "fallback": False
        }

def predict_frequency_simple(history: str) -> int:
    """Simple frequency-based prediction (no external dependencies)"""
    if not history:
        return 0
    
    # Count frequency of 0s and 1s
    count_0 = history.count('0')
    count_1 = history.count('1')
    
    # Return the more frequent digit, or 0 if tied
    return 0 if count_0 >= count_1 else 1
