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
        self.model_path = model_path or os.path.join("models", "batch_8_model_final.pth")
        self.is_loaded = False
        
        if TRANSFORMER_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load the transformer model for inference"""
        try:
            if os.path.exists(self.model_path):
                self.model = BinaryTransformer.load_model_from_file(self.model_path)
                self.is_loaded = True
                
                # Get model file info for identification
                import stat
                file_stats = os.stat(self.model_path)
                file_size = file_stats.st_size
                file_mtime = file_stats.st_mtime
                import datetime
                mod_time = datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Extract model name from path
                model_name = os.path.basename(self.model_path)
                
                print(f"✓ Transformer model loaded: {model_name}")
                print(f"  📁 Path: {self.model_path}")
                print(f"  📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                print(f"  🕒 Modified: {mod_time}")
                print(f"  🧠 Model ready for inference")
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
                # Log transformer usage
                model_name = os.path.basename(service.model_path)
                print(f"🤖 Using transformer model: {model_name} for sequence: '{history[:20]}{'...' if len(history) > 20 else ''}'")
                
                prediction, confidence = service.predict_next(history)
                return {
                    "prediction": prediction,
                    "method": "transformer",
                    "confidence": confidence,
                    "fallback": False
                }
            else:
                print(f"⚠ Transformer model not loaded, falling back to frequency method")
                # Fallback to simple frequency method
                prediction = predict_frequency_simple(history)
                return {
                    "prediction": prediction,
                    "method": "frequency",
                    "confidence": 0.5,
                    "fallback": True
                }
        except Exception as e:
            print(f"✗ Transformer prediction failed: {e}, falling back to frequency method")
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
