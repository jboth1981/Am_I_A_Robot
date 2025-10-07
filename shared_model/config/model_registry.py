"""Model registry for dynamic model discovery and loading."""

from typing import Dict, Type, Any
from importlib import import_module

# Registry of all available models
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "binary_transformer": {
        "model_class": "shared_model.models.binary_transformer.model.BinaryTransformer",
        "trainer_class": "shared_model.models.binary_transformer.model.BinaryTransformerTrainer", 
        "dataset_class": "shared_model.models.binary_transformer.dataset.BinarySequenceDataset",
        "config_class": "shared_model.models.binary_transformer.config.BinaryTransformerConfig",
        "description": "Transformer-based binary sequence predictor"
    },
    # Future models can be easily added:
    # "binary_lstm": {
    #     "model_class": "shared_model.models.binary_lstm.model.BinaryLSTM",
    #     "trainer_class": "shared_model.models.binary_lstm.trainer.LSTMTrainer",
    #     "dataset_class": "shared_model.models.binary_lstm.dataset.BinarySequenceDataset",
    #     "config_class": "shared_model.models.binary_lstm.config.LSTMConfig",
    #     "description": "LSTM-based binary sequence predictor"
    # }
}

def get_model_class(model_type: str) -> Type:
    """Get model class by type name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_type]["model_class"].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def get_trainer_class(model_type: str) -> Type:
    """Get trainer class by model type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_type]["trainer_class"].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def get_dataset_class(model_type: str) -> Type:
    """Get dataset class by model type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_type]["dataset_class"].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def get_available_models() -> Dict[str, str]:
    """Get list of available models and their descriptions."""
    return {model_type: info["description"] for model_type, info in MODEL_REGISTRY.items()}

def create_model(model_type: str, **kwargs) -> Any:
    """Create a new model instance."""
    model_class = get_model_class(model_type)
    return model_class(**kwargs)

def create_trainer(model_type: str, **kwargs) -> Any:
    """Create a new trainer instance."""
    trainer_class = get_trainer_class(model_type)
    return trainer_class(**kwargs)
