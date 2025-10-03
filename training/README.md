# Binary Sequence Transformer Guide

This guide explains how to use the transformer-based binary sequence predictor for your "Am I A Robot" application.

## Overview

The transformer model treats binary sequences like language, where each position can be 0 or 1. It learns patterns from your user data to make more sophisticated predictions than simple frequency or pattern-based methods.

## Files Location

All training scripts are now located in `backend/app/`:
- `data_extractor.py` - Extract training data from your database
- `train_transformer.py` - Training script with full configuration options
- `predict_transformer.py` - Standalone inference script
- `transformer_integration.py` - Integration with your FastAPI backend

## Quick Start

### 1. Install Dependencies

The training dependencies are automatically installed when you build the training container:

```bash
# Build the training container
docker-compose -f docker-compose.training.yml build transformer-training
```

### 2. Extract Training Data

First, make sure your database is running and has some submission data:

```bash
# Check what data you have
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.data_extractor --stats-only

# Extract all sequences
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.data_extractor --output /app/data/training_data.json

# Extract only high-unpredictability sequences (more "human-like")
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.data_extractor --output /app/data/human_sequences.json --min-unpredictability 0.4
```

### 3. Train the Model

```bash
# Basic training (uses database directly)
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.train_transformer --epochs 50 --batch-size 16

# Training with specific data file
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.train_transformer --data-source file --data-file /app/data/training_data.json --epochs 100

# Advanced training with custom architecture
docker-compose -f docker-compose.training.yml run --rm transformer-training python -m app.train_transformer \
    --d-model 256 \
    --nhead 8 \
    --num-layers 8 \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --output-dir models \
    --model-name advanced_transformer
```

### 4. Test Your Model

```bash
# Interactive mode
python -m app.predict_transformer models/binary_transformer.pth --interactive

# Single prediction
python -m app.predict_transformer models/binary_transformer.pth --sequence "010101"

# Evaluate on test data
python -m app.predict_transformer models/binary_transformer.pth --evaluate training_data.json

# Benchmark against simple methods
python -m app.predict_transformer models/binary_transformer.pth --benchmark training_data.json
```

## Detailed Usage

### Data Extraction Options

The `data_extractor.py` script provides several ways to filter your training data:

```bash
# Get sequences by prediction method
python -m app.data_extractor --method frequency --output freq_data.json
python -m app.data_extractor --method pattern --output pattern_data.json

# Get sequences by performance (unpredictability rate)
python -m app.data_extractor --min-unpredictability 0.6 --output high_unpred.json

# Just show database statistics
python -m app.data_extractor --stats-only
```

### Training Configuration

The training script supports extensive configuration:

#### Model Architecture
- `--d-model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 8)
- `--num-layers`: Number of transformer layers (default: 6)
- `--dim-feedforward`: Feedforward dimension (default: 512)
- `--max-seq-length`: Maximum sequence length (default: 100)
- `--dropout`: Dropout rate (default: 0.1)

#### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--val-ratio`: Validation data ratio (default: 0.2)

#### Data Processing
- `--min-length`: Minimum sequence length (default: 10)
- `--max-length`: Maximum sequence length (default: 100)
- `--augment`: Data augmentation factor (default: 1)

### Model Performance

The transformer model provides several advantages over simple methods:

1. **Context Awareness**: Considers the entire sequence history, not just recent digits
2. **Pattern Learning**: Learns complex patterns from your actual user data
3. **Confidence Scores**: Provides confidence estimates for predictions
4. **Adaptability**: Can be retrained as you collect more data

### Integration with FastAPI

To integrate the transformer with your existing FastAPI backend:

1. **Add the new prediction method** to your `InputData` model:

```python
class EnhancedInputData(BaseModel):
    history: str
    method: str = "frequency"  # "frequency", "pattern", or "transformer"
    temperature: float = 1.0  # For transformer sampling
```

2. **Update your prediction endpoint**:

```python
from app.transformer_integration import predict_enhanced

@app.post("/predict/")
def predict_next(data: EnhancedInputData):
    result = predict_enhanced(data.history, data.method, data.temperature)
    return {
        "prediction": result["prediction"],
        "method": result["method"],
        "confidence": result["confidence"],
        "fallback": result.get("fallback", False)
    }
```

3. **Update your frontend** to support the new method:

```javascript
// In your GamePage.js, add transformer to method options
const methodOptions = [
    {
        value: 'frequency',
        label: 'Frequency Analysis',
        description: 'Predicts the most frequently occurring digit in your sequence'
    },
    {
        value: 'pattern',
        label: 'Pattern Recognition', 
        description: 'Uses pattern rules: 000→0, 111→1, otherwise repeats the last digit'
    },
    {
        value: 'transformer',
        label: 'AI Transformer',
        description: 'Advanced AI model trained on real user data'
    }
];
```

## Retraining Your Model

As you collect more user data, you can retrain your model:

### Manual Retraining

```bash
# Retrain with all new data
python -m app.train_transformer --epochs 50 --model-name retrained_model

# Retrain with only high-quality data
python -m app.train_transformer --min-unpredictability 0.5 --epochs 100
```

### Programmatic Retraining

```python
from app.transformer_integration import retrain_model_from_database

# Retrain the model
success = retrain_model_from_database(
    min_unpredictability=0.4,
    epochs=50,
    output_path="models/binary_transformer_v2.pth"
)
```

### Automated Retraining

You could set up a scheduled task to retrain periodically:

```python
# Add to your FastAPI app
from fastapi import BackgroundTasks
from app.transformer_integration import async_retrain_model

@app.post("/admin/retrain-model/")
async def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(async_retrain_model, epochs=50)
    return {"message": "Model retraining started"}
```

## Performance Tips

### For Training
- Start with smaller models (d_model=64, num_layers=3) for quick experiments
- Use GPU if available: `--device cuda`
- Increase batch size if you have enough memory
- Use data augmentation for small datasets: `--augment 3`

### For Inference
- Models run faster on CPU for single predictions
- Use temperature < 1.0 for more confident predictions
- Use temperature > 1.0 for more random/creative predictions

### Model Size vs Performance
- Larger models (d_model=256, num_layers=8) can learn more complex patterns
- But they require more training data and computational resources
- Start small and scale up based on your data size

## Troubleshooting

### Common Issues

1. **"No sequences found"**
   - Check your database has submission data
   - Lower the `--min-unpredictability` threshold
   - Verify database connection string

2. **"Model not found"**
   - Check the model path is correct
   - Ensure training completed successfully
   - Look for `.pth` files in your models directory

3. **Poor prediction performance**
   - Try training for more epochs
   - Increase model size (d_model, num_layers)
   - Use more training data
   - Check data quality (filter by unpredictability)

4. **Out of memory errors**
   - Reduce batch size: `--batch-size 8`
   - Reduce model size: `--d-model 64 --num-layers 3`
   - Use CPU instead of GPU: `--device cpu`

### Monitoring Training

The training script provides detailed output:
- Loss curves (training and validation)
- Model checkpoints saved automatically
- Configuration saved alongside model

### Model Comparison

Use the benchmark feature to compare methods:

```bash
python -m app.predict_transformer models/binary_transformer.pth --benchmark test_data.json
```

This will show you how the transformer performs compared to your existing frequency and pattern methods.

## Advanced Usage

### Custom Model Architecture

You can experiment with different architectures:

```python
# In transformer_predictor.py
model = BinaryTransformer(
    d_model=512,        # Larger model
    nhead=16,           # More attention heads  
    num_layers=12,      # Deeper network
    dim_feedforward=2048,  # Larger feedforward
    max_seq_length=200, # Longer sequences
    dropout=0.2         # More regularization
)
```

### Ensemble Methods

Combine multiple models for better performance:

```python
def ensemble_predict(sequence, models):
    predictions = []
    confidences = []
    
    for model in models:
        pred, conf = model.predict_next(sequence)
        predictions.append(int(pred))
        confidences.append(conf)
    
    # Weighted average
    weights = np.array(confidences)
    avg_pred = np.average(predictions, weights=weights)
    
    return str(int(round(avg_pred))), np.mean(confidences)
```

### Fine-tuning

Start with a pre-trained model and fine-tune on new data:

```python
# Load existing model
model = BinaryTransformerTrainer.load_model("models/base_model.pth")

# Create trainer with lower learning rate
trainer = BinaryTransformerTrainer(
    model=model,
    train_dataset=new_dataset,
    learning_rate=1e-5  # Lower learning rate for fine-tuning
)

# Train for fewer epochs
trainer.train(num_epochs=20)
```

## Next Steps

1. **Collect More Data**: The more user submissions you have, the better your model will perform
2. **Experiment with Architecture**: Try different model sizes and configurations
3. **Add More Features**: Consider incorporating user metadata (method used, performance, etc.)
4. **Deploy Updates**: Set up automated retraining as your dataset grows
5. **Monitor Performance**: Track how well the transformer performs compared to simple methods

The transformer model should significantly outperform your current frequency and pattern methods, especially as you collect more diverse user data!

