#!/bin/bash
# Grid Search Script for Transformer Training
# Tests different epoch counts with fixed architecture parameters

echo "GRID SEARCH: Transformer Training"
echo "=================================="
echo "Fixed parameters:"
echo "  batch_size: 4"
echo "  learning_rate: 0.001"
echo "  d_model: 16"
echo "  nhead: 4"
echo "  num_layers: 1"
echo "  dim_feedforward: 64"
echo ""
echo "Variable parameter: epochs (100 to 10,000)"
echo "=================================="

# Define epoch values (10 cuts between 100 and 10,000)
epochs=(100 1200 2300 3400 4500 5600 6700 7800 8900 10000)

echo "Epoch values to test: ${epochs[*]}"
echo ""

# Results array
declare -a results

for i in "${!epochs[@]}"; do
    epochs_val=${epochs[$i]}
    experiment_num=$((i + 1))
    
    echo ""
    echo "============================================================"
    echo "EXPERIMENT $experiment_num/10: $epochs_val epochs"
    echo "============================================================"
    
    # Run training
    start_time=$(date +%s)
    
    docker-compose -f docker-compose.yml --profile training-gpu run --rm transformer-training-gpu \
        python -m src.train_transformer \
        --device cuda \
        --epochs $epochs_val \
        --batch-size 4 \
        --learning-rate 0.001 \
        --d-model 16 \
        --nhead 4 \
        --num-layers 1 \
        --dim-feedforward 64 \
        --model-name grid_search_${epochs_val}_epochs \
        --output-dir /app/models \
        --print-every $((epochs_val / 20))
    
    end_time=$(date +%s)
    training_time=$((end_time - start_time))
    
    # Extract validation accuracy from the last line of output
    # This is a simplified approach - in practice you'd parse the actual output
    echo "Training completed in ${training_time} seconds"
    
    # Store result
    results[$i]="$epochs_val,$training_time"
    
    echo "Experiment $experiment_num completed"
done

echo ""
echo "============================================================"
echo "GRID SEARCH COMPLETED"
echo "============================================================"
echo ""
echo "Summary:"
echo "Epochs | Training Time (s)"
echo "-------|------------------"

for i in "${!results[@]}"; do
    IFS=',' read -r epochs_val training_time <<< "${results[$i]}"
    printf "%6d | %15d\n" "$epochs_val" "$training_time"
done

echo ""
echo "All models saved in models/ directory"
echo "Check individual model performance by testing them manually"
