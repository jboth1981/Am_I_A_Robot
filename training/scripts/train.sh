#!/bin/bash
# Training script for running transformer training in Docker
# This script provides easy commands for common training scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create necessary directories
mkdir -p models training_data

echo -e "${GREEN}Binary Sequence Transformer Training${NC}"
echo "======================================"

# Function to check if database is running
check_database() {
    echo -e "${YELLOW}Checking database connection...${NC}"
    if docker-compose -f docker-compose.local.yml ps database | grep -q "Up"; then
        echo -e "${GREEN}✓ Database is running${NC}"
    else
        echo -e "${RED}✗ Database is not running. Starting database...${NC}"
        docker-compose -f docker-compose.local.yml up -d database
        echo "Waiting for database to be ready..."
        sleep 10
    fi
}

# Function to build training container
build_container() {
    echo -e "${YELLOW}Building training container...${NC}"
    docker-compose -f docker-compose.yml build transformer-training
    echo -e "${GREEN}✓ Training container built${NC}"
}

# Function to extract data
extract_data() {
    echo -e "${YELLOW}Extracting training data from database...${NC}"
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.data_extractor --output /app/data/training_data.json --min-unpredictability 0.2
    echo -e "${GREEN}✓ Training data extracted${NC}"
}

# Function to show database stats
show_stats() {
    echo -e "${YELLOW}Database statistics:${NC}"
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.data_extractor --stats-only
}

# Function to train model
train_model() {
    local epochs=${1:-50}
    local batch_size=${2:-16}
    local model_name=${3:-binary_transformer}
    
    echo -e "${YELLOW}Training transformer model...${NC}"
    echo "Epochs: $epochs, Batch size: $batch_size, Model name: $model_name"
    
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.train_transformer \
        --epochs $epochs \
        --batch-size $batch_size \
        --model-name $model_name \
        --output-dir /app/models \
        --print-every 10
    
    echo -e "${GREEN}✓ Model training completed${NC}"
}

# Function to deploy model to backend
deploy_model() {
    local model_name=${1:-binary_transformer}
    
    echo -e "${YELLOW}Deploying model to backend...${NC}"
    
    # Check if model exists
    if [ ! -f "models/${model_name}_final.pth" ]; then
        echo -e "${RED}✗ Model not found: models/${model_name}_final.pth${NC}"
        return 1
    fi
    
    # Copy to backend models directory
    cp "models/${model_name}_final.pth" "../models/"
    echo -e "${GREEN}✓ Model deployed to backend/models/${NC}"
    
    # Optionally restart backend
    echo -e "${YELLOW}Restarting backend to load new model...${NC}"
    cd .. && docker-compose -f docker-compose.local.yml restart backend
    cd training
    
    echo -e "${GREEN}✓ Backend restarted with new model${NC}"
}

# Function to test model
test_model() {
    local model_path=${1:-models/binary_transformer.pth}
    
    echo -e "${YELLOW}Testing model: $model_path${NC}"
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.predict_transformer /app/$model_path --sequence "010101"
}

# Function to run interactive mode
interactive_mode() {
    local model_path=${1:-models/binary_transformer.pth}
    
    echo -e "${YELLOW}Starting interactive mode with model: $model_path${NC}"
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.predict_transformer /app/$model_path --interactive
}

# Function to benchmark model
benchmark_model() {
    local model_path=${1:-models/binary_transformer.pth}
    
    echo -e "${YELLOW}Benchmarking model: $model_path${NC}"
    docker-compose -f docker-compose.yml run --rm transformer-training \
        python -m src.predict_transformer /app/$model_path --benchmark /app/data/training_data.json
}

# Main script logic
case "${1:-help}" in
    "setup")
        check_database
        build_container
        extract_data
        echo -e "${GREEN}✓ Setup completed! Ready for training.${NC}"
        ;;
    
    "stats")
        check_database
        show_stats
        ;;
    
    "extract")
        check_database
        extract_data
        ;;
    
    "train")
        check_database
        train_model $2 $3 $4
        ;;
    
    "quick-train")
        check_database
        echo -e "${YELLOW}Quick training (20 epochs, small model)${NC}"
        docker-compose -f docker-compose.yml run --rm transformer-training \
            python -m src.train_transformer \
            --epochs 20 \
            --batch-size 8 \
            --d-model 64 \
            --num-layers 3 \
            --model-name quick_model \
            --output-dir /app/models
        ;;
    
    "duplicate-train")
        check_database
        local duplicate_factor=${2:-100}
        echo -e "${YELLOW}Training with ${duplicate_factor}x duplication${NC}"
        docker-compose -f docker-compose.yml run --rm transformer-training \
            python -m src.train_transformer \
            --epochs 60 \
            --batch-size 4 \
            --learning-rate 0.005 \
            --duplicate $duplicate_factor \
            --model-name duplicated_model \
            --output-dir /app/models
        ;;
    
    "advanced-train")
        check_database
        echo -e "${YELLOW}Advanced training (100 epochs, large model)${NC}"
        docker-compose -f docker-compose.yml run --rm transformer-training \
            python -m src.train_transformer \
            --epochs 100 \
            --batch-size 32 \
            --d-model 256 \
            --num-layers 8 \
            --model-name advanced_model \
            --output-dir /app/models
        ;;
    
    "test")
        test_model $2
        ;;
    
    "interactive")
        interactive_mode $2
        ;;
    
    "benchmark")
        benchmark_model $2
        ;;
    
    "gpu-train")
        check_database
        echo -e "${YELLOW}GPU training (requires nvidia-docker)${NC}"
        docker-compose -f docker-compose.yml --profile training-gpu run --rm transformer-training-gpu \
            python3 -m src.train_transformer \
            --device cuda \
            --epochs ${2:-100} \
            --batch-size ${3:-64} \
            --model-name gpu_model \
            --output-dir /app/models
        ;;
    
    "gpu-duplicate-train")
        check_database
        local duplicate_factor=${2:-50}
        echo -e "${YELLOW}GPU training with ${duplicate_factor}x duplication${NC}"
        docker-compose -f docker-compose.yml --profile training-gpu run --rm transformer-training-gpu \
            python3 -m src.train_transformer \
            --device cuda \
            --epochs 60 \
            --batch-size 8 \
            --learning-rate 0.005 \
            --duplicate $duplicate_factor \
            --model-name gpu_duplicated_model \
            --output-dir /app/models
        ;;
    
    "deploy")
        deploy_model $2
        ;;
    
    "clean")
        echo -e "${YELLOW}Cleaning up training containers and images...${NC}"
        docker-compose -f docker-compose.yml down
        docker image prune -f
        echo -e "${GREEN}✓ Cleanup completed${NC}"
        ;;
    
    "help"|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup                 - Initial setup (build container, extract data)"
        echo "  stats                 - Show database statistics"
        echo "  extract               - Extract training data from database"
        echo "  train [epochs] [batch] [name] - Train model with custom parameters"
        echo "  quick-train           - Quick training (20 epochs, small model)"
        echo "  duplicate-train [N]   - Training with Nx duplication (default: 100x)"
        echo "  deploy [model_name]   - Deploy trained model to backend"
        echo "  advanced-train        - Advanced training (100 epochs, large model)"
        echo "  gpu-train [epochs] [batch] - GPU training (requires nvidia-docker)"
        echo "  gpu-duplicate-train [N] - GPU training with Nx duplication (default: 50x)"
        echo "  test [model_path]     - Test a trained model"
        echo "  interactive [model]   - Interactive prediction mode"
        echo "  benchmark [model]     - Benchmark model against simple methods"
        echo "  clean                 - Clean up containers and images"
        echo "  help                  - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 setup              # Initial setup"
        echo "  $0 quick-train        # Quick training session"
        echo "  $0 duplicate-train 100 # Training with 100x duplication"
        echo "  $0 gpu-duplicate-train 50 # GPU training with 50x duplication"
        echo "  $0 train 50 16 my_model  # Custom training"
        echo "  $0 test models/my_model.pth  # Test specific model"
        echo "  $0 interactive        # Interactive mode with default model"
        ;;
esac
