# PowerShell script for running transformer training in Docker
# This should work better than the batch file on Windows

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [int]$Epochs = 50,
    
    [Parameter(Position=2)]
    [int]$BatchSize = 16,
    
    [Parameter(Position=3)]
    [string]$ModelName = "binary_transformer"
)

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$Yellow = "`e[33m"
$Reset = "`e[0m"

Write-Host "${Green}Binary Sequence Transformer Training${Reset}"
Write-Host "======================================"

# Create necessary directories
if (!(Test-Path "models")) { New-Item -ItemType Directory -Path "models" }
if (!(Test-Path "training_data")) { New-Item -ItemType Directory -Path "training_data" }

function Check-Database {
    Write-Host "${Yellow}Checking database connection...${Reset}"
    
    try {
        $containers = docker ps --filter "name=postgres_local" --format "{{.Names}}" 2>$null
        if ($containers -contains "postgres_local") {
            $status = docker ps --filter "name=postgres_local" --format "{{.Status}}"
            if ($status -like "*Up*") {
                Write-Host "${Green}✓ Database is running${Reset}"
                return $true
            } else {
                Write-Host "${Yellow}Database container exists but not running. Starting...${Reset}"
                docker start postgres_local | Out-Null
                Start-Sleep 10
                return $true
            }
        } else {
            Write-Host "${Yellow}Database container not found. Starting with docker-compose...${Reset}"
            docker-compose -f docker-compose.local.yml up -d database | Out-Null
            Write-Host "Waiting for database to be ready..."
            Start-Sleep 15
            return $true
        }
    } catch {
        Write-Host "${Red}Error checking database: $_${Reset}"
        return $false
    }
}

function Build-Container {
    Write-Host "${Yellow}Building training container...${Reset}"
    try {
        docker-compose -f docker-compose.yml build transformer-training
        Write-Host "${Green}✓ Training container built${Reset}"
        return $true
    } catch {
        Write-Host "${Red}Error building container: $_${Reset}"
        return $false
    }
}

function Extract-Data {
    Write-Host "${Yellow}Extracting training data from database...${Reset}"
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.data_extractor --output /app/data/training_data.json --min-unpredictability 0.2
        Write-Host "${Green}✓ Training data extracted${Reset}"
        return $true
    } catch {
        Write-Host "${Red}Error extracting data: $_${Reset}"
        return $false
    }
}

function Show-Stats {
    Write-Host "${Yellow}Database statistics:${Reset}"
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.data_extractor --stats-only
    } catch {
        Write-Host "${Red}Error showing stats: $_${Reset}"
    }
}

function Train-Model {
    param($Epochs, $BatchSize, $ModelName)
    
    Write-Host "${Yellow}Training transformer model...${Reset}"
    Write-Host "Epochs: $Epochs, Batch size: $BatchSize, Model name: $ModelName"
    
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.train_transformer --epochs $Epochs --batch-size $BatchSize --model-name $ModelName --output-dir /app/models --print-every 10
        Write-Host "${Green}✓ Model training completed${Reset}"
        return $true
    } catch {
        Write-Host "${Red}Error training model: $_${Reset}"
        return $false
    }
}

function Test-Model {
    param($ModelPath = "models/binary_transformer.pth")
    
    Write-Host "${Yellow}Testing model: $ModelPath${Reset}"
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.predict_transformer /app/$ModelPath --sequence "010101"
    } catch {
        Write-Host "${Red}Error testing model: $_${Reset}"
    }
}

function Start-Interactive {
    param($ModelPath = "models/binary_transformer.pth")
    
    Write-Host "${Yellow}Starting interactive mode with model: $ModelPath${Reset}"
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.predict_transformer /app/$ModelPath --interactive
    } catch {
        Write-Host "${Red}Error starting interactive mode: $_${Reset}"
    }
}

function Benchmark-Model {
    param($ModelPath = "models/binary_transformer.pth")
    
    Write-Host "${Yellow}Benchmarking model: $ModelPath${Reset}"
    try {
        docker-compose -f docker-compose.yml run --rm transformer-training python -m src.predict_transformer /app/$ModelPath --benchmark /app/data/training_data.json
    } catch {
        Write-Host "${Red}Error benchmarking model: $_${Reset}"
    }
}

# Main script logic
switch ($Command.ToLower()) {
    "setup" {
        if ((Check-Database) -and (Build-Container) -and (Extract-Data)) {
            Write-Host "${Green}✓ Setup completed! Ready for training.${Reset}"
        } else {
            Write-Host "${Red}Setup failed. Check the errors above.${Reset}"
        }
    }
    
    "stats" {
        if (Check-Database) {
            Show-Stats
        }
    }
    
    "extract" {
        if (Check-Database) {
            Extract-Data
        }
    }
    
    "train" {
        if (Check-Database) {
            Train-Model $Epochs $BatchSize $ModelName
        }
    }
    
    "quick-train" {
        if (Check-Database) {
            Write-Host "${Yellow}Quick training (20 epochs, small model)${Reset}"
            docker-compose -f docker-compose.yml run --rm transformer-training python -m src.train_transformer --epochs 20 --batch-size 8 --d-model 64 --num-layers 3 --model-name quick_model --output-dir /app/models
        }
    }
    
    "advanced-train" {
        if (Check-Database) {
            Write-Host "${Yellow}Advanced training (100 epochs, large model)${Reset}"
            docker-compose -f docker-compose.yml run --rm transformer-training python -m src.train_transformer --epochs 100 --batch-size 32 --d-model 256 --num-layers 8 --model-name advanced_model --output-dir /app/models
        }
    }
    
    "test" {
        Test-Model $ModelName
    }
    
    "interactive" {
        Start-Interactive $ModelName
    }
    
    "benchmark" {
        Benchmark-Model $ModelName
    }
    
    "gpu-train" {
        if (Check-Database) {
            Write-Host "${Yellow}GPU training (requires nvidia-docker)${Reset}"
            docker-compose -f docker-compose.yml --profile training-gpu run --rm transformer-training-gpu python -m src.train_transformer --device cuda --epochs $Epochs --batch-size $BatchSize --model-name gpu_model --output-dir /app/models
        }
    }
    
    "clean" {
        Write-Host "${Yellow}Cleaning up training containers and images...${Reset}"
        docker-compose -f docker-compose.yml down
        docker image prune -f
        Write-Host "${Green}✓ Cleanup completed${Reset}"
    }
    
    default {
        Write-Host "Usage: .\train_in_docker.ps1 <command> [options]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  setup                 - Initial setup (build container, extract data)"
        Write-Host "  stats                 - Show database statistics"
        Write-Host "  extract               - Extract training data from database"
        Write-Host "  train [epochs] [batch] [name] - Train model with custom parameters"
        Write-Host "  quick-train           - Quick training (20 epochs, small model)"
        Write-Host "  advanced-train        - Advanced training (100 epochs, large model)"
        Write-Host "  gpu-train [epochs] [batch] - GPU training (requires nvidia-docker)"
        Write-Host "  test [model_path]     - Test a trained model"
        Write-Host "  interactive [model]   - Interactive prediction mode"
        Write-Host "  benchmark [model]     - Benchmark model against simple methods"
        Write-Host "  clean                 - Clean up containers and images"
        Write-Host "  help                  - Show this help"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\train_in_docker.ps1 setup              # Initial setup"
        Write-Host "  .\train_in_docker.ps1 quick-train        # Quick training session"
        Write-Host "  .\train_in_docker.ps1 train 50 16 my_model  # Custom training"
        Write-Host "  .\train_in_docker.ps1 test models/my_model.pth  # Test specific model"
        Write-Host "  .\train_in_docker.ps1 interactive        # Interactive mode with default model"
    }
}

