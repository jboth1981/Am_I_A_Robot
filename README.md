# Am I A Robot?

A web application that tests human randomness against AI prediction algorithms. Can you generate truly random binary sequences that fool machine learning models?

## 🎯 Project Overview

This application presents users with a challenge: generate binary sequences (0s and 1s) that are unpredictable enough to be classified as "human" rather than "robot" by AI algorithms. It's a fascinating exploration of free will, randomness, and machine learning.

## 🚀 Quick Start

### For Local Development
See [README_LOCAL.md](README_LOCAL.md) for detailed local development setup with HTTPS and hot reload.

### For Production Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment instructions.

## 🏗️ Architecture

- **Frontend**: React with CRACO for hot reload over HTTPS
- **Backend**: FastAPI with PostgreSQL database
- **ML Models**: Transformer-based binary sequence prediction
- **Infrastructure**: Docker Compose with nginx reverse proxy
- **Authentication**: JWT-based user authentication

## 📁 Project Structure

```
Am_I_A_Robot/
├── frontend/                   # React frontend application
├── backend/                    # FastAPI backend with ML integration
│   ├── app/                    # Backend application code
│   │   ├── (API endpoints)
│   │   ├── data_extractor.py   # Training data extraction
│   │   ├── train_transformer.py # Model training
│   │   └── predict_transformer.py # Model inference
│   ├── Dockerfile.training     # Training container
│   └── Dockerfile.training-gpu # GPU training container
├── nginx/                      # Reverse proxy configuration
├── shared_model/              # Shared ML model components
├── training/                  # Training documentation & config
├── training_data/             # Training data storage
├── training_scripts/          # Training wrapper scripts
├── models/                    # Trained model files
├── certs/                     # SSL certificates (local dev)
├── docker-compose.local.yml   # Local development setup
├── docker-compose.production.yml # Production setup
├── docker-compose.training.yml # Training environment setup
├── README_LOCAL.md            # Local development guide
└── DEPLOYMENT.md              # Production deployment guide
```

## 🔧 Key Features

- **User Authentication**: Register, login, and manage user accounts
- **Binary Sequence Generation**: Interactive interface for creating sequences
- **AI Prediction**: Multiple prediction algorithms (frequency, pattern, transformer)
- **Human vs Robot Classification**: ML models determine if sequences are "human-like"
- **Real-time Feedback**: Immediate results and accuracy scoring
- **Data Analysis**: Track user performance and sequence patterns

## 🛠️ Technology Stack

- **Frontend**: React, CRACO, WebSocket for hot reload
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML**: PyTorch, Transformer models, Custom prediction algorithms
- **Infrastructure**: Docker, nginx, Let's Encrypt SSL
- **Authentication**: JWT tokens, bcrypt password hashing

## 📊 ML Models

The application includes multiple prediction algorithms:

1. **Frequency-based**: Predicts based on overall digit frequency
2. **Pattern-based**: Uses n-gram patterns to predict next digits
3. **Transformer-based**: Advanced neural network for sequence prediction

## 🔒 Security

- HTTPS everywhere (local dev with mkcert, production with Let's Encrypt)
- JWT-based authentication
- bcrypt password hashing (pinned to stable version)
- CORS properly configured
- Input validation and sanitization

## 🚀 Getting Started

1. **Local Development**: Follow [README_LOCAL.md](README_LOCAL.md)
2. **Production**: Follow [DEPLOYMENT.md](DEPLOYMENT.md)

## 🤝 Contributing

This project uses Git for version control and is designed to work seamlessly between local development and production environments.

## 📝 License

This project is for educational and research purposes exploring the nature of human randomness and AI prediction capabilities.
