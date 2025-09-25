# Lightweight prediction server - no database dependencies
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.prediction import PredictionInput, get_prediction

# Create lightweight FastAPI app for predictions only
app = FastAPI(title="Am I A Robot - Prediction Service", version="1.0.0")

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Prediction service running"}

@app.post("/predict/")
def predict_next(data: PredictionInput):
    """Fast prediction endpoint - no database dependencies"""
    return get_prediction(data)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "prediction"}
