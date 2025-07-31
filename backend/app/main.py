from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    history: str  # e.g., "01101"
    method: str = "frequency"  # "frequency" or "pattern"

def predict_frequency(history: str) -> str:
    """Simple frequency-based prediction: predict most frequent digit"""
    if not history:
        return "0"
    
    count_0 = history.count("0")
    count_1 = history.count("1")
    
    prediction = "1" if count_1 > count_0 else "0"
    print(f"DEBUG: Frequency analysis - 0s={count_0}, 1s={count_1}, prediction={prediction}")
    
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

@app.post("/predict/")
def predict_next(data: InputData):
    history = data.history
    method = data.method
    
    print(f"DEBUG: Received history='{history}', method='{method}'")
    
    if method == "frequency":
        prediction = predict_frequency(history)
    elif method == "pattern":
        prediction = predict_pattern(history)
    else:
        # Default to frequency method
        prediction = predict_frequency(history)
    
    print(f"DEBUG: Returning prediction='{prediction}'")
    return {"prediction": prediction}

