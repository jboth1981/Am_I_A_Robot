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

@app.post("/predict/")
def predict_next(data: InputData):
    history = data.history
    # Simple prediction: predict most frequent next digit
    if not history:
        return {"prediction": "0"}

    count_0 = history.count("0")
    count_1 = history.count("1")

    prediction = "1" if count_1 > count_0 else "0"
    return {"prediction": prediction}

