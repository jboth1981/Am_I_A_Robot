from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import timedelta, datetime
import os

# Import our custom modules
from app.models import User, PasswordResetToken, Submission, Base
from app.database import engine, get_db
from app.schemas import UserCreate, UserLogin, UserResponse, Token, PasswordResetRequest, PasswordReset, SubmissionCreate, SubmissionResponse
from app.auth import create_access_token, verify_token, ACCESS_TOKEN_EXPIRE_MINUTES

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Am I A Robot API", version="1.0.0")

# Security
security = HTTPBearer()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Existing prediction models and functions (unchanged)
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

# Authentication endpoints
@app.post("/register/", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account"""
    # Check if user already exists
    db_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = User.hash_password(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/login/", response_model=Token)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    """Login user and return JWT token"""
    # Find user
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user or not db_user.verify_password(user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me/", response_model=UserResponse)
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current user information from JWT token"""
    username = verify_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Helper function to get current user from JWT token"""
    username = verify_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

@app.post("/request-password-reset/")
async def request_password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    """Request a password reset token for the given email"""
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a reset link has been sent"}
    
    # Generate reset token
    token = PasswordResetToken.generate_token()
    expires_at = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
    
    # Create reset token record
    reset_token = PasswordResetToken(
        user_id=user.id,
        token=token,
        expires_at=expires_at
    )
    
    db.add(reset_token)
    db.commit()
    
    # Send email with reset link
    reset_url = f"https://amiarobot.ca/reset-password?token={token}"
    
    try:
        from app.email import send_password_reset_email
        await send_password_reset_email(user.email, reset_url)
        print(f"Password reset email sent to {user.email}")
    except Exception as e:
        print(f"Failed to send password reset email: {e}")
        # Still return success to not reveal if email exists
        pass
    
    return {"message": "If the email exists, a reset link has been sent"}

@app.post("/reset-password/")
def reset_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    """Reset password using a valid token"""
    # Find the reset token
    reset_token = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == reset_data.token,
        PasswordResetToken.used == False,
        PasswordResetToken.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_token:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token"
        )
    
    # Get the user
    user = db.query(User).filter(User.id == reset_token.user_id).first()
    if not user:
        raise HTTPException(
            status_code=400,
            detail="User not found"
        )
    
    # Update password
    user.hashed_password = User.hash_password(reset_data.new_password)
    
    # Mark token as used
    reset_token.used = True
    
    db.commit()
    
    return {"message": "Password reset successfully"}

# Optimized prediction endpoint - no database dependencies
@app.post("/predict/")
def predict_next(data: InputData):
    """Fast prediction endpoint - no database or auth dependencies"""
    history = data.history
    method = data.method
    
    if method == "frequency":
        prediction = predict_frequency(history)
    elif method == "pattern":
        prediction = predict_pattern(history)
    else:
        # Default to frequency method
        prediction = predict_frequency(history)
    
    # Removed debug print to improve performance
    return {"prediction": prediction}

# Submission endpoints
@app.post("/submissions/", response_model=SubmissionResponse)
def create_submission(
    submission_data: SubmissionCreate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Save a completed submission to the database"""
    
    # Create new submission
    db_submission = Submission(
        user_id=current_user.id,
        binary_sequence=submission_data.binary_sequence,
        prediction_method=submission_data.prediction_method,
        total_predictions=submission_data.total_predictions,
        correct_predictions=submission_data.correct_predictions,
        accuracy_percentage=submission_data.accuracy_percentage,
        is_human_result=submission_data.is_human_result
    )
    
    db.add(db_submission)
    db.commit()
    db.refresh(db_submission)
    
    return db_submission

@app.get("/submissions/", response_model=List[SubmissionResponse])
def get_user_submissions(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """Get the current user's submission history"""
    submissions = db.query(Submission)\
        .filter(Submission.user_id == current_user.id)\
        .order_by(Submission.completed_at.desc())\
        .limit(limit)\
        .all()
    
    return submissions