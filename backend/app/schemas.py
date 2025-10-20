from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """Base user schema with common fields"""
    username: str
    email: str

class UserCreate(UserBase):
    """Schema for user registration requests"""
    password: str

class UserLogin(BaseModel):
    """Schema for user login requests"""
    username: str
    password: str

class UserResponse(UserBase):
    """Schema for user data returned by API (excludes sensitive data)"""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for JWT token responses"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for token payload data"""
    username: Optional[str] = None

class PasswordResetRequest(BaseModel):
    """Schema for password reset request"""
    email: EmailStr

class PasswordReset(BaseModel):
    """Schema for password reset with token"""
    token: str
    new_password: str

class SubmissionCreate(BaseModel):
    """Schema for creating a new submission"""
    binary_sequence: str
    prediction_method: str  # 'frequency', 'pattern', or 'transformer'
    total_predictions: int
    correct_predictions: int
    accuracy_percentage: float
    is_human_result: bool
    predictions_json: Optional[str] = None  # JSON string of all predictions made
    confidence_scores_json: Optional[str] = None  # JSON string of confidence scores
    average_confidence: Optional[float] = None  # Average confidence across all predictions
    session_id: Optional[str] = None  # For guest users

class SubmissionResponse(BaseModel):
    """Schema for submission data returned by API"""
    id: int
    user_id: Optional[int]  # NULL for guest users
    binary_sequence: str
    prediction_method: str
    total_predictions: int
    correct_predictions: int
    accuracy_percentage: float
    is_human_result: bool
    completed_at: datetime
    predictions_json: Optional[str] = None
    confidence_scores_json: Optional[str] = None
    average_confidence: Optional[float] = None
    session_id: Optional[str] = None  # For guest users
    
    class Config:
        from_attributes = True