from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import secrets

# Security configuration
# SECRET_KEY must come from the environment. If it is missing or still set to the
# old insecure placeholder, generate a random ephemeral key instead of falling
# back to a publicly-known value (which would let anyone forge JWTs). An ephemeral
# key keeps the service secure but does not survive a restart, so production must
# set a stable SECRET_KEY in its environment.
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "your-secret-key-change-in-production":
    SECRET_KEY = secrets.token_urlsafe(48)
    print(
        "⚠ WARNING: SECRET_KEY is not set (or is the insecure default). "
        "Generated a random ephemeral key — existing sessions will be invalidated "
        "on every restart. Set a strong SECRET_KEY in the environment for production."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token for authenticated users"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify and decode a JWT token, return username if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None