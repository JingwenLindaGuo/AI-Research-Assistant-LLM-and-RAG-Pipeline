from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
import datetime
from typing import Optional

# FastAPI app initialization
app = FastAPI()

# Secret key for JWT token encoding
SECRET_KEY = "mysecretkey" # will replace with a more secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Password hashing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user database (replace with a real database in production)
users_db = {
    "admin": {"username": "admin", "password": pwd_context.hash("password123")},
    "test_user": {"username": "test_user", "password": pwd_context.hash("testpassword")},
}

# Pydantic models
class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str
    
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatRequest(BaseModel):
    user_message: str

# Function to hash passwords
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Function to verify passwords
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Function to create JWT tokens
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (expires_delta if expires_delta else datetime.timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Function to authenticate user
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

# Function to decode JWT token and verify authentication
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# User registration endpoint
@app.post("/register")
def register_user(req: RegisterRequest):
    if req.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists.")
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    
    hashed_password = hash_password(req.password)
    users_db[req.username] = {"username": req.username, "password": hashed_password}
    return {"message": f"User '{req.username}' registered successfully."}

# Login endpoint (returns JWT token)
@app.post("/login")
def login_user(req: LoginRequest):
    user = users_db.get(req.username)  # Fetch user from database (replace with mongodb)
    if not user or not verify_password(req.password, user["password"]):  # Use bcrypt verification
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate JWT token
    access_token = create_access_token(data={"sub": req.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected chat endpoint (requires authentication)
@app.post("/chat")
def chat_with_llm(req: ChatRequest, username: str = Depends(get_current_user)):
    user_input = req.user_message
    fake_llm_reply = f"Echo from GPT-4o-mini: {user_input}"
    return {"assistant_message": fake_llm_reply}