from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Pre-populated in-memory database for user management
users_db = {
    "admin": "password123",  # Admin user
    "test_user": "testpassword"    # Another example user
}

# Request models
class ChatRequest(BaseModel):
    user_message: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Chat endpoint
@app.post("/chat")
def chat_with_llm(req: ChatRequest):
    user_input = req.user_message
    fake_llm_reply = f"Echo from GPT-4o-mini: {user_input}"
    return {"assistant_message": fake_llm_reply}

# Registration endpoint
@app.post("/register")
def register_user(req: RegisterRequest):
    if req.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists.")
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    users_db[req.username] = req.password
    return {"message": f"User '{req.username}' registered successfully."}

# Login endpoint
@app.post("/login")
def login_user(req: LoginRequest):
    if req.username not in users_db or users_db[req.username] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"message": f"Welcome back, {req.username}!"}
