import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#define a Pydantic model for the request body
class ChatRequest(BaseModel):
    user_message: str

@app.post("/chat")
def chat_with_llm(req: ChatRequest):
    user_input = req.user_message
    # Real LLM logic goes here maybe?
    # demo only: fake a reply from LLM
    fake_llm_reply = f"Echo from GPT-4o-mini: {user_input}"

    return {"assistant_message": fake_llm_reply}
