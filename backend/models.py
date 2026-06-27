from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    history: List[str] = []
    session_id: str

class ChatResponse(BaseModel):
    status: str
    answer: str
    session_id: str