from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class Message(BaseModel):
    role: str
    content: str

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str = "hrm-27m"
    messages: List[Message]
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict]] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
