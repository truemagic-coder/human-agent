from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class Message(BaseModel):
    role: str
    content: Optional[str] = None  # Changed to Optional to allow None for function calls
    function_call: Optional[Dict[str, Any]] = None

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

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    function_result: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class FunctionCall(BaseModel):
    name: str
    arguments: str

class FunctionMessage(Message):
    name: Optional[str] = None

class AssistantMessage(Message):
    function_call: Optional[FunctionCall] = None
    