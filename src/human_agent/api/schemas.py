from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str = "function"
    function: Function

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall
    
class Message(BaseModel):
    role: str
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None  # For backward compatibility
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For role="tool"

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None

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
    usage: Usage
    system_fingerprint: Optional[str] = None
