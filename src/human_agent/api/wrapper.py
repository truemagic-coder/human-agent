import torch
import json
import re
from typing import List, Dict, Any, Optional, Union
from ..core.model import HierarchicalReasoningModel
from ..core.tokenizer import Tokenizer
from ..functions.registry import FunctionRegistry
from ..functions.builtin import register_builtin_functions

class HRMChatWrapper:
    """OpenAI-compatible wrapper for HRM model with function calling"""
    
    def __init__(
        self,
        model: HierarchicalReasoningModel,
        tokenizer: Tokenizer,
        function_registry: FunctionRegistry = None,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.function_registry = function_registry or FunctionRegistry()
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Register builtin functions if no registry provided
        if function_registry is None:
            register_builtin_functions(self.function_registry)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single string"""
        formatted = ""
        for msg in messages:
            # Handle both dict and Pydantic Message objects
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
                function_call = getattr(msg, 'function_call', None)
            else:
                role = msg.get("role")
                content = msg.get("content")
                function_call = msg.get("function_call")
            
            # Handle function call messages
            if role == "assistant" and function_call:
                if hasattr(function_call, 'name'):
                    func_name = function_call.name
                    func_args = function_call.arguments
                else:
                    func_name = function_call['name']
                    func_args = function_call['arguments']
                formatted += f"<assistant><function_call>{func_name}({func_args})</function_call></assistant>"
            elif role == "function":
                formatted += f"<function_result>{content}</function_result>"
            else:
                formatted += f"<{role}>{content}</{role}>"
        return formatted

    def _parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse function call from generated text using multiple strategies"""
        
        # Strategy 1: Explicit function call syntax
        explicit_call = self._parse_explicit_function_call(text)
        if explicit_call:
            return explicit_call
        
        # Strategy 2: Natural language intent detection (for untrained models)
        intent_call = self._detect_function_intent(text)
        if intent_call:
            return intent_call
        
        return None
    
    def _parse_explicit_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse explicit function call syntax"""
        pattern = r'<function_call>\s*(\w+)\s*\((.*?)\)\s*</function_call>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            func_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Skip generic template responses
            if func_name == "function_name" or "arg1=value1" in args_str:
                return None
            
            # Validate function exists
            available_funcs = [f["name"] for f in self.function_registry.get_schemas()]
            if func_name not in available_funcs:
                return None
                
            args = self._parse_arguments(args_str, func_name)
            return {"name": func_name, "arguments": args}
        
        return None
    
    def _parse_arguments(self, args_str: str, func_name: str) -> Dict[str, Any]:
        """Parse function arguments"""
        if not args_str.strip():
            return {}
        
        args = {}
        
        # Try JSON parsing first
        try:
            if args_str.strip().startswith('{'):
                return json.loads(args_str)
        except Exception:
            pass
        
        # Parse key=value pairs
        try:
            for arg in args_str.split(','):
                arg = arg.strip()
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    args[key] = value
                elif arg:
                    # Positional argument - map to first parameter
                    schema = next((s for s in self.function_registry.get_schemas() if s["name"] == func_name), None)
                    if schema and schema["parameters"]["properties"]:
                        first_param = list(schema["parameters"]["properties"].keys())[0]
                        args[first_param] = arg.strip('"\'')
        except Exception:
            pass
        
        return args
    
    def _detect_function_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect function intent from natural language"""
        
        # Math expressions
        math_patterns = [
            r'(\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*[\d\+\-\*\/\^\(\)\s\.]+)',
            r'what(?:\'s|\s+is)\s+(\d+(?:\s*[\+\-\*\/\^]\s*\d+)+)',
            r'calculate\s+(.+?)(?:\?|$)',
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                expression = match.group(1) if len(match.groups()) == 1 else match.group(0)
                # Clean up
                expression = re.sub(r'what(?:\'s|\s+is)\s+', '', expression, flags=re.IGNORECASE)
                expression = re.sub(r'calculate\s+', '', expression, flags=re.IGNORECASE)
                expression = expression.strip('?.,!').strip()
                
                if any(op in expression for op in ['+', '-', '*', '/', '^']):
                    return {"name": "calculate", "arguments": {"expression": expression}}
        
        # Weather queries
        weather_patterns = [
            r'weather.*?(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$|\.)',
            r'(?:what(?:\'s|\s+is)|how(?:\'s|\s+is))\s+(?:the\s+)?weather.*?(?:in|at|for)\s+([a-zA-Z\s]+?)(?:\?|$|\.)',
        ]
        
        for pattern in weather_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                return {"name": "get_weather", "arguments": {"location": location}}
        
        # Time queries
        if re.search(r'(?:what(?:\'s|\s+is)|current)\s+(?:the\s+)?(?:time|date)|time\s+is\s+it', text, re.IGNORECASE):
            return {"name": "get_current_time", "arguments": {}}
        
        return None

    def _generate_response(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        stop_tokens: List[str] = None
    ) -> str:
        """Generate response using HRM model"""
        stop_tokens = stop_tokens or ["</assistant>", "<user>", "<function_result>", "<system>"]
        
        # Check for simple patterns that should have predefined responses
        user_input = ""
        if "<user>" in prompt:
            user_input = prompt.split("<user>")[-1].split("</user>")[0].lower().strip()
        
        # Handle common greetings with predefined responses
        greeting_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "hey": "Hey! How can I assist you?",
            "how are you": "I'm doing well, thank you! How are you?",
            "good morning": "Good morning! How can I help you?",
            "good afternoon": "Good afternoon! What can I do for you?",
            "good evening": "Good evening! How may I assist you?",
        }
        
        for greeting, response in greeting_responses.items():
            if greeting in user_input:
                return response
        
        # For untrained models, limit the prompt to avoid repetition
        input_ids = self.tokenizer.encode(prompt[-500:])  # Take last 500 chars
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Ensure tensor is on correct device
                current_tensor = torch.tensor([generated_ids], device=self.device)
                
                # Limit sequence length to prevent memory issues
                if len(generated_ids) > 512:
                    current_tensor = current_tensor[:, -512:]
                    generated_ids = generated_ids[-512:]
                
                result = self.model(current_tensor)
                logits = result['outputs'][0, -1]
                
                # Apply temperature and sample
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = logits.argmax().item()
                
                generated_ids.append(next_token)
                
                # Decode current text to check for stop conditions
                current_text = self.tokenizer.decode(generated_ids)
                
                # Stop if we hit a stop token
                for stop in stop_tokens:
                    if stop in current_text:
                        break
                else:
                    # Stop if EOS token
                    if next_token == self.tokenizer.eos_token_id:
                        break
                    continue
                break
        
        # Extract just the new response
        full_text = self.tokenizer.decode(generated_ids)
        
        # Remove the input prompt from the response
        response = full_text
        for stop in stop_tokens:
            if stop in response:
                response = response[:response.index(stop)]
        
        return response.strip()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI-compatible chat completion"""
        
        # Convert Pydantic objects to dicts for easier handling
        if messages and hasattr(messages[0], 'role'):
            messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Check if this is a function call continuation
        last_message = messages[-1] if messages else None
        if last_message and last_message.get("role") == "function":
            # This is a function result being fed back
            return self._handle_function_result(messages, max_tokens, temperature)
        
        # Check for direct function call intent in user message
        user_message = last_message.get("content", "") if last_message else ""
        function_call_parsed = self._parse_function_call(user_message)
        
        if function_call_parsed:
            # Direct function call detected
            func_result = self.function_registry.call_function(
                function_call_parsed["name"],
                function_call_parsed["arguments"]
            )
            
            return {
                "id": f"chatcmpl-hrm-{abs(hash(str(messages))) % 10000}",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "hrm-27m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",  # Changed from None to empty string
                            "function_call": {
                                "name": function_call_parsed["name"],
                                "arguments": json.dumps(function_call_parsed["arguments"])
                            }
                        },
                        "finish_reason": "function_call"
                    }
                ],
                "function_result": func_result
            }
        
        # Regular chat - generate simple response without complex system prompts
        conversation = self._format_messages(messages)
        conversation += "<assistant>"
        
        response_text = self._generate_response(
            conversation,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Clean up response
        response_text = response_text.replace("<assistant>", "").strip()
        
        # If response is too short or looks like noise, provide a more helpful fallback
        if len(response_text) < 5 or response_text.count(" ") < 2:
            # Check what the user asked for better fallback
            user_content = last_message.get("content", "").lower() if last_message else ""
            
            if any(word in user_content for word in ["hello", "hi", "hey"]):
                response_text = "Hello! I'm an AI assistant. How can I help you today?"
            elif "how are you" in user_content:
                response_text = "I'm doing well, thank you for asking! How can I assist you?"
            elif any(word in user_content for word in ["thank", "thanks"]):
                response_text = "You're welcome! Is there anything else I can help you with?"
            elif any(word in user_content for word in ["bye", "goodbye"]):
                response_text = "Goodbye! Have a great day!"
            else:
                response_text = "I can help with calculations, weather information, and general questions. What would you like to know?"
        
        return {
            "id": f"chatcmpl-hrm-{abs(hash(str(messages))) % 10000}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "hrm-27m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(conversation)),
                "completion_tokens": len(self.tokenizer.encode(response_text)),
                "total_tokens": len(self.tokenizer.encode(conversation + response_text))
            }
        }
    
    def _handle_function_result(self, messages: List[Dict], max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Handle function result and generate final response"""
        conversation = self._format_messages(messages)
        conversation += "<assistant>"
        
        response_text = self._generate_response(
            conversation,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        response_text = response_text.replace("<assistant>", "").strip()
        
        if len(response_text) < 5:
            # Extract function result and format it nicely
            func_result_msg = next((m for m in reversed(messages) if m.get("role") == "function"), None)
            if func_result_msg:
                response_text = f"Here's the result: {func_result_msg['content']}"
        
        return {
            "id": f"chatcmpl-hrm-{abs(hash(str(messages))) % 10000}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "hrm-27m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(conversation)),
                "completion_tokens": len(self.tokenizer.encode(response_text)),
                "total_tokens": len(self.tokenizer.encode(conversation + response_text))
            }
        }

def create_chat_model(
    vocab_size: int = 10000,
    model_kwargs: Dict = None,
    device: str = "cpu"
) -> HRMChatWrapper:
    """Create a chat-compatible HRM model"""
    model_kwargs = model_kwargs or {}
    
    # Create tokenizer
    tokenizer = Tokenizer(vocab_size)
    
    # Create model with updated vocab size
    actual_vocab_size = len(tokenizer.vocab)
    model = HierarchicalReasoningModel(
        vocab_size=actual_vocab_size,
        **model_kwargs
    )
    
    # Move to device
    model = model.to(device)
    
    # Create wrapper
    return HRMChatWrapper(model, tokenizer, device=device)
