import time
import torch
import json
import re
from typing import List, Dict, Any, Optional
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
        """Format messages into a single string for the model prompt."""
        formatted = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            function_call = msg.get("function_call")
            
            if role == "assistant" and function_call:
                func_name = function_call.get('name')
                func_args = function_call.get('arguments', '')
                if func_name:
                    formatted += f"<assistant><function_call>{func_name}({func_args})</function_call></assistant>"
            elif role == "function":
                formatted += f"<function_result>{content}</function_result>"
            elif content:
                formatted += f"<{role}>{content}</{role}>"
        return formatted

    def _parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse function call from generated text."""
        explicit_call = self._parse_explicit_function_call(text)
        if explicit_call:
            return explicit_call
        
        intent_call = self._detect_function_intent(text)
        if intent_call:
            return intent_call
        
        return None
    
    def _parse_explicit_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse <function_call>...</function_call> syntax."""
        pattern = r'<function_call>\s*(\w+)\s*\((.*?)\)\s*</function_call>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            func_name = match.group(1)
            args_str = match.group(2).strip()
            args = self._parse_arguments(args_str, func_name)
            return {"name": func_name, "arguments": json.dumps(args)}
        return None
    
    def _parse_arguments(self, args_str: str, func_name: str) -> Dict[str, Any]:
        """Parse key=value or JSON arguments."""
        try:
            if args_str.strip().startswith('{'):
                return json.loads(args_str)
        except json.JSONDecodeError:
            pass
        
        args = {}
        try:
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    args[key.strip()] = value.strip().strip('"\'')
        except Exception:
            pass
        return args

    def _detect_function_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect function intent from natural language using improved regex."""
        text_lower = text.lower()

        # Percentage
        perc_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', text_lower)
        if perc_match:
            p, num = perc_match.groups()
            expression = f"({p} / 100) * {num}"
            return {"name": "calculate", "arguments": {"expression": expression}}

        # General Math
        math_match = re.search(r'(?:calculate|compute|what is|what\'s)\s+([\d\s\.\+\-\*\/\(\)\^]+)', text_lower)
        if math_match:
            expression = math_match.group(1).strip()
            # Convert power operator for Python
            expression = expression.replace('^', '**')
            return {"name": "calculate", "arguments": {"expression": expression}}

        # Weather
        weather_loc_match = re.search(r'weather.*(?:in|for|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_loc_match:
            location = weather_loc_match.group(1).strip()
            # Clean up trailing words
            location = re.sub(r'\b(please|today|now)\b', '', location, flags=re.IGNORECASE).strip('?.,! ')
            return {"name": "get_weather", "arguments": {"location": location}}
        
        weather_temp_match = re.search(r'(?:temperature|temp)\s+(?:in|for|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_temp_match:
            location = weather_temp_match.group(1).strip().strip('?.,! ')
            return {"name": "get_weather", "arguments": {"location": location}}
            
        weather_cond_match = re.search(r'is it (raining|sunny|cloudy|snowing).*(?:in|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_cond_match:
            condition, location = weather_cond_match.groups()
            location = location.strip().strip('?.,! ')
            return {"name": "get_weather", "arguments": {"location": location}}

        # Time
        if re.search(r'time|date', text_lower):
            return {"name": "get_current_time", "arguments": {}}
            
        return None

    def _generate_response(self, prompt: str, max_len: int, temperature: float) -> str:
        """Generates a response from the model, token by token."""
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated_ids = []
        for _ in range(max_len):
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs['outputs'][:, -1, :]  # Get logits for the last token

                # --- START DEBUGGING PRINTS ---
                print(f"\n[DEBUG] Logits shape: {logits.shape}")
                top_k_logits, top_k_indices = torch.topk(logits, 5)
                print(f"[DEBUG] Top 5 Logit Values: {top_k_logits.tolist()}")
                print(f"[DEBUG] Top 5 Token IDs: {top_k_indices.tolist()}")
                print(f"[DEBUG] Top 5 Tokens: {[self.tokenizer.decode([i]) for i in top_k_indices[0]]}")
                # --- END DEBUGGING PRINTS ---

                # Apply temperature and sample
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                # Check for end-of-sequence token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

        return self.tokenizer.decode(generated_ids)

    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Main OpenAI-compatible chat completion entry point."""
        last_message = messages[-1] if messages else {}
        
        # If the last message is a function result, generate a summary
        if last_message.get("role") == "function":
            return self._handle_function_result(messages, **kwargs)
        
        # Otherwise, process the user's message
        prompt = self._format_messages(messages)
        prompt += "<assistant>"
        
        # Generate a response, which might be text or a function call
        generated_text = self._generate_response(prompt, kwargs.get('max_tokens', 150), kwargs.get('temperature', 0.7))
        
        # Check if the model decided to call a function
        function_call = self._parse_function_call(generated_text)
        
        if function_call:
            # Model wants to call a function
            choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call
                },
                "finish_reason": "function_call"
            }
        else:
            # Model generated a text response
            response_text = generated_text.replace("<assistant>", "").strip()
            if not response_text:
                response_text = "I'm not sure how to answer that. I can help with calculations, weather, and time. What would you like to know?"
            
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }
            
        return {
            "id": f"chatcmpl-hrm-{abs(hash(str(messages)))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "hrm-agent",
            "choices": [choice]
        }

    def _handle_function_result(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate a final response after a function has been executed."""
        prompt = self._format_messages(messages)
        prompt += "<assistant>"
        
        # Generate a natural language summary of the function result
        response_text = self._generate_response(prompt, kwargs.get('max_tokens', 150), kwargs.get('temperature', 0.7))
        
        # FIX: Add the same fallback logic here for robustness.
        if not response_text or len(response_text) < 5:
            response_text = "I have processed the information. What would you like to do next?"

        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }
            
        return {
            "id": f"chatcmpl-hrm-{abs(hash(str(messages)))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "hrm-agent",
            "choices": [choice]
        }
    