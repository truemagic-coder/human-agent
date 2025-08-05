import time
import torch
import json
import re
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import logging
from ..core.model import HierarchicalReasoningModel
from ..core.tokenizer import Tokenizer
from ..functions.registry import FunctionRegistry
from ..functions.builtin import register_builtin_functions
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRMChatWrapper:
    """OpenAI-compatible wrapper for HRM model with tool calling"""
    
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
        self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Register builtin functions if no registry provided
        if function_registry is None:
            register_builtin_functions(self.function_registry)
        
        # Validate tokenizer vocabulary size matches model
        vocab_size = self.model.config.get('vocab_size', len(self.tokenizer.vocab))
        if vocab_size != len(self.tokenizer.vocab):
            logger.warning(f"Tokenizer vocab size ({len(self.tokenizer.vocab)}) does not match model vocab size ({vocab_size})")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages espírito into a single string for the model prompt."""
        formatted = ""
        valid_roles = {"user", "assistant", "tool", "function"}
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role not in valid_roles:
                logger.warning(f"Invalid role '{role}' in message, skipping")
                continue
            
            if role in ("tool", "function") and content:
                formatted += f"<function_result>{content}</function_result>"
            elif role in ("user", "assistant") and content:
                formatted += f"<{role}>{content}</{role}>"
            else:
                logger.debug(f"Skipping message with role '{role}' and no valid content")
        
        return formatted

    def _parse_explicit_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse explicit function call from generated text."""
        pattern = r'<function_call>\s*(\w+)\s*\((.*?)\)\s*</function_call>'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            logger.debug(f"No explicit function call found in text: {text}")
            return None
        
        func_name = match.group(1)
        args_str = match.group(2).strip()
        args = self._parse_arguments(args_str, func_name)
        
        if func_name not in self.function_registry.functions:
            logger.warning(f"Function {func_name} not found in registry")
            return None
        
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": func_name, "arguments": json.dumps(args)}
        }
    
    def _parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse function call from generated text, supporting both explicit and intent-based syntax."""
        logger.debug(f"Parsing function call from text: {text}")
        explicit_call = self._parse_explicit_function_call(text)
        if explicit_call:
            return explicit_call
        
        intent_call = self._detect_function_intent(text)
        if intent_call:
            return intent_call
        
        return None
    
    def _parse_arguments(self, args_str: str, func_name: str) -> Dict[str, Any]:
        """Parse key=value or JSON arguments, preserving quoted strings."""
        args = {}
        try:
            if args_str.strip().startswith('{'):
                args = json.loads(args_str)
            else:
                parts = re.split(r',(?=(?:[^\'"]|\'[^\']*\'|"[^"]*")*$)', args_str)
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        args[key] = value
        except Exception as e:
            logger.warning(f"Failed to parse arguments for {func_name}: {args_str}, error: {str(e)}")
            return {}
        
        # Validate against schema
        if func_name in self.function_registry.functions:
            schema = self.function_registry.functions[func_name].get('parameters', {})
            required = schema.get('required', [])
            properties = schema.get('properties', {})
            for key in required:
                if key not in args:
                    logger.warning(f"Missing required parameter {key} for function {func_name}")
                    return {}
                prop_type = properties.get(key, {}).get('type')
                if prop_type == 'string' and not isinstance(args[key], str):
                    logger.warning(f"Invalid type for {key}: expected string, got {type(args[key])}")
                    return {}
                elif prop_type in ('integer', 'number') and not isinstance(args[key], (int, float)):
                    logger.warning(f"Invalid type for {key}: expected number, got {type(args[key])}")
                    return {}
        return args

    def _detect_function_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect function intent from natural language using improved regex."""
        logger.debug(f"Detecting function intent from text: {text}")
        text_lower = text.lower()

        # Percentage
        perc_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', text_lower)
        if perc_match:
            p, num = perc_match.groups()
            expression = f"({p} / 100) * {num}"
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "calculate", "arguments": json.dumps({"expression": expression})}
            }

        # General Math
        math_match = re.search(r'(?:calculate|compute|what is|what\'s)\s+([\d\s\.\+\-\*\/\(\)\^]+)', text_lower)
        if math_match:
            expression = math_match.group(1).strip().replace('^', '**')
            try:
                eval(expression, {"__builtins__": {}}, {})  # Safe eval for syntax check
                return {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": json.dumps({"expression": expression})}
                }
            except Exception as e:
                logger.debug(f"Invalid math expression in intent: {expression}, error: {str(e)}")
                return None

        # Weather
        weather_loc_match = re.search(r'weather.*(?:in|for|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_loc_match:
            location = weather_loc_match.group(1).strip()
            location = re.sub(r'\b(please|today|now)\b', '', location, flags=re.IGNORECASE).strip('?.,! ')
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"location": location})}
            }
        
        weather_temp_match = re.search(r'(?:temperature|temp)\s+(?:in|for|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_temp_match:
            location = weather_temp_match.group(1).strip().strip('?.,! ')
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"location": location})}
            }
            
        weather_cond_match = re.search(r'is it (raining|sunny|cloudy|snowing).*(?:in|at)\s+([a-zA-Z\s,]+)', text_lower)
        if weather_cond_match:
            condition, location = weather_cond_match.groups()
            location = location.strip().strip('?.,! ')
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"location": location})}
            }

        # Time
        if re.search(r'time|date', text_lower):
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "get_current_time", "arguments": json.dumps({})}
            }
            
        return None

    def _generate_response(self, prompt: str, max_len: int, temperature: float, top_p: float, stop: Optional[Union[str, List[str]]]) -> tuple[str, int, int]:
        """Generates a response from the model, token by token, with token counting."""
        temperature = max(0.1, min(temperature, 2.0))
        top_p = max(0.1, min(top_p, 1.0))
        
        input_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(input_ids)
        max_seq_len = self.model.config.get('max_seq_len', 256)
        
        if len(input_ids) > max_seq_len - max_len:
            input_ids = input_ids[-(max_seq_len - max_len):]
            prompt_tokens = len(input_ids)
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_ids = []
        
        stop_tokens = []
        if stop:
            if isinstance(stop, str):
                stop_tokens = [self.tokenizer.encode(stop)[0]] if stop else []
            elif isinstance(stop, list):
                stop_tokens = [self.tokenizer.encode(s)[0] for s in stop if s]
        
        for _ in range(max_len):
            if input_tensor.shape[1] >= max_seq_len:
                logger.warning(f"Sequence length {input_tensor.shape[1]} exceeds max_seq_len {max_seq_len}. Stopping generation.")
                break

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs['logits'][:, -1, :]

                # Apply temperature and top-p sampling
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= top_p
                sorted_probs = sorted_probs * mask.float()
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token_id = torch.multinomial(sorted_probs, num_samples=1)
                next_token_id = sorted_indices.gather(-1, next_token_id)

                # Validate token ID
                if next_token_id.item() not in self.tokenizer.reverse_vocab:
                    logger.warning(f"Invalid token ID generated: {next_token_id.item()}")
                    break
                
                # Stop if EOS or stop token is generated
                if next_token_id.item() == self.tokenizer.eos_token_id or next_token_id.item() in stop_tokens:
                    break
                
                generated_ids.append(next_token_id.item())
                if input_tensor.shape[1] >= max_seq_len:
                    input_tensor = input_tensor[:, -max_seq_len + 1:]
                input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

        completion_tokens = len(generated_ids)
        decoded_text = self.tokenizer.decode(generated_ids)
        logger.debug(f"Generated text: {decoded_text}")
        return decoded_text, prompt_tokens, completion_tokens

    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Main OpenAI-compatible chat completion entry point."""
        last_message = messages[-1] if messages else {}
        
        # Handle tool_choice
        tool_choice = kwargs.get('tool_choice', 'auto')
        tools = kwargs.get('tools', [])
        if tool_choice != 'auto' and tools:
            if isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
                logger.debug(f"Tool choice specified: {tool_choice}, not yet implemented")
        
        # If the last message is a tool result, generate a summary
        if last_message.get("role") in ("tool", "function"):
            return self._handle_tool_result(messages, **kwargs)
        
        # Otherwise, process the user's message
        prompt = self._format_messages(messages)
        prompt += "<assistant>"
        
        # Generate a response
        generated_text, prompt_tokens, completion_tokens = self._generate_response(
            prompt, 
            kwargs.get('max_tokens', 150), 
            kwargs.get('temperature', 0.7),
            kwargs.get('top_p', 1.0),
            kwargs.get('stop', None)
        )
        
        response_text = re.sub(r'^<assistant>\s*|\s*</assistant>$', '', generated_text).strip()
        
        # Check if the model decided to call a tool
        tool_call = self._parse_function_call(generated_text)
        
        if tool_call:
            choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                },
                "finish_reason": "tool_calls"
            }
        else:
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
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": None
        }

    def _handle_tool_result(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate a final response after a tool has been executed."""
        prompt = self._format_messages(messages)
        prompt += "<assistant>"
        
        generated_text, prompt_tokens, completion_tokens = self._generate_response(
            prompt, 
            kwargs.get('max_tokens', 150), 
            kwargs.get('temperature', 0.7),
            kwargs.get('top_p', 1.0),
            kwargs.get('stop', None)
        )
        
        response_text = re.sub(r'^<assistant>\s*|\s*</assistant>$', '', generated_text).strip()
        
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
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": None
        }