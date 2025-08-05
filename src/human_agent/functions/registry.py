import inspect
import logging
import re
from typing import Dict, Any, List, Callable, Optional
import traceback
import typing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FunctionRegistry:
    """Registry for managing callable functions with OpenAI-compatible schemas."""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_schemas: Dict[str, Dict] = {}
    
    def register_function(
        self,
        func: Callable,
        description: str = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a function with an optional parameters schema.

        Args:
            func: The function to register.
            description: A description of the function (defaults to docstring).
            parameters: Optional JSON schema for parameters (overrides auto-generated schema).
        """
        name = func.__name__
        self.functions[name] = func
        
        # Use provided parameters if given, otherwise generate schema
        if parameters:
            schema = {
                "name": name,
                "description": description or inspect.getdoc(func) or f"Call the {name} function",
                "parameters": parameters
            }
        else:
            # Generate OpenAI-compatible schema
            sig = inspect.signature(func)
            type_hints = typing.get_type_hints(func)
            
            schema = {
                "name": name,
                "description": description or inspect.getdoc(func) or f"Call the {name} function",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, str)
                param_info = {"type": "string"}  # Default
                
                # Map Python types to JSON schema types
                if param_type in (int, float):
                    param_info["type"] = "number" if param_type is float else "integer"
                elif param_type is bool:
                    param_info["type"] = "boolean"
                elif param_type in (list, typing.List):
                    param_info["type"] = "array"
                elif param_type in (dict, typing.Dict):
                    param_info["type"] = "object"
                
                # Try to extract description from docstring
                doc = inspect.getdoc(func) or ""
                param_desc = None
                if doc:
                    param_match = re.search(rf"{param_name}\s*:\s*.*?\n\s*(.*?)(?=\n\s*\w+|\Z)", doc, re.DOTALL)
                    if param_match:
                        param_desc = param_match.group(1).strip()
                param_info["description"] = param_desc or f"Parameter {param_name}"
                
                schema["parameters"]["properties"][param_name] = param_info
                if param.default == inspect.Parameter.empty:
                    schema["parameters"]["required"].append(param_name)
        
        self.function_schemas[name] = schema
    
    def get_schemas(self) -> List[Dict]:
        """Get all function schemas in OpenAI-compatible format."""
        return [{"type": "function", "function": schema} for schema in self.function_schemas.values()]
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a registered function with validated arguments.

        Args:
            name: The function name.
            arguments: Dictionary of argument names and values.

        Returns:
            A string result or error message.
        """
        if name not in self.functions:
            logger.error(f"Function {name} not found")
            return f"Error: Function {name} not found"
        
        # Validate arguments against schema
        schema = self.function_schemas.get(name, {})
        required = schema.get("parameters", {}).get("required", [])
        properties = schema.get("parameters", {}).get("properties", {})
        
        for req_param in required:
            if req_param not in arguments:
                logger.error(f"Missing required parameter {req_param} for function {name}")
                return f"Error: Missing required parameter {req_param}"
            
            param_type = properties.get(req_param, {}).get("type", "string")
            value = arguments[req_param]
            if param_type == "string" and not isinstance(value, str):
                logger.error(f"Invalid type for {req_param}: expected string, got {type(value)}")
                return f"Error: Invalid type for {req_param}, expected string"
            elif param_type in ("integer", "number") and not isinstance(value, (int, float)):
                logger.error(f"Invalid type for {req_param}: expected number, got {type(value)}")
                return f"Error: Invalid type for {req_param}, expected number"
        
        try:
            result = self.functions[name](**arguments)
            return str(result)  # Ensure string output for <function_result>
        except Exception as e:
            logger.error(f"Error calling {name}: {str(e)}\n{traceback.format_exc()}")
            return f"Error calling {name}: {str(e)}"
        