import inspect
from typing import Dict, Any, List, Callable
import traceback

class FunctionRegistry:
    """Registry for managing callable functions"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_schemas: Dict[str, Dict] = {}
    
    def register_function(self, func: Callable, description: str = None):
        """Register a function for calling"""
        name = func.__name__
        self.functions[name] = func
        
        # Generate OpenAI-compatible schema
        sig = inspect.signature(func)
        schema = {
            "name": name,
            "description": description or func.__doc__ or f"Call the {name} function",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if isinstance(param.annotation, int):
                    param_info["type"] = "integer"
                elif isinstance(param.annotation, float):
                    param_info["type"] = "number"
                elif isinstance(param.annotation, bool):
                    param_info["type"] = "boolean"
                elif isinstance(param.annotation, list):
                    param_info["type"] = "array"
                elif isinstance(param.annotation, dict):
                    param_info["type"] = "object"
            
            schema["parameters"]["properties"][param_name] = param_info
            
            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)
        
        self.function_schemas[name] = schema
    
    def get_schemas(self) -> List[Dict]:
        """Get all function schemas in OpenAI format"""
        return list(self.function_schemas.values())
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a registered function"""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}
        
        try:
            result = self.functions[name](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": f"Error calling {name}: {str(e)}", "traceback": traceback.format_exc()}
