import ast
import random
from .registry import FunctionRegistry

def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression"""
    try:
        # Simple whitelist of allowed operations
        allowed_nodes = (ast.Expression, ast.BinOp, ast.operator,
                        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                        ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)
        
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Operation not allowed: {type(node).__name__}")
        
        # Allow only specific names (like pi, e)
        allowed_names = {'pi': 3.14159, 'e': 2.71828}
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    # Mock weather function
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temp = random.randint(-10, 35)
    condition = random.choice(conditions)
    return f"The weather in {location} is {condition} with a temperature of {temp}°C"

def search_web(query: str) -> str:
    """Search the web for information"""
    # Mock search function
    return f"Search results for '{query}': This is a mock search result. In a real implementation, this would call a web search API."

def get_current_time() -> str:
    """Get the current time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def register_builtin_functions(registry: FunctionRegistry) -> None:
    """Register all builtin functions"""
    registry.register_function(calculate, "Safely evaluate mathematical expressions")
    registry.register_function(get_weather, "Get current weather for a location")
    registry.register_function(search_web, "Search the web for information")
    registry.register_function(get_current_time, "Get the current date and time")
