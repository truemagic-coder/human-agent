import ast
import random
from datetime import datetime
from typing import Dict
from .registry import FunctionRegistry

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression and return result as a string.

    Args:
        expression: A string containing a mathematical expression (e.g., "2 + 3 * 4").

    Returns:
        A string representing the result of the evaluation.
    """
    try:
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.operator,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant
        )
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        allowed_names = {'pi': 3.141592653589793, 'e': 2.718281828459045}
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)  # Return as string to preserve precision
    except SyntaxError:
        raise ValueError(f"Invalid expression syntax: {expression}")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed")
    except Exception as e:
        raise ValueError(f"Evaluation error: {str(e)}")

def get_weather(location: str) -> str:
    """Get weather information for a location (mock implementation).

    Args:
        location: The name of the city or location.

    Returns:
        A string describing the weather conditions and temperature.
    """
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "windy"]
    temp = random.randint(-5, 30)
    condition = random.choice(conditions)
    return f"Current weather in {location}: {condition}, {temp}°C, humidity {random.randint(40, 80)}%"

def search_web(query: str) -> str:
    """Search the web for information (mock implementation).

    Args:
        query: The search query string.

    Returns:
        A string with mock search results.
    """
    return f"Search results for '{query}': Mock response. Top hits include general information about '{query}'. Try a real web search API for detailed results."

def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        A string with the current date and time in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def register_builtin_functions(registry: FunctionRegistry) -> None:
    """Register all built-in functions with their JSON schemas."""
    registry.register_function(
        calculate,
        description="Safely evaluate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression (e.g., '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        }
    )
    registry.register_function(
        get_weather,
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location name (e.g., 'New York')"
                }
            },
            "required": ["location"]
        }
    )
    registry.register_function(
        search_web,
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                }
            },
            "required": ["query"]
        }
    )
    registry.register_function(
        get_current_time,
        description="Get the current date and time",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    