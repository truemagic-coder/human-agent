import ast
import operator
from .registry import FunctionRegistry

# Safe operators for arithmetic evaluation
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.BitXor: operator.xor,  # allow ^ if needed
}

def _eval_ast(node) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Num):  # py<3.8
        return float(node.n)
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed.")
        return float(_OPS[op_type](left, right))
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed.")
        return float(_OPS[op_type](operand))
    raise ValueError("Unsupported expression.")

def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely and return a float."""
    tree = ast.parse(expression, mode="eval")
    return float(_eval_ast(tree))

def get_weather(location: str) -> str:
    """Return a simple mock weather string for a location."""
    # Keep as string because tests check substrings
    return f"Weather for {location}: temperature 20°C, condition Sunny"

def search_web(query: str) -> str:
    """Return a mock web search summary string."""
    return f"Search results for '{query}': [mock summary]"

def register_builtin_functions(registry: FunctionRegistry) -> None:
    """Register builtin functions into the provided registry."""
    registry.register_function(calculate, "Evaluate an arithmetic expression safely.")
    registry.register_function(get_weather, "Get mock weather for a location.")
    registry.register_function(search_web, "Search the web (mock).")
