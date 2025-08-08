import inspect
from typing import Any, Callable, Dict, List, Optional

_JSON_TYPE_MAP = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
}

def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    return _JSON_TYPE_MAP.get(annotation, {"type": "string"})

class FunctionRegistry:
    """Simple function registry with JSON schema and safe calling."""

    def __init__(self):
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_function(
        self,
        func: Callable[..., Any],
        description: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Register a function and build a JSON schema from its signature."""
        func_name = name or func.__name__
        sig = inspect.signature(func)

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                # Skip *args/**kwargs in schema
                continue
            schema = _annotation_to_schema(param.annotation)
            if param.default is not inspect._empty:
                schema["default"] = param.default
            else:
                required.append(param_name)
            properties[param_name] = schema

        schema: Dict[str, Any] = {
            "name": func_name,
            "description": description or func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        self._functions[func_name] = func
        self._schemas[func_name] = schema

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return list of registered function schemas."""
        return list(self._schemas.values())

    def call_function(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a registered function with JSON arguments and wrap the result."""
        if name not in self._functions:
            raise KeyError(f"Function '{name}' is not registered.")
        func = self._functions[name]
        arguments = arguments or {}
        result = func(**arguments)
        return {"result": result}
    