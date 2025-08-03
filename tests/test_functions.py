from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import calculate, get_weather, register_builtin_functions

def test_function_registry():
    """Test function registry"""
    registry = FunctionRegistry()
    
    def test_func(x: int, y: str = "default") -> str:
        return f"{x}-{y}"
    
    registry.register_function(test_func, "Test function")
    schemas = registry.get_schemas()
    
    assert len(schemas) == 1
    assert schemas[0]["name"] == "test_func"
    assert "x" in schemas[0]["parameters"]["properties"]

def test_function_calling():
    """Test function calling"""
    registry = FunctionRegistry()
    
    def add(a: int, b: int) -> int:
        return a + b
    
    registry.register_function(add)
    result = registry.call_function("add", {"a": 5, "b": 3})
    
    assert "result" in result
    assert result["result"] == 8

def test_builtin_functions():
    """Test builtin functions"""
    # Test calculate function
    result = calculate("2 + 3 * 4")
    assert result == 14.0
    
    # Test weather function
    weather = get_weather("London")
    assert "London" in weather
    assert "temperature" in weather

def test_register_builtins():
    """Test registering builtin functions"""
    registry = FunctionRegistry()
    register_builtin_functions(registry)
    
    schemas = registry.get_schemas()
    function_names = [schema["name"] for schema in schemas]
    
    assert "calculate" in function_names
    assert "get_weather" in function_names
    assert "search_web" in function_names
