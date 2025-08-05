import requests
import json
import time
import logging
import re
from typing import Dict, Any, List
import ast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HrmApiClient:
    """A simple and robust client for the HRM API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def get_api_info(self) -> Dict[str, Any]:
        """Gets information from the API's root endpoint."""
        try:
            response = self.session.get(self.base_url, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API: {str(e)}")
            return {}

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Sends a chat completion request to the API."""
        payload = {
            "model": "hrm-agent",
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 150),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0),
            "tools": kwargs.get('tools', [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Safely evaluate mathematical expressions",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string", "description": "A mathematical expression"}},
                            "required": ["expression"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string", "description": "The city or location name"}},
                            "required": ["location"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string", "description": "The search query string"}},
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current date and time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ])
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

def execute_function_call(func_call: Dict[str, Any]) -> str:
    """
    Executes a function call on the client side, mimicking server-side behavior.

    Args:
        func_call: Dictionary with 'name' and 'arguments' (JSON string or dict).

    Returns:
        A string result or error message.
    """
    name = func_call.get("name")
    try:
        args = json.loads(func_call.get("arguments", "{}")) if isinstance(func_call.get("arguments"), str) else func_call.get("arguments", {})
        
        if name == "calculate":
            expression = args.get("expression", "0")
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
            return str(result)
        elif name == "get_weather":
            location = args.get("location", "Unknown")
            return f"The weather in {location} is sunny and 25°C."
        elif name == "search_web":
            query = args.get("query", "Unknown")
            return f"Search results for '{query}': Mock response."
        elif name == "get_current_time":
            return time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return f"Error: Unknown function: {name}"
    except Exception as e:
        logger.error(f"Error executing function {name}: {str(e)}")
        return f"Error executing function {name}: {str(e)}"

def is_gibberish(text: str) -> bool:
    """Detects if text is gibberish (e.g., excessive repetition or invalid characters)."""
    if not text:
        return False
    # Check for excessive repetition of a tag or character
    if re.search(r'(\b\w+\b)\1{5,}', text) or re.search(r'(.)\1{10,}', text):
        return True
    # Check for high proportion of non-alphanumeric characters
    non_alnum = len(re.sub(r'[a-zA-Z0-9\s]', '', text))
    return non_alnum > len(text) * 0.5

def run_conversation_step(client: HrmApiClient, messages: List[Dict[str, Any]], user_prompt: str):
    """Runs a single turn of the conversation, handling potential tool calls."""
    logger.info(f"User: {user_prompt}")
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat_completion(messages=messages)
    
    if "error" in response:
        logger.error(f"Error: {response['error']}")
        return

    if not response.get("choices") or not isinstance(response["choices"], list) or not response["choices"]:
        logger.error("Invalid response format: no choices found")
        return

    choice = response["choices"][0]
    finish_reason = choice.get("finish_reason")
    
    if finish_reason == "tool_calls":
        tool_calls = choice["message"].get("tool_calls", [])
        if not tool_calls:
            logger.error("No tool_calls found despite finish_reason='tool_calls'")
            return
        
        for tool_call in tool_calls:
            func_call = tool_call.get("function", {})
            logger.info(f"Model wants to call: {func_call.get('name')}({func_call.get('arguments')})")
            
            result = execute_function_call(func_call)
            logger.info(f"Client executed function. Result: {result}")
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call.get("id")
            })
            
            final_response = client.chat_completion(messages=messages)
            if "error" not in final_response and final_response.get("choices"):
                final_content = final_response["choices"][0]["message"]["content"]
                if is_gibberish(final_content):
                    logger.warning(f"Gibberish response detected: {final_content}")
                    final_content = "I'm sorry, I generated an invalid response. Please try again."
                logger.info(f"Assistant: {final_content}")
                messages.append({"role": "assistant", "content": final_content})
            else:
                logger.error(f"Error on follow-up: {final_response.get('error', 'Unknown error')}")
    elif finish_reason == "function_call":  # Backward compatibility
        func_call = choice["message"].get("function_call", {})
        logger.info(f"Model wants to call: {func_call.get('name')}({func_call.get('arguments')})")
        
        result = execute_function_call(func_call)
        logger.info(f"Client executed function. Result: {result}")
        
        messages.append(choice["message"])
        messages.append({
            "role": "function",
            "name": func_call.get("name"),
            "content": str(result)
        })
        
        final_response = client.chat_completion(messages=messages)
        if "error" not in final_response and final_response.get("choices"):
            final_content = final_response["choices"][0]["message"]["content"]
            if is_gibberish(final_content):
                logger.warning(f"Gibberish response detected: {final_content}")
                final_content = "I'm sorry, I generated an invalid response. Please try again."
            logger.info(f"Assistant: {final_content}")
            messages.append({"role": "assistant", "content": final_content})
        else:
            logger.error(f"Error on follow-up: {final_response.get('error', 'Unknown error')}")
    else:
        assistant_response = choice["message"].get("content", "")
        if is_gibberish(assistant_response):
            logger.warning(f"Gibberish response detected: {assistant_response}")
            assistant_response = "I'm sorry, I generated an invalid response. Please try again."
        logger.info(f"Assistant: {assistant_response}")
        messages.append({"role": "assistant", "content": assistant_response})

def run_test_suite(client: HrmApiClient):
    """Runs a suite of tests to verify model capabilities."""
    test_cases = {
        "🗣️ Basic Conversation": [
            "Hello! How are you?",
            "What can you do?",
            "Thank you!",
        ],
        "🧮 Math Capabilities": [
            "What is 25 * 4?",
            "Calculate (100 / 5) + 3",
            "What is 15% of 300?",
        ],
        "🌤️ Weather Capabilities": [
            "What's the weather like in Tokyo?",
            "How is the weather in Paris today?",
        ],
        "⏰ Time Capabilities": [
            "What time is it?",
            "What is today's date?",
        ],
    }

    for category, prompts in test_cases.items():
        logger.info("\n" + "="*50)
        logger.info(category)
        logger.info("="*50)
        messages = [] # Reset history for each category
        for prompt in prompts:
            run_conversation_step(client, messages, prompt)
            logger.info("-" * 20)

def interactive_mode(client: HrmApiClient):
    """Starts an interactive chat session with the model."""
    logger.info("\n" + "="*50)
    logger.info("💬 Interactive Mode")
    logger.info("="*50)
    logger.info("Type 'quit' or 'exit' to end.")
    messages = []
    while True:
        try:
            user_prompt = input("👤 You: ").strip()
            if user_prompt.lower() in ["quit", "exit"]:
                logger.info("Goodbye!")
                break
            if user_prompt:
                run_conversation_step(client, messages, user_prompt)
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break

def main():
    """Main function to run the client."""
    logger.info("HRM API Client")
    client = HrmApiClient()
    
    api_info = client.get_api_info()
    if not api_info:
        logger.error("Please make sure the server is running: python examples/serve_trained_model.py")
        return
        
    logger.info("API Server is online.")
    logger.info(f"Model: {api_info.get('model', 'Unknown')}")
    
    run_test_suite(client)
    
    interactive_mode(client)

if __name__ == "__main__":
    main()
    