import requests
import json
import time
from typing import Dict, Any, List

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
            print(f"❌ Failed to connect to API: {e}")
            return {}

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Sends a chat completion request to the API."""
        payload = {
            "model": "hrm-agent",
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 150),
            "temperature": kwargs.get('temperature', 0.7)
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {e}"}

def execute_function_call(func_call: Dict[str, Any]) -> Any:
    """
    Simulates the execution of a function call on the client side.
    In a real application, this would call your actual tools.
    """
    name = func_call.get("name")
    try:
        args = json.loads(func_call.get("arguments", "{}"))
        
        if name == "calculate":
            # WARNING: eval is unsafe in production. Use a safe math parser.
            expression = args.get("expression", "0")
            return eval(expression, {"__builtins__": {}}, {})
        elif name == "get_weather":
            location = args.get("location", "Unknown")
            return f"The weather in {location} is sunny and 25°C."
        elif name == "get_current_time":
            return time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return f"Unknown function: {name}"
    except Exception as e:
        return f"Error executing function {name}: {e}"

def run_conversation_step(client: HrmApiClient, messages: List[Dict[str, Any]], user_prompt: str):
    """Runs a single turn of the conversation, handling potential function calls."""
    print(f"👤 User: {user_prompt}")
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat_completion(messages=messages)
    
    if "error" in response:
        print(f"   ❌ Error: {response['error']}")
        return

    choice = response["choices"][0]
    finish_reason = choice.get("finish_reason")
    
    if finish_reason == "function_call":
        # The model wants to call a function.
        func_call = choice["message"]["function_call"]
        print(f"   🔧 Model wants to call: {func_call['name']}({func_call['arguments']})")
        
        # Execute the function and get the result.
        result = execute_function_call(func_call)
        print(f"   ✅ Client executed function. Result: {result}")
        
        # Add the function call and result to the conversation history.
        messages.append(choice["message"]) # Assistant's function call turn
        messages.append({"role": "function", "name": func_call["name"], "content": str(result)})
        
        # Call the model AGAIN with the function result to get a natural language response.
        final_response = client.chat_completion(messages=messages)
        if "error" not in final_response:
            final_content = final_response["choices"][0]["message"]["content"]
            print(f"   🤖 Assistant: {final_content}")
            messages.append({"role": "assistant", "content": final_content})
        else:
            print(f"   ❌ Error on follow-up: {final_response['error']}")

    else:
        # The model gave a direct text response.
        assistant_response = choice["message"]["content"]
        print(f"   🤖 Assistant: {assistant_response}")
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
        print("\n" + "="*50)
        print(category)
        print("="*50)
        messages = [] # Reset history for each category
        for prompt in prompts:
            run_conversation_step(client, messages, prompt)
            print("-" * 20)

def interactive_mode(client: HrmApiClient):
    """Starts an interactive chat session with the model."""
    print("\n" + "="*50)
    print("💬 Interactive Mode")
    print("="*50)
    print("Type 'quit' or 'exit' to end.")
    messages = []
    while True:
        try:
            user_prompt = input("👤 You: ").strip()
            if user_prompt.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            if user_prompt:
                run_conversation_step(client, messages, user_prompt)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function to run the client."""
    print("🧠 HRM API Client")
    client = HrmApiClient()
    
    api_info = client.get_api_info()
    if not api_info:
        print("Please make sure the server is running: python examples/serve_trained_model.py")
        return
        
    print("✅ API Server is online.")
    print(f"   Model: {api_info.get('model', 'Unknown')}")
    
    run_test_suite(client)
    
    interactive_mode(client)

if __name__ == "__main__":
    main()
