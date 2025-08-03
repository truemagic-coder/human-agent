import requests
import json
from typing import Dict, Any, List

class HRMClient:
    """Client for testing the HRM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 150, 
        temperature: float = 0.7,
        model: str = "hrm-27m"
    ) -> Dict[str, Any]:
        """Send a chat completion request"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle both direct dict and Pydantic model responses
                if isinstance(result, dict):
                    return result
                else:
                    # Convert Pydantic model to dict if needed
                    return result
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return {"error": response.text}
                
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to API server. Is it running?"}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """Get available functions"""
        try:
            response = self.session.get(f"{self.base_url}/v1/functions")
            if response.status_code == 200:
                return response.json().get("functions", [])
            return []
        except Exception:
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def health_check(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
        
def test_basic_conversation():
    """Test basic conversation capabilities"""
    print("🗣️  Testing Basic Conversation")
    print("=" * 50)
    
    client = HRMClient()
    
    test_cases = [
        "Hello! How are you?",
        "What's your name?",
        "Tell me about yourself",
        "Thank you for helping me",
        "Goodbye!"
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n{i}. User: {message}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": message}],
            max_tokens=100,
            temperature=0.7
        )
        
        if "error" not in response:
            choice = response["choices"][0]
            print(f"   Assistant: {choice['message']['content']}")
        else:
            print(f"   Error: {response['error']}")

def test_math_capabilities():
    """Test mathematical function calling"""
    print("\n\n🧮 Testing Mathematical Capabilities")
    print("=" * 50)
    
    client = HRMClient()
    
    math_problems = [
        "What is 15 + 25?",
        "Calculate 12 * 8",
        "What's 100 - 37?",
        "Compute 144 / 12",
        "What is 2^8?",
        "Calculate (5 + 3) * 4",
        "What's 15% of 200?"
    ]
    
    for i, problem in enumerate(math_problems, 1):
        print(f"\n{i}. User: {problem}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": problem}],
            max_tokens=100,
            temperature=0.1  # Low temperature for consistent math
        )
        
        if "error" not in response:
            choice = response["choices"][0]
            
            if choice["finish_reason"] == "function_call":
                func_call = choice["message"]["function_call"]
                func_result = response.get("function_result", {})
                
                print(f"   🔧 Function: {func_call['name']}")
                print(f"   📝 Arguments: {func_call['arguments']}")
                print(f"   ✅ Result: {func_result.get('result', 'Error')}")
                
                # Test follow-up conversation
                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "function_call": json.loads(func_call['arguments']) if isinstance(func_call['arguments'], str) else func_call['arguments']},
                    {"role": "function", "name": func_call['name'], "content": str(func_result.get('result', 'Error'))}
                ]
                
                follow_up = client.chat_completion(
                    messages=messages,
                    max_tokens=50,
                    temperature=0.3
                )
                
                if "error" not in follow_up:
                    final_response = follow_up["choices"][0]["message"]["content"]
                    print(f"   💬 Final: {final_response}")
            else:
                print(f"   💬 Response: {choice['message']['content']}")
        else:
            print(f"   ❌ Error: {response['error']}")

def test_weather_capabilities():
    """Test weather function calling"""
    print("\n\n🌤️  Testing Weather Capabilities")
    print("=" * 50)
    
    client = HRMClient()
    
    weather_queries = [
        "What's the weather in Tokyo?",
        "How's the weather in London today?",
        "Weather forecast for New York?",
        "Is it raining in Paris?",
        "Temperature in Sydney?",
        "Weather in Berlin please"
    ]
    
    for i, query in enumerate(weather_queries, 1):
        print(f"\n{i}. User: {query}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
            temperature=0.1
        )
        
        if "error" not in response:
            choice = response["choices"][0]
            
            if choice["finish_reason"] == "function_call":
                func_call = choice["message"]["function_call"]
                func_result = response.get("function_result", {})
                
                print(f"   🔧 Function: {func_call['name']}")
                print(f"   📍 Location: {json.loads(func_call['arguments'])['location'] if isinstance(func_call['arguments'], str) else func_call['arguments']['location']}")
                print(f"   🌡️  Result: {func_result.get('result', 'Error')}")
            else:
                print(f"   💬 Response: {choice['message']['content']}")
        else:
            print(f"   ❌ Error: {response['error']}")

def test_time_capabilities():
    """Test time function calling"""
    print("\n\n⏰ Testing Time Capabilities")
    print("=" * 50)
    
    client = HRMClient()
    
    time_queries = [
        "What time is it?",
        "Current time please",
        "Tell me the current date and time",
        "What's today's date?",
        "Show me the timestamp"
    ]
    
    for i, query in enumerate(time_queries, 1):
        print(f"\n{i}. User: {query}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
            temperature=0.1
        )
        
        if "error" not in response:
            choice = response["choices"][0]
            
            if choice["finish_reason"] == "function_call":
                func_call = choice["message"]["function_call"]
                func_result = response.get("function_result", {})
                
                print(f"   🔧 Function: {func_call['name']}")
                print(f"   ⏰ Result: {func_result.get('result', 'Error')}")
            else:
                print(f"   💬 Response: {choice['message']['content']}")
        else:
            print(f"   ❌ Error: {response['error']}")

def test_mixed_conversation():
    """Test a mixed conversation with multiple capabilities"""
    print("\n\n🎭 Testing Mixed Conversation")
    print("=" * 50)
    
    client = HRMClient()
    
    conversation = [
        "Hello! I need help with some calculations and information.",
        "What is 25 * 4?",
        "Great! Now what's the weather like in London?",
        "Perfect. What time is it right now?",
        "Thank you for all the help!"
    ]
    
    messages = []
    
    for i, user_message in enumerate(conversation, 1):
        print(f"\n{i}. User: {user_message}")
        
        # Add user message to conversation
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.5
        )
        
        if "error" not in response:
            choice = response["choices"][0]
            
            if choice["finish_reason"] == "function_call":
                func_call = choice["message"]["function_call"]
                func_result = response.get("function_result", {})
                
                print(f"   🔧 Function: {func_call['name']} → {func_result.get('result', 'Error')}")
                
                # Add function call and result to conversation
                messages.append({"role": "assistant", "function_call": func_call})
                messages.append({"role": "function", "name": func_call['name'], "content": str(func_result.get('result', 'Error'))})
                
                # Get final response
                final_response = client.chat_completion(
                    messages=messages,
                    max_tokens=50,
                    temperature=0.3
                )
                
                if "error" not in final_response:
                    final_content = final_response["choices"][0]["message"]["content"]
                    print(f"   💬 Assistant: {final_content}")
                    messages.append({"role": "assistant", "content": final_content})
            else:
                assistant_response = choice["message"]["content"]
                print(f"   💬 Assistant: {assistant_response}")
                messages.append({"role": "assistant", "content": assistant_response})
        else:
            print(f"   ❌ Error: {response['error']}")

def show_api_info():
    """Show API and model information"""
    print("ℹ️  API Information")
    print("=" * 50)
    
    client = HRMClient()
    
    # Health check
    if client.health_check():
        print("✅ API is healthy and responding")
    else:
        print("❌ API is not responding")
        return
    
    # Model info
    model_info = client.get_model_info()
    if model_info:
        print("\n🧠 Model Information:")
        print(f"   Name: {model_info.get('model_name', 'Unknown')}")
        print(f"   Parameters: {model_info.get('parameters', 'Unknown')}")
        print(f"   Vocabulary: {model_info.get('vocabulary_size', 'Unknown')}")
        print(f"   Device: {model_info.get('device', 'Unknown')}")
        
        arch = model_info.get('architecture', {})
        if arch:
            print("   Architecture:")
            print(f"     - Dimension: {arch.get('dim', 'Unknown')}")
            print(f"     - Attention Heads: {arch.get('n_heads', 'Unknown')}")
            print(f"     - HRM Cycles (N): {arch.get('N_cycles', 'Unknown')}")
            print(f"     - HRM Steps (T): {arch.get('T_steps', 'Unknown')}")
    
    # Available functions
    functions = client.get_functions()
    if functions:
        print(f"\n🔧 Available Functions ({len(functions)}):")
        for func in functions:
            print(f"   - {func['name']}: {func.get('description', 'No description')}")

def interactive_mode():
    """Interactive chat mode"""
    print("\n\n💬 Interactive Mode")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'clear' to clear conversation history")
    print("-" * 50)
    
    client = HRMClient()
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                messages = []
                print("🗑️  Conversation cleared")
                continue
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            response = client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            if "error" not in response:
                choice = response["choices"][0]
                
                if choice["finish_reason"] == "function_call":
                    func_call = choice["message"]["function_call"]
                    func_result = response.get("function_result", {})
                    
                    print(f"🔧 [Function Call] {func_call['name']}")
                    print(f"📝 Arguments: {func_call['arguments']}")
                    print(f"✅ Result: {func_result.get('result', 'Error')}")
                    
                    # Add to conversation
                    messages.append({"role": "assistant", "function_call": func_call})
                    messages.append({"role": "function", "name": func_call['name'], "content": str(func_result.get('result', 'Error'))})
                    
                    # Get natural language response
                    final_response = client.chat_completion(
                        messages=messages,
                        max_tokens=100,
                        temperature=0.5
                    )
                    
                    if "error" not in final_response:
                        final_content = final_response["choices"][0]["message"]["content"]
                        print(f"Assistant: {final_content}")
                        messages.append({"role": "assistant", "content": final_content})
                else:
                    assistant_response = choice["message"]["content"]
                    print(f"Assistant: {assistant_response}")
                    messages.append({"role": "assistant", "content": assistant_response})
            else:
                print(f"❌ Error: {response['error']}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    """Main test function"""
    print("🧠 HRM API Client Test Suite")
    print("=" * 60)
    
    # Check if API is available
    client = HRMClient()
    if not client.health_check():
        print("❌ API server is not responding at http://localhost:8000")
        print("Please start the server with: python examples/serve_trained_model.py")
        return
    
    # Show API info
    show_api_info()
    
    # Run tests
    test_basic_conversation()
    test_math_capabilities()
    test_weather_capabilities()
    test_time_capabilities()
    test_mixed_conversation()
    
    # Interactive mode
    print("\n" + "=" * 60)
    interactive_input = input("Would you like to enter interactive mode? (y/n): ").strip().lower()
    if interactive_input in ['y', 'yes']:
        interactive_mode()

if __name__ == "__main__":
    main()
