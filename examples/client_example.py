import requests

def test_api():
    """Example usage of the OpenAI-compatible API"""
    base_url = "http://localhost:8000"
    
    print("=== Testing HRM API ===\n")
    
    # Test basic chat
    print("1. Testing basic chat...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "hrm-27m",
        "messages": [
            {"role": "user", "content": "Hello! How are you today?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.status_code}")
    print()
    
    # Test math function calling
    print("2. Testing math function calling...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "hrm-27m",
        "messages": [
            {"role": "user", "content": "What's 15 + 25 * 3?"}
        ],
        "max_tokens": 100,
        "temperature": 0.1
    })
    
    if response.status_code == 200:
        data = response.json()
        choice = data['choices'][0]
        if choice['finish_reason'] == 'function_call':
            func_call = choice['message']['function_call']
            func_result = data.get('function_result', {})
            print(f"Function called: {func_call['name']}")
            print(f"Arguments: {func_call['arguments']}")
            print(f"Result: {func_result}")
            
            # Now feed the result back to get a natural language response
            print("\n   Feeding result back...")
            follow_up = requests.post(f"{base_url}/v1/chat/completions", json={
                "model": "hrm-27m",
                "messages": [
                    {"role": "user", "content": "What's 15 + 25 * 3?"},
                    {"role": "assistant", "function_call": func_call},
                    {"role": "function", "name": func_call['name'], "content": str(func_result.get('result', 'Error'))}
                ],
                "max_tokens": 50,
                "temperature": 0.3
            })
            
            if follow_up.status_code == 200:
                follow_data = follow_up.json()
                print(f"   Final response: {follow_data['choices'][0]['message']['content']}")
        else:
            print(f"Regular response: {choice['message']['content']}")
    print()
    
    # Test weather function
    print("3. Testing weather function...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "hrm-27m",
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        "max_tokens": 100,
        "temperature": 0.1
    })
    
    if response.status_code == 200:
        data = response.json()
        choice = data['choices'][0]
        if choice['finish_reason'] == 'function_call':
            func_call = choice['message']['function_call']
            func_result = data.get('function_result', {})
            print(f"Function called: {func_call['name']}")
            print(f"Result: {func_result.get('result', 'Error')}")
        else:
            print(f"Regular response: {choice['message']['content']}")
    print()
    
    # Test time function
    print("4. Testing time function...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "hrm-27m",
        "messages": [
            {"role": "user", "content": "What time is it?"}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    })
    
    if response.status_code == 200:
        data = response.json()
        choice = data['choices'][0]
        if choice['finish_reason'] == 'function_call':
            func_call = choice['message']['function_call']
            func_result = data.get('function_result', {})
            print(f"Function called: {func_call['name']}")
            print(f"Result: {func_result.get('result', 'Error')}")
        else:
            print(f"Regular response: {choice['message']['content']}")
    print()
    
    # List available functions
    print("5. Available functions:")
    response = requests.get(f"{base_url}/v1/functions")
    if response.status_code == 200:
        data = response.json()
        for func in data['functions']:
            print(f"   - {func['name']}: {func['description']}")
    print()

if __name__ == "__main__":
    test_api()
