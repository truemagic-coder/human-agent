import torch
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions

def load_trained_model(checkpoint_path: str = 'hrm_best_model.pt'):
    """Load the trained HRM model"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Add safe globals for tokenizer
    torch.serialization.add_safe_globals([SimpleTokenizer])
    
    # Load checkpoint with weights_only=False for custom objects
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    tokenizer = checkpoint['tokenizer']
    
    # Recreate model with same parameters
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=256,
        n_heads=4,
        N=2,
        T=4,
        use_act=True,
        dropout=0.1
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully! Training loss was: {checkpoint['loss']:.4f}")
    return model, tokenizer

def test_model_capabilities():
    """Test the trained model's capabilities"""
    
    # Load the trained model
    model, tokenizer = load_trained_model()
    
    # Create function registry
    function_registry = FunctionRegistry()
    register_builtin_functions(function_registry)
    
    # Create chat wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chat_wrapper = HRMChatWrapper(
        model=model,
        tokenizer=tokenizer,
        function_registry=function_registry,
        device=device
    )
    
    print("\n" + "="*60)
    print("TESTING TRAINED HRM MODEL")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "messages": [{"role": "user", "content": "Hello! How are you?"}],
            "description": "Basic Greeting"
        },
        {
            "messages": [{"role": "user", "content": "What is 15 + 25?"}],
            "description": "Simple Addition"
        },
        {
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "description": "Weather Query"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}:")
        print("-" * 40)
        print(f"Input: {test_case['messages'][0]['content']}")
        
        try:
            response = chat_wrapper.chat_completion(
                messages=test_case['messages'],
                max_tokens=100,
                temperature=0.7
            )
            
            choice = response['choices'][0]
            
            if choice['finish_reason'] == 'function_call':
                func_call = choice['message']['function_call']
                func_result = response.get('function_result', {})
                print(f"Function Call: {func_call['name']}({func_call['arguments']})")
                print(f"Function Result: {func_result}")
            else:
                print(f"Response: {choice['message']['content']}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    import os
    
    if os.path.exists('hrm_best_model.pt'):
        print("Found trained model: hrm_best_model.pt")
        test_model_capabilities()
    elif os.path.exists('hrm_final_model.pt'):
        print("Found final model: hrm_final_model.pt")
        # Modify function for final model
        def load_trained_model(checkpoint_path: str = 'hrm_final_model.pt'):
            torch.serialization.add_safe_globals([SimpleTokenizer])
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            tokenizer = checkpoint['tokenizer']
            
            model = create_hrm_model(
                vocab_size=len(tokenizer.vocab),
                dim=256, n_heads=4, N=2, T=4,
                use_act=True, dropout=0.1
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, tokenizer
        
        test_model_capabilities()
    else:
        print("No trained model found. Please run training first.")
        