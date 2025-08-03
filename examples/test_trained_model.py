import torch
from human_agent.core.model import create_hrm_model
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions

def load_trained_model(checkpoint_path: str = 'hrm_best_model.pt'):
    """Load the trained HRM model"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
        # Basic conversation
        {
            "messages": [{"role": "user", "content": "Hello! How are you?"}],
            "description": "Basic Greeting"
        },
        
        # Simple math
        {
            "messages": [{"role": "user", "content": "What is 15 + 25?"}],
            "description": "Simple Addition"
        },
        
        # Complex math
        {
            "messages": [{"role": "user", "content": "Calculate 12 * 8"}],
            "description": "Multiplication"
        },
        
        # Weather function
        {
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "description": "Weather Query"
        },
        
        # Time function
        {
            "messages": [{"role": "user", "content": "What time is it?"}],
            "description": "Time Query"
        },
        
        # Code generation
        {
            "messages": [{"role": "user", "content": "Write a function to add two numbers"}],
            "description": "Code Generation"
        },
        
        # Instruction following
        {
            "messages": [{"role": "user", "content": "Explain what AI is"}],
            "description": "Instruction Following"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}:")
        print("-" * 40)
        print(f"Input: {test_case['messages'][0]['content']}")
        
        try:
            # Generate response
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
    
    print("\n" + "="*60)
    print("Testing completed!")
    
    # Interactive mode
    print("\nEntering interactive mode (type 'quit' to exit):")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
                
            response = chat_wrapper.chat_completion(
                messages=[{"role": "user", "content": user_input}],
                max_tokens=150,
                temperature=0.7
            )
            
            choice = response['choices'][0]
            
            if choice['finish_reason'] == 'function_call':
                func_call = choice['message']['function_call']
                func_result = response.get('function_result', {})
                print(f"Assistant: [Function Call] {func_call['name']}({func_call['arguments']})")
                print(f"Result: {func_result.get('result', 'Error')}")
            else:
                print(f"Assistant: {choice['message']['content']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nGoodbye! 👋")

def analyze_model_performance():
    """Analyze the model's performance and capabilities"""
    
    model, tokenizer = load_trained_model()
    
    print("\n" + "="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Vocabulary Size: {len(tokenizer.vocab):,}")
    print(f"Model Dimension: {model.dim}")
    print(f"Number of Attention Heads: {model.n_heads}")
    print(f"Number of High-Level Cycles (N): {getattr(model, 'N', 'Unknown')}")
    print(f"Number of Low-Level Steps (T): {getattr(model, 'T', 'Unknown')}")
    
    # Test generation quality
    print("\nTesting generation with different temperatures:")
    print("-" * 40)
    
    test_prompt = "<user>What is 5 + 3?</user><assistant>"
    input_ids = tokenizer.encode(test_prompt)
    
    for temp in [0.1, 0.5, 1.0]:
        print(f"\nTemperature {temp}:")
        
        with torch.no_grad():
            # Generate with current model
            generated_ids = input_ids.copy()
            input_tensor = torch.tensor([generated_ids])
            
            for _ in range(20):  # Generate 20 tokens
                result = model(input_tensor, training=False)
                logits = result['final_output'][0, -1]
                
                if temp > 0:
                    logits = logits / temp
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = logits.argmax().item()
                
                generated_ids.append(next_token)
                input_tensor = torch.tensor([generated_ids])
                
                if next_token == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(generated_ids)
            response = generated_text[len(test_prompt):].strip()
            print(f"  {response}")

if __name__ == "__main__":
    # Test if model file exists
    import os
    
    if os.path.exists('hrm_best_model.pt'):
        print("Found trained model: hrm_best_model.pt")
        analyze_model_performance()
        test_model_capabilities()
    elif os.path.exists('hrm_final_model.pt'):
        print("Found final model: hrm_final_model.pt")
        # Modify the load function to use final model
        def load_trained_model(checkpoint_path: str = 'hrm_final_model.pt'):
            print(f"Loading model from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            tokenizer = checkpoint['tokenizer']
            
            model = create_hrm_model(
                vocab_size=len(tokenizer.vocab),
                dim=256, n_heads=4, N=2, T=4,
                use_act=True, dropout=0.1
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded successfully!")
            return model, tokenizer
        
        analyze_model_performance()
        test_model_capabilities()
    else:
        print("No trained model found. Please run training first.")
        print("Expected files: hrm_best_model.pt or hrm_final_model.pt")
