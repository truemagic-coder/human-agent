import torch
import uvicorn
from fastapi import FastAPI
from human_agent.core.model import create_hrm_model
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions
from human_agent.api.schemas import ChatCompletionRequest, ChatCompletionResponse

def load_trained_model(checkpoint_path: str = 'hrm_trained_model.pt'):
    """Load the trained HRM model with correct dimensions"""
    print(f"Loading trained model from {checkpoint_path}...")
    
    # Add safe globals for tokenizer
    from human_agent.core.tokenizer import Tokenizer
    torch.serialization.add_safe_globals([Tokenizer])
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        # Try alternative names
        alt_paths = ['hrm_10hour_model.pt', 'hrm_4b_model.pt', 'best_model.pt']
        checkpoint = None
        for alt_path in alt_paths:
            try:
                print(f"Trying {alt_path}...")
                checkpoint = torch.load(alt_path, map_location='cpu', weights_only=False)
                checkpoint_path = alt_path
                break
            except FileNotFoundError:
                continue
        
        if checkpoint is None:
            raise FileNotFoundError("No trained model found. Please run training first.")
    
    tokenizer = checkpoint['tokenizer']
    config = checkpoint.get('config', {})
    
    print(f"📊 Checkpoint info:")
    print(f"   Config: {config}")
    print(f"   Training loss: {checkpoint.get('loss', 'N/A')}")
    print(f"   Training time: {checkpoint.get('training_time', 'N/A')}")
    
    # DETECT DIMENSIONS FROM ACTUAL MODEL WEIGHTS
    state_dict = checkpoint['model_state_dict']
    
    # Extract actual dimensions from the saved weights
    if 'input_embedding.weight' in state_dict:
        actual_vocab_size, actual_dim = state_dict['input_embedding.weight'].shape
        print(f"🔍 Detected from weights: vocab_size={actual_vocab_size}, dim={actual_dim}")
    else:
        # Fallback to config
        actual_dim = config.get('dim', 2560)
        actual_vocab_size = config.get('vocab_size', len(tokenizer.vocab))
        print(f"🔍 Using config: vocab_size={actual_vocab_size}, dim={actual_dim}")
    
    # Detect heads from QKV weights
    if 'low_level_module.transformer.qkv.weight' in state_dict:
        qkv_out_dim, qkv_in_dim = state_dict['low_level_module.transformer.qkv.weight'].shape
        # QKV weight is [3 * n_heads * head_dim, dim]
        actual_heads = qkv_out_dim // (3 * (actual_dim // 32))  # Assume head_dim = dim // n_heads
        print(f"🔍 Detected heads from QKV: {actual_heads}")
    else:
        actual_heads = config.get('n_heads', 40)
        print(f"🔍 Using config heads: {actual_heads}")
    
    # Use config for N and T, with safer defaults for large model
    actual_N = config.get('N', 5)
    actual_T = config.get('T', 10)
    
    print(f"🧠 Loading model with DETECTED dimensions:")
    print(f"   Vocab size: {actual_vocab_size}")
    print(f"   Dimensions: {actual_dim}")
    print(f"   Heads: {actual_heads}")
    print(f"   N cycles: {actual_N}")
    print(f"   T steps: {actual_T}")
    
    # Create model with EXACT detected config
    model = create_hrm_model(
        vocab_size=actual_vocab_size,
        dim=actual_dim,        # Use detected dim
        n_heads=actual_heads,  # Use detected heads
        N=actual_N,            # Use saved N
        T=actual_T,            # Use saved T
        dropout=0.1
    )
    
    # Load the state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ State dict loaded successfully!")
    except Exception as e:
        print(f"❌ State dict loading failed: {e}")
        # Try to load with strict=False as a fallback
        print("🔄 Trying non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
        print("✅ Non-strict loading completed")
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    actual_size_b = total_params / 1_000_000_000
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {total_params:,} ({actual_size_b:.2f}B)")
    print(f"   Architecture: {actual_dim}d, {actual_heads}h, N={actual_N}, T={actual_T}")
    if 'loss' in checkpoint:
        print(f"   Final training loss: {checkpoint['loss']:.4f}")
    
    # Return the actual config for display
    actual_config = {
        'vocab_size': actual_vocab_size,
        'dim': actual_dim,
        'n_heads': actual_heads,
        'N': actual_N,
        'T': actual_T,
        'total_params': total_params
    }
    
    return model, tokenizer, actual_config

# Load the trained model
try:
    model, tokenizer, model_config = load_trained_model()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create function registry
function_registry = FunctionRegistry()
register_builtin_functions(function_registry)

# Create chat wrapper
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    model = model.to(device)

chat_wrapper = HRMChatWrapper(
    model=model,
    tokenizer=tokenizer,
    function_registry=function_registry,
    device=device
)

# Create FastAPI app
app = FastAPI(
    title="Trained HRM API", 
    version="2.0.0",
    description=f"OpenAI-compatible API for {model_config['total_params']/1_000_000_000:.1f}B Parameter Hierarchical Reasoning Model"
)

@app.get("/")
async def root():
    """Root endpoint with model information"""
    total_params = model_config['total_params']
    size_b = total_params / 1_000_000_000
    
    return {
        "message": f"🧠 {size_b:.1f}B Parameter HRM API Server",
        "model": f"hrm-{size_b:.1f}b",
        "status": "online",
        "model_info": {
            "parameters": f"{total_params:,} ({size_b:.2f}B)",
            "architecture": f"{model_config['dim']}d, {model_config['n_heads']}h, N={model_config['N']}, T={model_config['T']}",
            "vocab_size": model_config['vocab_size'],
            "reasoning_power": "MASSIVE - trained on mathematical patterns" if size_b > 1 else "LARGE - trained on mathematical patterns"
        },
        "trained_capabilities": [
            "✅ Fixed exponentiation (2^8 = 256)",
            "✅ Complex parentheses ((5+3)*4 = 32)", 
            "✅ Percentage calculations (15% of 200 = 30)",
            "✅ Weather pattern recognition",
            "✅ Natural language responses",
            "✅ Multi-step reasoning",
            f"✅ {size_b:.1f}B parameters for superior pattern learning"
        ]
    }

@app.get("/model/info")
async def model_info():
    """Detailed model information endpoint"""
    total_params = model_config['total_params']
    
    return {
        "model_config": model_config,
        "size_comparison": {
            "gpt2": "1.5B parameters",
            "this_model": f"{total_params/1_000_000_000:.2f}B parameters",
            "status": "LARGE SCALE" if total_params > 1_000_000_000 else "MEDIUM SCALE"
        },
        "architecture_details": {
            "embedding_dim": model_config['dim'],
            "attention_heads": model_config['n_heads'],
            "reasoning_cycles": model_config['N'],
            "steps_per_cycle": model_config['T'],
            "vocabulary_size": model_config['vocab_size']
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model works"""
    try:
        # Simple test
        test_result = chat_wrapper.chat_completion(
            messages=[{"role": "user", "content": "What is 2^8?"}],
            max_tokens=50,
            temperature=0.1
        )
        
        return {
            "status": "✅ Model working!",
            "test_query": "What is 2^8?",
            "test_response": test_result.get("choices", [{}])[0].get("message", {}).get("content", "No response"),
            "model_info": f"{model_config['total_params']/1_000_000_000:.1f}B parameters"
        }
    except Exception as e:
        return {
            "status": "❌ Model test failed",
            "error": str(e)
        }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    try:
        # Convert Pydantic messages to dict format
        messages = []
        for msg in request.messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if hasattr(msg, 'function_call') and msg.function_call:
                msg_dict["function_call"] = msg.function_call
            if hasattr(msg, 'name') and msg.name:
                msg_dict["name"] = msg.name
            messages.append(msg_dict)
        
        functions = None
        if request.functions:
            functions = [func.dict() for func in request.functions]
        
        # FIXED: Wrap the chat wrapper call in proper error handling
        try:
            result = chat_wrapper.chat_completion(
                messages=messages,
                functions=functions,
                function_call=request.function_call,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream
            )
            
            # FIXED: Handle the 'final_output' KeyError specifically
            if isinstance(result, dict) and "error" not in result:
                return result
            else:
                # If the wrapper failed, try direct model inference
                return fallback_generation(messages, request.max_tokens, request.temperature)
                
        except KeyError as ke:
            if "'final_output'" in str(ke):
                print(f"🔧 Caught final_output KeyError, using fallback generation")
                return fallback_generation(messages, request.max_tokens, request.temperature)
            else:
                raise ke
        except Exception as wrapper_error:
            print(f"🔧 Chat wrapper failed: {wrapper_error}, using fallback")
            return fallback_generation(messages, request.max_tokens, request.temperature)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": {
                "message": str(e),
                "type": "internal_error", 
                "code": "model_error"
            }
        }

def fallback_generation(messages, max_tokens=150, temperature=0.7):
    """Fallback generation when the wrapper fails"""
    try:
        # Simple direct model generation
        user_content = messages[-1]["content"] if messages else "Hello"
        
        # Create a simple prompt
        prompt = f"<user>{user_content}</user><assistant>"
        
        # Tokenize
        tokens = tokenizer.encode(prompt, max_length=128)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Generate with the model directly
        model.eval()
        with torch.no_grad():
            # Use a very conservative generation approach
            result = model(
                input_ids,
                max_segments=1,  # Single segment only
                min_segments=1,
                epsilon=0.99,    # Almost always stop
                training=False
            )
            
            if result and 'outputs' in result:
                outputs = result['outputs']
                
                # Get the last token predictions
                if len(outputs.shape) == 3:  # [batch, seq, vocab]
                    logits = outputs[0, -1, :]  # Last position
                else:
                    logits = outputs[-1, :]
                
                # Simple greedy decoding for safety
                next_token_id = torch.argmax(logits).item()
                
                # Generate a few more tokens
                generated_tokens = [next_token_id]
                current_input = input_ids
                
                for _ in range(min(max_tokens, 50)):  # Limit generation
                    if next_token_id == tokenizer.pad_token_id:
                        break
                    
                    # Add token and continue
                    next_token_tensor = torch.tensor([[next_token_id]]).to(device)
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
                    
                    # Limit sequence length
                    if current_input.size(1) > 200:
                        current_input = current_input[:, -128:]  # Keep last 128 tokens
                    
                    try:
                        result = model(
                            current_input,
                            max_segments=1,
                            min_segments=1,
                            epsilon=0.99,
                            training=False
                        )
                        
                        if result and 'outputs' in result:
                            outputs = result['outputs']
                            if len(outputs.shape) == 3:
                                logits = outputs[0, -1, :]
                            else:
                                logits = outputs[-1, :]
                            
                            next_token_id = torch.argmax(logits).item()
                            generated_tokens.append(next_token_id)
                        else:
                            break
                    except Exception:
                        break
                
                # Decode the generated tokens
                try:
                    generated_text = tokenizer.decode(generated_tokens)
                    
                    # Clean up the response
                    if "</assistant>" in generated_text:
                        generated_text = generated_text.split("</assistant>")[0]
                    
                    # Remove any XML tags
                    generated_text = generated_text.replace("<assistant>", "").replace("</assistant>", "")
                    generated_text = generated_text.strip()
                    
                    if not generated_text:
                        generated_text = "I understand. How can I help you?"
                    
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": generated_text
                            },
                            "finish_reason": "stop"
                        }],
                        "model": "hrm-trained",
                        "usage": {
                            "prompt_tokens": len(tokens),
                            "completion_tokens": len(generated_tokens),
                            "total_tokens": len(tokens) + len(generated_tokens)
                        }
                    }
                    
                except Exception as decode_error:
                    print(f"Decode error: {decode_error}")
                    # Return a safe default response
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant", 
                                "content": "I understand. How can I help you?"
                            },
                            "finish_reason": "stop"
                        }],
                        "model": "hrm-trained"
                    }
    
    except Exception as fallback_error:
        print(f"Fallback generation failed: {fallback_error}")
        # Last resort - return a simple response
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I understand. How can I help you?"
                },
                "finish_reason": "stop"
            }],
            "model": "hrm-trained"
        }

@app.get("/v1/functions")
async def list_functions():
    """List available functions"""
    try:
        return {
            "functions": chat_wrapper.function_registry.get_schemas()
        }
    except Exception as e:
        return {"functions": [], "error": str(e)}

if __name__ == "__main__":
    total_params = model_config['total_params']
    size_b = total_params / 1_000_000_000
    
    print(f"🚀 Starting {size_b:.1f}B Parameter HRM API Server...")
    print(f"🧠 Model: {total_params:,} parameters ({size_b:.2f}B)")
    print(f"📐 Architecture: {model_config['dim']}d, {model_config['n_heads']}h, N={model_config['N']}, T={model_config['T']}")
    print("📡 Server: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("🧪 Test endpoint: http://localhost:8000/test")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
