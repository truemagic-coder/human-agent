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
    
    # FIXED: Use the exact config from the saved checkpoint
    actual_dim = config.get('dim', 2048)  # Default to larger model
    actual_heads = config.get('n_heads', 32)
    actual_N = config.get('N', 4)
    actual_T = config.get('T', 8)
    actual_vocab_size = config.get('vocab_size', len(tokenizer.vocab))
    
    print(f"🧠 Loading model with ACTUAL saved dimensions:")
    print(f"   Vocab size: {actual_vocab_size}")
    print(f"   Dimensions: {actual_dim}")
    print(f"   Heads: {actual_heads}")
    print(f"   N cycles: {actual_N}")
    print(f"   T steps: {actual_T}")
    
    # Create model with EXACT same config as saved
    model = create_hrm_model(
        vocab_size=actual_vocab_size,
        dim=actual_dim,        # Use saved dim
        n_heads=actual_heads,  # Use saved heads
        N=actual_N,            # Use saved N
        T=actual_T,            # Use saved T
        dropout=0.1
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
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
    exit(1)

# Create function registry
function_registry = FunctionRegistry()
register_builtin_functions(function_registry)

# Create chat wrapper
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    try:
        result = chat_wrapper.chat_completion(
            messages=request.messages,
            functions=request.functions,
            function_call=request.function_call,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        return ChatCompletionResponse(**result)
    
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "internal_error", 
                "code": "model_error"
            }
        }

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

if __name__ == "__main__":
    total_params = model_config['total_params']
    size_b = total_params / 1_000_000_000
    
    print(f"🚀 Starting {size_b:.1f}B Parameter HRM API Server...")
    print(f"🧠 Model: {total_params:,} parameters ({size_b:.2f}B)")
    print(f"📐 Architecture: {model_config['dim']}d, {model_config['n_heads']}h, N={model_config['N']}, T={model_config['T']}")
    print("📡 Server: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
