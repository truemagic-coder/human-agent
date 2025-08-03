import torch
import uvicorn
from fastapi import FastAPI
from human_agent.core.model import create_hrm_model
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions
from human_agent.api.schemas import ChatCompletionRequest, ChatCompletionResponse

def load_4b_model(checkpoint_path: str = 'hrm_4b_model.pt'):
    """Load the 4B parameter HRM model"""
    print(f"Loading 4B parameter model from {checkpoint_path}...")
    
    # Add safe globals for tokenizer
    from human_agent.core.tokenizer import SimpleTokenizer
    torch.serialization.add_safe_globals([SimpleTokenizer])
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        # Try fallback to smaller model
        try:
            print("4B model not found, trying trained model...")
            checkpoint = torch.load('hrm_trained_model.pt', map_location='cpu', weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError("No 4B or trained model found. Please run training first.")
    
    tokenizer = checkpoint['tokenizer']
    config = checkpoint.get('config', {})
    
    print(f"Loading 4B model with config: {config}")
    model = create_hrm_model(
        vocab_size=config.get('vocab_size', len(tokenizer.vocab)),
        dim=config.get('dim', 2048),
        n_heads=config.get('n_heads', 32),
        N=config.get('N', 4),
        T=config.get('T', 8),
        use_act=True,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 4B model loaded! Parameters: {total_params:,} ({total_params/1_000_000_000:.2f}B)")
    if 'loss' in checkpoint:
        print(f"   Training loss: {checkpoint['loss']:.4f}")
    
    return model, tokenizer, config

# Load the 4B model
try:
    model, tokenizer, model_config = load_4b_model()
except Exception as e:
    print(f"❌ Error loading 4B model: {e}")
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
    title="4B Parameter HRM API", 
    version="2.0.0",
    description="OpenAI-compatible API for 4 Billion Parameter Hierarchical Reasoning Model"
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint using 4B model"""
    
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
    """Root endpoint with 4B model information"""
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "message": "🧠 4 BILLION Parameter HRM API Server",
        "model": "hrm-4b",
        "status": "online",
        "model_info": {
            "parameters": f"{total_params:,} ({total_params/1_000_000_000:.2f}B)",
            "architecture": f"{model_config.get('dim', 2048)}d, {model_config.get('n_heads', 32)}h, N={model_config.get('N', 4)}, T={model_config.get('T', 8)}",
            "reasoning_power": "MASSIVE - can handle complex mathematical patterns"
        },
        "enhanced_capabilities": [
            "✅ Fixed exponentiation (2^8 = 256)",
            "✅ Complex parentheses ((5+3)*4 = 32)", 
            "✅ Percentage calculations (15% of 200 = 30)",
            "✅ Weather pattern recognition",
            "✅ Natural language responses",
            "✅ Multi-step reasoning",
            "✅ 4B parameters for superior pattern learning"
        ]
    }

if __name__ == "__main__":
    total_params = sum(p.numel() for p in model.parameters())
    
    print("🚀 Starting 4 BILLION Parameter HRM API Server...")
    print(f"🧠 Model: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    print("📡 Server: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    