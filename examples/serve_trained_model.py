import torch
import uvicorn
from fastapi import FastAPI
from human_agent.core.model import create_hrm_model
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions
from human_agent.api.schemas import ChatCompletionRequest, ChatCompletionResponse

def load_trained_model(checkpoint_path: str = 'hrm_best_model.pt'):
    """Load the trained HRM model"""
    print(f"Loading trained model from {checkpoint_path}...")
    
    # Add safe globals for tokenizer
    from human_agent.core.tokenizer import SimpleTokenizer
    torch.serialization.add_safe_globals([SimpleTokenizer])
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        # Try alternative checkpoint names
        alt_paths = ['hrm_final_model.pt', 'hrm_checkpoint_epoch_15.pt', 'hrm_checkpoint_epoch_10.pt']
        for alt_path in alt_paths:
            try:
                print(f"Trying alternative path: {alt_path}")
                checkpoint = torch.load(alt_path, map_location='cpu', weights_only=False)
                checkpoint_path = alt_path
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("No trained model checkpoint found. Please run training first.")
    
    tokenizer = checkpoint['tokenizer']
    
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=256, n_heads=4, N=2, T=4,
        use_act=True, dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Trained model loaded from {checkpoint_path}!")
    if 'loss' in checkpoint:
        print(f"   Training loss was: {checkpoint['loss']:.4f}")
    
    return model, tokenizer

# Load the trained model
try:
    model, tokenizer = load_trained_model()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure you have a trained model checkpoint in the current directory.")
    print("Expected files: hrm_best_model.pt, hrm_final_model.pt, or hrm_checkpoint_epoch_*.pt")
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
    version="1.0.0",
    description="OpenAI-compatible API for trained Hierarchical Reasoning Model"
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint using trained model"""
    
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
        # Return error in OpenAI format
        return {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "model_error"
            }
        }

@app.get("/v1/functions")
async def list_functions():
    """List available functions"""
    return {"functions": function_registry.get_schemas()}

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "hrm-trained-27m",
                "object": "model",
                "created": 1234567890,
                "owned_by": "hrm",
                "permission": [],
                "root": "hrm-trained-27m",
                "parent": None
            }
        ]
    }

@app.get("/model/info")
async def model_info():
    """Get information about the trained model"""
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model_name": "hrm-trained-27m",
        "parameters": total_params,
        "vocabulary_size": len(tokenizer.vocab),
        "architecture": {
            "dim": 256,
            "n_heads": 4,
            "N_cycles": 2,
            "T_steps": 4,
            "use_act": True
        },
        "device": str(device),
        "status": "trained"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "timestamp": 1234567890
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🧠 Trained HRM API Server",
        "model": "hrm-trained-27m",
        "status": "online",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "functions": "/v1/functions",
            "models": "/v1/models",
            "info": "/model/info",
            "health": "/health",
            "docs": "/docs"
        },
        "example_usage": {
            "curl": 'curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" -d \'{"messages":[{"role":"user","content":"What is 5 + 3?"}],"max_tokens":100}\'',
            "python": """
import requests
response = requests.post('http://localhost:8000/v1/chat/completions', json={
    'messages': [{'role': 'user', 'content': 'What is 5 + 3?'}],
    'max_tokens': 100
})
print(response.json())
"""
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Trained HRM API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🧠 Model info at: http://localhost:8000/model/info")
    print("❤️  Health check at: http://localhost:8000/health")
    print("\n🔥 Try this curl command:")
    print('curl -X POST "http://localhost:8000/v1/chat/completions" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"messages":[{"role":"user","content":"What is 15 + 25?"}],"max_tokens":100}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    