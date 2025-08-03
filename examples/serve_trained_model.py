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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    tokenizer = checkpoint['tokenizer']
    
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=256, n_heads=4, N=2, T=4,
        use_act=True, dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Trained model loaded! Training loss was: {checkpoint.get('loss', 'Unknown'):.4f}")
    return model, tokenizer

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

# Create FastAPI app
app = FastAPI(title="Trained HRM API", version="1.0.0")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint using trained model"""
    
    result = chat_wrapper.chat_completion(
        messages=request.messages,
        functions=request.functions,
        function_call=request.function_call,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream
    )
    
    return ChatCompletionResponse(**result)

@app.get("/v1/functions")
async def list_functions():
    """List available functions"""
    return {"functions": function_registry.get_schemas()}

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

@app.get("/")
async def root():
    return {
        "message": "Trained HRM API Server",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/functions", 
            "/model/info"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Trained HRM API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🧠 Model info at: http://localhost:8000/model/info")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    