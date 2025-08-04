import torch
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from human_agent.core.model import create_hrm_model
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions
from human_agent.api.schemas import ChatCompletionRequest

def load_trained_model(checkpoint_path: str = 'hrm_trained_model.pt'):
    """
    Loads the trained model and tokenizer from a checkpoint, relying on the
    saved config for model creation. This is the reliable way to load a model.
    """
    print(f"🧠 Loading trained model from '{checkpoint_path}'...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except FileNotFoundError:
        print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'.")
        print("   Please run training first using 'training_example.py'.")
        raise
        
    # Validate the checkpoint has the required keys
    required_keys = ['model_state_dict', 'config', 'tokenizer']
    if not all(key in checkpoint for key in required_keys):
        print(f"❌ ERROR: Checkpoint is invalid. Missing one of {required_keys}.")
        raise ValueError("Invalid checkpoint format")

    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']
    
    print("✅ Checkpoint loaded. Creating model from saved config...")
    print(f"   Config: {config}")

    # Create the model with the exact config it was trained with
    model = create_hrm_model(**config)
    
    # Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded successfully.")
    
    model.eval()
    
    # Add total_params to config for display purposes
    config['total_params'] = sum(p.numel() for p in model.parameters())
    
    return model, tokenizer, config

# --- Main Application Setup ---

try:
    # Load the model using the reliable loader
    model, tokenizer, model_config = load_trained_model()
    
    # Create function registry and register built-in functions
    function_registry = FunctionRegistry()
    register_builtin_functions(function_registry)
    
    # Determine device and create the chat wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    chat_wrapper = HRMChatWrapper(
        model=model,
        tokenizer=tokenizer,
        function_registry=function_registry,
        device=device
    )
    
    # Create the FastAPI application
    app = FastAPI(
        title="HRM API",
        version="3.0",
        description="Reliable OpenAI-compatible API for the Hierarchical Reasoning Model."
    )

except Exception:
    print("❌ Failed to initialize the application.")
    traceback.print_exc()

# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint with model information."""
    params = model_config.get('total_params', 0)
    size_b = params / 1_000_000_000
    return {
        "message": "HRM API Server is online.",
        "model": f"hrm-{size_b:.1f}b",
        "model_info": model_config
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        messages = [msg.dict() for msg in request.messages]
        
        result = chat_wrapper.chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return result
        
    except Exception as e:
        print(f"❌ Error during chat completion: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting HRM API Server...")
    print(f"   Device: {device.upper()}")
    params = model_config.get('total_params', 0)
    print(f"   Model: {params:,} parameters ({params/1e9:.2f}B)")
    print("   API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
