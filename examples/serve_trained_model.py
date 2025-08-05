import torch
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import Tokenizer
from human_agent.api.wrapper import HRMChatWrapper
from human_agent.functions.registry import FunctionRegistry
from human_agent.functions.builtin import register_builtin_functions
from human_agent.api.schemas import ChatCompletionRequest

def load_trained_model(checkpoint_path: str = 'hrm_trained_model.pt'):
    """
    Loads the trained model and tokenizer from a checkpoint, relying on the
    saved config for model creation.
    """
    print(f"🧠 Loading trained model from '{checkpoint_path}'...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'.")
        print("   Please run training first using 'training_example.py'.")
        raise HTTPException(status_code=500, detail=f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load checkpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")

    # Validate the checkpoint has the required keys
    required_keys = ['model_state_dict', 'config', 'tokenizer_config']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        print(f"❌ ERROR: Checkpoint is invalid. Missing keys: {missing_keys}")
        raise HTTPException(status_code=500, detail=f"Invalid checkpoint format. Missing keys: {missing_keys}")

    config = checkpoint['config']
    tokenizer_config = checkpoint['tokenizer_config']
    
    # Validate tokenizer config
    required_tokenizer_keys = ['vocab', 'special_tokens', 'vocab_size']
    missing_tokenizer_keys = [key for key in required_tokenizer_keys if key not in tokenizer_config]
    if missing_tokenizer_keys:
        print(f"❌ ERROR: Tokenizer config is invalid. Missing keys: {missing_tokenizer_keys}")
        raise HTTPException(status_code=500, detail=f"Invalid tokenizer config. Missing keys: {missing_tokenizer_keys}")

    print("✅ Checkpoint loaded. Creating model and tokenizer from saved config...")
    print(f"   Model Config: {config}")

    # Create the tokenizer
    tokenizer = Tokenizer(vocab_size=tokenizer_config['vocab_size'])
    tokenizer.vocab = tokenizer_config['vocab']
    tokenizer.reverse_vocab = {v: k for k, v in tokenizer_config['vocab'].items()}
    tokenizer.special_tokens_set = set(tokenizer_config['special_tokens'])
    tokenizer._compile_special_tokens_regex()
    tokenizer.pad_token_id = tokenizer.vocab['<pad>']
    tokenizer.eos_token_id = tokenizer.vocab['<eos>']
    tokenizer.bos_token_id = tokenizer.vocab['<bos>']

    # Create the model with the exact config it was trained with
    model = create_hrm_model(**config)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model weights: {str(e)}")

    model.eval()
    model.to(device)
    
    # Compute total parameters
    total_params = sum(p.numel() for p in model.parameters())
    config['total_params'] = total_params
    
    return model, tokenizer, config

# --- Main Application Setup ---
app = None
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model using the reliable loader
    model, tokenizer, model_config = load_trained_model()
    
    # Create function registry and register built-in functions
    function_registry = FunctionRegistry()
    register_builtin_functions(function_registry)
    
    # Create the chat wrapper
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

except Exception as e:
    print(f"❌ Failed to initialize the application: {str(e)}")
    traceback.print_exc()
    raise

# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint with model information."""
    params = model_config.get('total_params', 0)
    size_b = params / 1_000_000_000
    
    safe_config = {
        k: v for k, v in model_config.items() 
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    return {
        "message": "HRM API Server is online.",
        "model": f"hrm-{size_b:.1f}b",
        "model_info": safe_config
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
        print(f"❌ Error during chat completion: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting HRM API Server...")
    print(f"   Device: {device.upper()}")
    params = model_config.get('total_params', 0)
    print(f"   Model: {params:,} parameters ({params/1e9:.2f}B)")
    print("   API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    