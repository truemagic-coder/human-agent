import logging
import time
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path: str = 'hrm_trained_model.pt'):
    """
    Loads the trained model and tokenizer from a checkpoint, relying on the
    saved config for model creation.
    """
    logger.info(f"Loading trained model from '{checkpoint_path}'...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        logger.error(f"Checkpoint not found at '{checkpoint_path}'. Please run training first.")
        raise HTTPException(status_code=500, detail=f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")

    # Validate the checkpoint has the required keys
    required_keys = ['model_state_dict', 'config', 'tokenizer_config']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        logger.error(f"Checkpoint is invalid. Missing keys: {missing_keys}")
        raise HTTPException(status_code=500, detail=f"Invalid checkpoint format. Missing keys: {missing_keys}")

    config = checkpoint['config']
    tokenizer_config = checkpoint['tokenizer_config']
    
    # Validate tokenizer config
    required_tokenizer_keys = ['vocab', 'special_tokens', 'vocab_size']
    missing_tokenizer_keys = [key for key in required_tokenizer_keys if key not in tokenizer_config]
    if missing_tokenizer_keys:
        logger.error(f"Tokenizer config is invalid. Missing keys: {missing_tokenizer_keys}")
        raise HTTPException(status_code=500, detail=f"Invalid tokenizer config. Missing keys: {missing_tokenizer_keys}")

    logger.info("Checkpoint loaded. Creating model and tokenizer from saved config...")
    logger.info(f"Model Config: {config}")

    # Create the tokenizer
    # Rebuild Tokenizer using saved config (matches the current Tokenizer API)
    tokenizer = Tokenizer(
        vocab_size=tokenizer_config['vocab_size'],
        special_tokens=tokenizer_config['special_tokens']
    )
    # Overwrite vocab with trained vocab and rebuild id_to_token
    tokenizer.vocab = tokenizer_config['vocab']
    tokenizer.special_tokens = tokenizer_config['special_tokens']
    tokenizer.id_to_token = [None] * len(tokenizer.vocab)
    for tok, idx in tokenizer.vocab.items():
        if idx >= len(tokenizer.id_to_token):
            tokenizer.id_to_token.extend([None] * (idx - len(tokenizer.id_to_token) + 1))
        tokenizer.id_to_token[idx] = tok
    # Refresh special token IDs
    tokenizer.pad_token_id = tokenizer.vocab.get('<pad>', 0)
    tokenizer.eos_token_id = tokenizer.vocab.get('<eos>', 3)
    tokenizer.bos_token_id = tokenizer.vocab.get('<bos>', 2)
    # Build reverse map for wrapper compatibility
    tokenizer.reverse_vocab = {idx: tok for tok, idx in tokenizer.vocab.items()}

    # Create the model with the exact config it was trained with
    model = create_hrm_model(**config)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load model weights: {str(e)}")

    model.eval()
    model.to(device)
    
    # Compute total parameters
    total_params = sum(p.numel() for p in model.parameters())
    config['total_params'] = total_params
    
    return model, tokenizer, config

# --- Main Application Setup ---
app = FastAPI(
    title="HRM API",
    version="3.0",
    description="Reliable OpenAI-compatible API for the Hierarchical Reasoning Model."
)

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

except Exception as e:
    logger.error(f"Failed to initialize the application: {str(e)}\n{traceback.format_exc()}")
    raise

# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint with model information."""    
    safe_config = {
        k: v for k, v in model_config.items() 
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    return {
        "message": "HRM API Server is online.",
        "model": "hrm-agent",
        "model_info": safe_config
    }

@app.get("/v1/models")
def list_models():
    """List available models in OpenAI-compatible format."""
    return {
        "object": "list",
        "data": [
            {
                "id": "hrm-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hrm"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        messages = [msg.dict() for msg in request.messages]
        tools = [tool.dict() for tool in request.tools] if request.tools else []
        
        result = chat_wrapper.chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            tools=tools,
            tool_choice=request.tool_choice
        )
        return result
        
    except Exception as e:
        logger.error(f"Error during chat completion: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting HRM API Server...")
    logger.info(f"Device: {device.type.upper()}")
    params = model_config.get('total_params', 0)
    logger.info(f"Model: hrm-agent, {params:,} parameters ({params/1e9:.2f}B)")
    logger.info("API Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
    