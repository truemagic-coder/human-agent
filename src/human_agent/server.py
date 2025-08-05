import logging
import time
import torch
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException, Depends
from .api.wrapper import HRMChatWrapper
from .api.schemas import ChatCompletionRequest
from .core.model import load_trained_model
from .functions.registry import FunctionRegistry
from .functions.builtin import register_builtin_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HRM OpenAI-Compatible API",
    version="1.0.0",
    description="OpenAI-compatible API for the Hierarchical Reasoning Model"
)

def get_chat_model():
    """Dependency to initialize and provide the chat model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading HRM model on {device}...")
        model, tokenizer, config = load_trained_model(checkpoint_path="hrm_trained_model.pt")
        function_registry = FunctionRegistry()
        register_builtin_functions(function_registry)
        chat_model = HRMChatWrapper(
            model=model,
            tokenizer=tokenizer,
            function_registry=function_registry,
            device=device
        )
        logger.info("Model loaded successfully!")
        return chat_model, config
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

@app.get("/")
async def root(config: dict = Depends(lambda: get_chat_model()[1])):
    """Root endpoint with model information."""
    params = config.get('total_params', 0)
    size_b = params / 1_000_000_000
    safe_config = {
        k: v for k, v in config.items() 
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    return {
        "message": "HRM OpenAI-Compatible API Server",
        "version": "1.0.0",
        "model": f"hrm-{size_b:.1f}b",
        "model_info": safe_config
    }

@app.get("/v1/models")
async def list_models(config: dict = Depends(lambda: get_chat_model()[1])):
    """List available models in OpenAI-compatible format."""
    params = config.get('total_params', 0)
    size_b = params / 1_000_000_000
    return {
        "object": "list",
        "data": [
            {
                "id": f"hrm-{size_b:.1f}b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hrm"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    chat_model: HRMChatWrapper = Depends(lambda: get_chat_model()[0])
):
    """OpenAI-compatible chat completions endpoint."""
    try:
        messages = [msg.dict() for msg in request.messages]
        tools = [tool.dict() for tool in request.tools] if request.tools else []
        
        response = chat_model.chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            tools=tools,
            tool_choice=request.tool_choice
        )
        return response
        
    except Exception as e:
        logger.error(f"Error during chat completion: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the server."""
    logger.info("Starting HRM OpenAI-Compatible API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
