from fastapi import FastAPI, HTTPException
from typing import Optional
import torch
import uvicorn
from .api.wrapper import create_chat_model
from .api.schemas import ChatCompletionRequest

app = FastAPI(title="HRM OpenAI-Compatible API", version="1.0.0")

# Global model instance
chat_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global chat_model
    print("Loading HRM model...")
    chat_model = create_chat_model(
        vocab_size=10000,
        model_kwargs={
            'dim': 256,
            'N': 2,
            'T': 4,
        },
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Change to "cuda" if you have GPU
    )
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "HRM OpenAI-Compatible API Server", "version": "1.0.0"}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "hrm-27m",
                "object": "model",
                "created": 1234567890,
                "owned_by": "hrm"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if chat_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        functions = [func.dict() for func in request.functions] if request.functions else None
        
        response = chat_model.chat_completion(
            messages=messages,
            functions=functions,
            function_call=request.function_call,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
