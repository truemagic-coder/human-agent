# Human Agent - Hierarchical Reasoning Model with OpenAI-Compatible API

A Python implementation of the Hierarchical Reasoning Model (HRM) with function calling capabilities and an OpenAI-compatible API.

## Features

- **Hierarchical Reasoning Model**: Implementation based on the paper "Hierarchical Reasoning Model"
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Function Calling**: Built-in support for Python function execution
- **Extensible**: Easy to register custom functions
- **Production Ready**: Built with FastAPI and proper error handling

## Installation

```bash
# Clone the repository
cd human-agent

# Install with Poetry
poetry install

# Or install with pip
pip install -e .
```

## Quick Start

### 1. Start the API Server

```bash
# Using Poetry
poetry run human-agent-server

# Or directly
python -m human_agent.server
```

### 2. Test the API

```bash
# Run the example client
python examples/client_example.py
```

### 3. Use with any OpenAI-compatible client

```python
import openai

# Point to your local server
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"  # Not used but required

response = openai.ChatCompletion.create(
    model="hrm-27m",
    messages=[
        {"role": "user", "content": "What's 15 + 25 * 3?"}
    ]
)

print(response.choices[0].message.content)
```

## API Endpoints

- `GET /` - Server info
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `GET /v1/functions` - List available functions
- `POST /v1/functions/register` - Register new functions

## Function Calling

The model supports automatic function calling. Built-in functions include:

- `calculate(expression)` - Safe mathematical expression evaluation
- `get_weather(location)` - Mock weather information
- `search_web(query)` - Mock web search
- `get_current_time()` - Current date and time

### Register Custom Functions

```python
# Via API
import requests

function_code = """
def greet(name: str) -> str:
    return f"Hello, {name}!"
"""

response = requests.post("http://localhost:8000/v1/functions/register", json={
    "name": "greet",
    "code": function_code,
    "description": "Greet a person by name"
})
```

## Model Architecture

The HRM model features:

- **Hierarchical Processing**: Two-level recurrent architecture
- **Temporal Separation**: Different timescales for planning vs. execution
- **Adaptive Computation Time**: Dynamic resource allocation
- **Deep Supervision**: Multiple training signals
- **One-step Gradients**: Memory-efficient training

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest
```

## Training

See `examples/training_example.py` for a basic training loop:

```python
from hrm_api.core.model import create_hrm_model

model = create_hrm_model(
    vocab_size=1000,
    dim=256,
    N=2,  # High-level cycles
    T=4,  # Low-level steps per cycle
    use_act=True
)

# Train on your reasoning tasks...
```

## Citation

```bibtex
@article{arXiv:2506.21734,
  title={Hierarchical Reasoning Model},
  author={Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori},
  journal={arXiv preprint arXiv:2506.21734v2},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
