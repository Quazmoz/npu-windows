"""
Intel NPU LLM Server - Multi-Model Support
Serves multiple LLMs via OpenAI-compatible API using Intel NPU acceleration.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")

import argparse
import uvicorn
import time
import uuid
import torch
import json
import asyncio
import os
from pathlib import Path
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Load .env file for HuggingFace token
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_paths = [
        Path(__file__).parent / ".env",  # intel-npu-llm/.env
        Path(__file__).parent.parent / ".env",  # repo root/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
            break

load_env_file()

# Set HuggingFace token if available
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"HuggingFace token loaded (length: {len(HF_TOKEN)})")
else:
    print("No HuggingFace token found. Gated models (Llama) will not work.")
    print("To use Llama models, create a .env file with: HF_TOKEN=hf_your_token_here")

# CRITICAL: Use the NPU-specific model loader!
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

app = FastAPI(title="Intel NPU LLM Server")

# --- Available Models Configuration ---
# Models verified compatible with ipex-llm NPU (from official docs)
AVAILABLE_MODELS = {
    # === QWEN 1.5 SERIES - Verified Working ===
    "qwen1.5-1.8b": {
        "hf_id": "Qwen/Qwen1.5-1.8B-Chat",
        "name": "Qwen1.5 1.8B",
        "description": "✅ VERIFIED - Fast, stable (~8 tok/s)"
    },
    "qwen1.5-4b": {
        "hf_id": "Qwen/Qwen1.5-4B-Chat",
        "name": "Qwen1.5 4B",
        "description": "Better quality (~5 tok/s)"
    },
    "qwen1.5-7b": {
        "hf_id": "Qwen/Qwen1.5-7B-Chat",
        "name": "Qwen1.5 7B",
        "description": "Best Qwen1.5 quality (~3 tok/s)"
    },
    # === QWEN 2 SERIES - Officially Listed ===
    "qwen2-1.5b": {
        "hf_id": "Qwen/Qwen2-1.5B-Instruct",
        "name": "Qwen2 1.5B",
        "description": "Fast Qwen2 (~10 tok/s)"
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2-7B-Instruct",
        "name": "Qwen2 7B",
        "description": "Officially supported (~3 tok/s)"
    },
    # === LLAMA SERIES - Officially Listed ===
    "llama2-7b": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        "name": "Llama 2 7B",
        "description": "Classic Llama (~3 tok/s)"
    },
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "name": "Llama 3 8B",
        "description": "Best open Llama (~2 tok/s)"
    },
    # === DEEPSEEK - Officially Listed ===
    "deepseek-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "DeepSeek R1 1.5B",
        "description": "Reasoning model (~10 tok/s)"
    },
    "deepseek-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "name": "DeepSeek R1 7B",
        "description": "Best reasoning (~3 tok/s)"
    },
    # === LLAMA 3.2 SERIES - Officially Verified for NPU ===
    "llama3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B",
        "description": "⚡ Fastest Llama (~15 tok/s), requires HF login"
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B",
        "description": "Fast Llama (~10 tok/s), requires HF login"
    },
    # === QWEN 2.5 SERIES - Officially Verified for NPU ===
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen 2.5 3B",
        "description": "🔥 Latest Qwen (~8 tok/s)"
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B",
        "description": "🔥 Best Qwen 2.5 (~3 tok/s)"
    },
    # === GLM-EDGE SERIES - Officially Verified for NPU ===
    "glm-edge-1.5b": {
        "hf_id": "THUDM/glm-edge-1.5b-chat",
        "name": "GLM-Edge 1.5B",
        "description": "Chinese/English bilingual (~10 tok/s)"
    },
    "glm-edge-4b": {
        "hf_id": "THUDM/glm-edge-4b-chat",
        "name": "GLM-Edge 4B",
        "description": "Larger bilingual model (~5 tok/s)"
    },
    # === MINICPM SERIES - Officially Verified for NPU ===
    "minicpm-1b": {
        "hf_id": "openbmb/MiniCPM-1B-sft-bf16",
        "name": "MiniCPM 1B",
        "description": "Ultra-compact (~15 tok/s)"
    },
    "minicpm-2b": {
        "hf_id": "openbmb/MiniCPM-2B-sft-bf16",
        "name": "MiniCPM 2B",
        "description": "Small but capable (~10 tok/s)"
    },
    # === BAICHUAN2 - Officially Verified for NPU ===
    "baichuan2-7b": {
        "hf_id": "baichuan-inc/Baichuan2-7B-Chat",
        "name": "Baichuan2 7B",
        "description": "Chinese-focused LLM (~3 tok/s)"
    },
}

# --- Global State ---
loaded_models: Dict[str, Any] = {}
default_model_id = "qwen1.5-1.8b"  # Use the verified working model

# --- NPU Model Cache Directory ---
NPU_MODEL_CACHE = os.path.join(os.path.dirname(__file__), "npu_model_cache")

# --- OpenAI API Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]

# --- OpenAI Responses API Models (for N8N compatibility) ---
class ResponseInputMessage(BaseModel):
    role: str
    content: str

class ResponseRequest(BaseModel):
    """OpenAI Responses API request format (used by N8N)."""
    model: str
    input: Any  # Can be string or list of messages
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ResponseOutputMessage(BaseModel):
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: List[Dict[str, Any]]

class ResponseObject(BaseModel):
    """OpenAI Responses API response format."""
    id: str
    object: str = "response"
    created_at: int
    model: str
    output: List[ResponseOutputMessage]
    status: str = "completed"

# --- Model Loading ---
def load_npu_model(model_id: str, hf_model_path: str):
    """Load a single model with NPU optimization."""
    global loaded_models
    
    print(f"\n{'='*50}")
    print(f"Loading '{model_id}' ({hf_model_path}) for Intel NPU...")
    
    npu_env = os.environ.get("IPEX_LLM_NPU_MTL", "not set")
    print(f" NPU Environment: IPEX_LLM_NPU_MTL={npu_env}")
    
    # Create cache directory for NPU model
    model_cache_dir = os.path.join(NPU_MODEL_CACHE, hf_model_path.replace("/", "_"))
    
    if not os.path.exists(model_cache_dir):
        # Create parent directories and convert model
        os.makedirs(model_cache_dir, exist_ok=True)
        print(f" Converting model to NPU format (first time only)...")
        print(f" Cache: {model_cache_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_low_bit="sym_int4",
            optimize_model=True,
            max_context_len=1024,
            max_prompt_len=512,
            save_directory=model_cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(model_cache_dir)
        print(f" -> Model converted and cached.")
    else:
        print(f" Loading from cache: {model_cache_dir}")
        model = AutoModelForCausalLM.load_low_bit(
            model_cache_dir,
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
        print(f" -> Loaded from cache.")
    
    loaded_models[model_id] = {
        "model": model,
        "tokenizer": tokenizer,
        "hf_id": hf_model_path
    }
    print(f" ✓ '{model_id}' ready on Intel NPU!")

def load_all_models(model_ids: List[str]):
    """Load all specified models at startup."""
    print("\n" + "="*60)
    print("  Intel NPU LLM Server - Loading Models")
    print("="*60)
    
    for model_id in model_ids:
        if model_id in AVAILABLE_MODELS:
            hf_id = AVAILABLE_MODELS[model_id]["hf_id"]
            load_npu_model(model_id, hf_id)
        else:
            print(f"WARNING: Unknown model '{model_id}', skipping.")
    
    print("\n" + "="*60)
    print(f"  {len(loaded_models)} model(s) loaded and ready!")
    print("="*60 + "\n")

def get_model_and_tokenizer(model_id: str):
    """Get model and tokenizer for the given model ID."""
    # Try exact match
    if model_id in loaded_models:
        return loaded_models[model_id]["model"], loaded_models[model_id]["tokenizer"]
    
    # Fallback to default
    if default_model_id in loaded_models:
        return loaded_models[default_model_id]["model"], loaded_models[default_model_id]["tokenizer"]
    
    # Use first available
    if loaded_models:
        first_key = next(iter(loaded_models))
        return loaded_models[first_key]["model"], loaded_models[first_key]["tokenizer"]
    
    raise HTTPException(status_code=500, detail="No models loaded")

# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model, tokenizer = get_model_and_tokenizer(request.model)
    
    # NPU model context limits (set during compilation)
    MAX_CONTEXT_LEN = 1024
    MAX_PROMPT_LEN = 512
    
    # Format prompt (ChatML format for Qwen/compatible models)
    prompt = ""
    for msg in request.messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    # Encode and check length
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_ids.shape[1]
    
    # Truncate input if too long (keep last MAX_PROMPT_LEN tokens)
    if input_length > MAX_PROMPT_LEN:
        input_ids = input_ids[:, -MAX_PROMPT_LEN:]
        input_length = MAX_PROMPT_LEN
        print(f"[WARN] Input truncated to {MAX_PROMPT_LEN} tokens")
    
    # Cap max_new_tokens to stay within context limit
    available_tokens = MAX_CONTEXT_LEN - input_length - 10  # Leave some buffer
    max_new_tokens = min(request.max_tokens or 512, available_tokens, 500)
    max_new_tokens = max(max_new_tokens, 10)  # At least 10 tokens
    
    # Generation config for NPU
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # --- Streaming Response ---
    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        
        def generate_in_thread():
            try:
                model.generate(input_ids, **gen_kwargs)
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
        
        thread = Thread(target=generate_in_thread)
        thread.start()

        async def stream_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            for text in streamer:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            end_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Standard Response ---
    else:
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)
        
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ]
        )

@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    """
    OpenAI Responses API endpoint (for N8N compatibility).
    Converts Responses API format to internal format and returns response.
    """
    model, tokenizer = get_model_and_tokenizer(request.model)
    
    # NPU model context limits
    MAX_CONTEXT_LEN = 1024
    MAX_PROMPT_LEN = 512
    
    # Convert input to prompt
    # Input can be a string or a list of messages
    if isinstance(request.input, str):
        # Simple string input
        prompt = ""
        if request.instructions:
            prompt += f"<|im_start|>system\n{request.instructions}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{request.input}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    elif isinstance(request.input, list):
        # List of messages
        prompt = ""
        if request.instructions:
            prompt += f"<|im_start|>system\n{request.instructions}<|im_end|>\n"
        for msg in request.input:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    else:
        raise HTTPException(status_code=400, detail="Input must be a string or list of messages")
    
    # Encode and check length
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_ids.shape[1]
    
    # Truncate if too long
    if input_length > MAX_PROMPT_LEN:
        input_ids = input_ids[:, -MAX_PROMPT_LEN:]
        input_length = MAX_PROMPT_LEN
        print(f"[WARN] Input truncated to {MAX_PROMPT_LEN} tokens")
    
    # Cap max tokens
    available_tokens = MAX_CONTEXT_LEN - input_length - 10
    max_new_tokens = min(request.max_output_tokens or 512, available_tokens, 500)
    max_new_tokens = max(max_new_tokens, 10)
    
    # Generation config
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    
    # Generate response (non-streaming for now)
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
    
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Build Responses API format response
    response_id = f"resp-{uuid.uuid4()}"
    message_id = f"msg-{uuid.uuid4()}"
    
    return ResponseObject(
        id=response_id,
        created_at=int(time.time()),
        model=request.model,
        output=[
            ResponseOutputMessage(
                id=message_id,
                content=[{"type": "output_text", "text": generated_text}]
            )
        ]
    )

@app.get("/v1/models")
async def list_models():
    """Return list of available models for OpenAI API compatibility."""
    models_list = []
    for model_id, data in loaded_models.items():
        model_info = AVAILABLE_MODELS.get(model_id, {})
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "intel-npu",
            "name": model_info.get("name", model_id),
            "description": model_info.get("description", "")
        })
    
    return {"object": "list", "data": models_list}

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(loaded_models)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel NPU LLM Server")
    parser.add_argument(
        "--models", 
        type=str, 
        default="qwen-1.8b",
        help="Comma-separated list of models to load (e.g., 'qwen-1.8b,qwen2-1.5b,phi3-mini')"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Models:")
        print("-" * 60)
        for model_id, info in AVAILABLE_MODELS.items():
            print(f"  {model_id:15} - {info['name']}")
            print(f"                    {info['description']}")
            print(f"                    HF: {info['hf_id']}")
            print()
        exit(0)
    
    # Parse model list
    model_ids = [m.strip() for m in args.models.split(",")]
    
    # Load models
    load_all_models(model_ids)
    
    print(f"Server starting on http://0.0.0.0:{args.port}")
    print(f"Models available: {', '.join(loaded_models.keys())}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
