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
import logging
import psutil
import re
import contextlib
from pathlib import Path
from threading import Thread
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("npu-server")

# Load .env file for HuggingFace token
def find_and_load_dotenv():
    """Search for .env in current and parent directories and load it."""
    # Look for .env in two places:
    # 1. intel-npu-llm/.env (where the script is)
    # 2. repo root/.env (where start_backend.bat is)
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env"
    ]
    for p in env_paths:
        if p.exists():
            logger.info(f"Loading environment from: {p}")
            load_dotenv(dotenv_path=p)
            break

find_and_load_dotenv()

# Set HuggingFace token if available
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    logger.info(f"HuggingFace token loaded (length: {len(HF_TOKEN)})")
else:
    logger.warning("No HuggingFace token found. Gated models (Llama) will not work.")
    logger.info("To use Llama models, create a .env file with: HF_TOKEN=hf_your_token_here")

# CRITICAL: Use the NPU-specific model loader!
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

# --- Available Models Configuration ---
def load_models_config():
    """Load model definitions from models.json."""
    config_path = Path(__file__).parent / "models.json"
    if config_path.exists():
        try:
            # utf-8-sig handles both UTF-8 and UTF-8-with-BOM files (common Windows issue)
            with open(config_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models.json: {e}")
    return {}

AVAILABLE_MODELS = load_models_config()

# Module-level defaults (override per model in models.json via "max_context_len" / "max_prompt_len")
DEFAULT_MAX_CONTEXT_LEN = 1024
DEFAULT_MAX_PROMPT_LEN = 512


def get_model_context_limits(model_id: str) -> tuple[int, int]:
    """Return (max_context_len, max_prompt_len) for a given model."""
    model_cfg = AVAILABLE_MODELS.get(model_id, {})
    return (
        model_cfg.get("max_context_len", DEFAULT_MAX_CONTEXT_LEN),
        model_cfg.get("max_prompt_len", DEFAULT_MAX_PROMPT_LEN),
    )


def get_model_lock(model_id: str) -> asyncio.Lock:
    """Return the asyncio lock for a given model, falling back to a shared lock."""
    return model_locks.get(model_id, npu_resource_lock)

# --- Global State ---
loaded_models: Dict[str, Any] = {}
default_model_id = "qwen1.5-1.8b"  # Use the verified working model
npu_resource_lock = asyncio.Lock()  # Fallback shared lock for safety
model_locks: Dict[str, asyncio.Lock] = {}
model_load_tasks: Dict[str, asyncio.Task] = {}
model_status_overrides: Dict[str, Dict[str, Any]] = {}
model_load_lock = asyncio.Lock()
is_generating = False  # Explicit state for tracking
models_ready = asyncio.Event()
model_ids_to_load: List[str] = []
model_loader_task: Optional[asyncio.Task] = None


def get_hf_home_dir() -> str:
    """Return the Hugging Face cache home directory."""
    return os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def get_npu_cache_dir(hf_model_path: str) -> Path:
    """Return the compiled NPU cache directory for a model."""
    return Path(NPU_MODEL_CACHE) / hf_model_path.replace("/", "_")


def has_npu_cache(hf_model_path: str) -> bool:
    """Return True if an NPU-compiled cache exists for a model."""
    cache_dir = get_npu_cache_dir(hf_model_path)
    try:
        return cache_dir.exists() and any(cache_dir.iterdir())
    except OSError:
        return False


def has_hf_cache(hf_model_path: str) -> bool:
    """Return True if the Hugging Face hub already has cached snapshots for a model."""
    repo_dir = Path(get_hf_home_dir()) / "hub" / f"models--{hf_model_path.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    try:
        return snapshots_dir.exists() and any(snapshots_dir.iterdir())
    except OSError:
        return False


def set_model_status(model_id: str, status: str, phase: Optional[str] = None, error: Optional[str] = None) -> None:
    """Persist a transient model status used by the UI while loading."""
    model_status_overrides[model_id] = {
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at": int(time.time())
    }


def clear_model_status(model_id: str) -> None:
    """Clear any transient status override for a model."""
    model_status_overrides.pop(model_id, None)


def get_model_status_label(status: str, phase: Optional[str] = None) -> str:
    """Return a human-readable label for a model state."""
    if status == "loaded":
        return "Loaded"
    if status == "queued":
        return "Queued..."
    if status == "loading":
        phase_labels = {
            "download": "Downloading...",
            "prepare_cache": "Preparing NPU cache...",
            "load_cache": "Loading cached model...",
        }
        return phase_labels.get(phase, "Loading...")
    if status == "ready_to_load":
        return "Ready to load"
    if status == "not_downloaded":
        return "Download required"
    if status == "error":
        return "Load failed"
    return "Unknown"


def get_model_catalog_entry(model_id: str) -> Dict[str, Any]:
    """Return UI-friendly status metadata for a model."""
    model_info = AVAILABLE_MODELS.get(model_id, {})
    hf_id = model_info.get("hf_id", "")
    override = model_status_overrides.get(model_id, {})
    task = model_load_tasks.get(model_id)
    task_running = bool(task and not task.done())
    compiled_cached = bool(hf_id and has_npu_cache(hf_id))
    hf_cached = bool(hf_id and has_hf_cache(hf_id))

    if model_id in loaded_models:
        status = "loaded"
        phase = None
    elif override.get("status") == "error":
        status = "error"
        phase = override.get("phase")
    elif task_running:
        status = override.get("status", "loading")
        phase = override.get("phase")
    elif compiled_cached or hf_cached:
        status = "ready_to_load"
        phase = None
    else:
        status = "not_downloaded"
        phase = None

    return {
        "id": model_id,
        "name": model_info.get("name", model_id),
        "description": model_info.get("description", ""),
        "status": status,
        "status_label": get_model_status_label(status, phase),
        "phase": phase,
        "is_loaded": model_id in loaded_models,
        "is_loading": status in {"queued", "loading"},
        "is_downloaded": compiled_cached or hf_cached,
        "has_npu_cache": compiled_cached,
        "has_hf_cache": hf_cached,
        "error": override.get("error"),
    }


async def _load_single_model_task(model_id: str) -> None:
    """Load one model in the background, serializing heavy model work."""
    model_info = AVAILABLE_MODELS[model_id]
    hf_id = model_info["hf_id"]

    try:
        async with model_load_lock:
            if model_id in loaded_models:
                clear_model_status(model_id)
                return

            if has_npu_cache(hf_id):
                set_model_status(model_id, "loading", phase="load_cache")
            elif has_hf_cache(hf_id):
                set_model_status(model_id, "loading", phase="prepare_cache")
            else:
                set_model_status(model_id, "loading", phase="download")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, load_npu_model, model_id, hf_id)
            clear_model_status(model_id)
    except Exception as e:
        loaded_models.pop(model_id, None)
        model_locks.pop(model_id, None)
        set_model_status(model_id, "error", phase="error", error=str(e))
        logger.exception(f"Failed to load model '{model_id}': {e}")


def schedule_model_load(model_id: str) -> Optional[asyncio.Task]:
    """Schedule a background load for a model if needed."""
    if model_id not in AVAILABLE_MODELS:
        logger.warning(f"Unknown model '{model_id}', skipping.")
        return None

    if model_id in loaded_models:
        clear_model_status(model_id)
        return None

    existing_task = model_load_tasks.get(model_id)
    if existing_task and not existing_task.done():
        return existing_task

    set_model_status(model_id, "queued", phase="queued")
    task = asyncio.create_task(_load_single_model_task(model_id))
    model_load_tasks[model_id] = task
    return task


async def _load_models_in_background() -> None:
    """Load configured models without blocking request handling."""
    loaded_models.clear()
    model_locks.clear()
    model_load_tasks.clear()
    model_status_overrides.clear()

    startup_tasks = [schedule_model_load(model_id) for model_id in model_ids_to_load]
    startup_tasks = [task for task in startup_tasks if task is not None]

    if startup_tasks:
        await asyncio.gather(*startup_tasks, return_exceptions=True)

    models_ready.set()
    logger.info("All models loaded. Server ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader_task

    models_ready.clear()
    model_loader_task = asyncio.create_task(_load_models_in_background())

    try:
        yield
    finally:
        if model_loader_task and not model_loader_task.done():
            model_loader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await model_loader_task
        for task in list(model_load_tasks.values()):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        loaded_models.clear()
        model_locks.clear()
        model_load_tasks.clear()
        model_status_overrides.clear()
        models_ready.clear()
        logger.info("Models unloaded.")


app = FastAPI(title="Intel NPU LLM Server", lifespan=lifespan)


@app.get("/", response_class=FileResponse)
async def read_index():
    """Serve the built-in test UI."""
    return FileResponse(
        Path(__file__).parent / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

# --- NPU Model Cache Directory ---
NPU_MODEL_CACHE = os.path.join(os.path.dirname(__file__), "npu_model_cache")

# --- OpenAI API Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

# --- Tool/Function Calling Models ---
class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class ModelLoadRequest(BaseModel):
    model: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", or specific tool

class ChatCompletionMessageWithTools(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessageWithTools
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None

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
    max_context_len, max_prompt_len = get_model_context_limits(model_id)
    
    logger.info(f"Loading '{model_id}' ({hf_model_path}) for Intel NPU...")
    
    npu_env = os.environ.get("IPEX_LLM_NPU_MTL", "not set")
    logger.info(f"NPU Environment: IPEX_LLM_NPU_MTL={npu_env}")
    
    # Create cache directory for NPU model
    model_cache_dir = str(get_npu_cache_dir(hf_model_path))
    
    if not os.path.exists(model_cache_dir):
        # Create parent directories and convert model
        os.makedirs(model_cache_dir, exist_ok=True)
        logger.info(f"Converting model to NPU format (first time only)...")
        logger.info(f"Cache: {model_cache_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_low_bit="sym_int4",
            optimize_model=True,
            max_context_len=max_context_len,
            max_prompt_len=max_prompt_len,
            save_directory=model_cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(model_cache_dir)
        logger.info(f" -> Model converted and cached.")
    else:
        logger.info(f"Loading from cache: {model_cache_dir}")
        model = AutoModelForCausalLM.load_low_bit(
            model_cache_dir,
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
        logger.info(f" -> Loaded from cache.")
    
    loaded_models[model_id] = {
        "model": model,
        "tokenizer": tokenizer,
        "hf_id": hf_model_path
    }
    model_locks[model_id] = asyncio.Lock()
    logger.info(f" ✓ '{model_id}' ready on Intel NPU!")

def load_all_models(model_ids: List[str]):
    """Load all specified models at startup."""
    logger.info(f"Intel NPU LLM Server - Loading {len(model_ids)} Model(s)")
    
    for model_id in model_ids:
        if model_id in AVAILABLE_MODELS:
            hf_id = AVAILABLE_MODELS[model_id]["hf_id"]
            load_npu_model(model_id, hf_id)
        else:
            logger.warning(f"Unknown model '{model_id}', skipping.")
    
    logger.info(f"Total models ready: {len(loaded_models)}")

def get_model_and_tokenizer(model_id: str):
    """Get model and tokenizer for the given model ID."""
    # Try exact match
    if model_id in loaded_models:
        return loaded_models[model_id]["model"], loaded_models[model_id]["tokenizer"]

    if model_id in AVAILABLE_MODELS:
        model_entry = get_model_catalog_entry(model_id)
        if model_entry["is_loading"]:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is still loading")
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is not loaded yet")
    
    # Fallback to default
    if default_model_id in loaded_models:
        return loaded_models[default_model_id]["model"], loaded_models[default_model_id]["tokenizer"]
    
    # Use first available
    if loaded_models:
        first_key = next(iter(loaded_models))
        return loaded_models[first_key]["model"], loaded_models[first_key]["tokenizer"]
    
    raise HTTPException(status_code=500, detail="No models loaded")

# --- Tool Calling Helpers ---
def format_tools_for_prompt(tools: List[ToolDefinition], tool_choice: Any = None) -> str:
    """
    Format tools into a system prompt section for Qwen.
    
    Args:
        tools: List of tool definitions
        tool_choice: "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
    """
    if not tools:
        return ""
    
    # Handle tool_choice="none" - don't include tools at all
    if tool_choice == "none":
        return ""
    
    tools_json = []
    for tool in tools:
        tools_json.append({
            "type": tool.type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "parameters": tool.function.parameters or {"type": "object", "properties": {}}
            }
        })
    
    # Filter to specific tool if tool_choice specifies one
    forced_tool = None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        forced_tool = tool_choice.get("function", {}).get("name")
        if forced_tool:
            tools_json = [t for t in tools_json if t["function"]["name"] == forced_tool]
    
    tools_str = json.dumps(tools_json, indent=2)
    
    # Build instruction based on tool_choice
    if forced_tool:
        tool_instruction = f"You MUST call the '{forced_tool}' function. Do not respond with anything else."
    elif tool_choice == "required":
        tool_instruction = "You MUST call at least one of the available tools. Do not respond without calling a tool."
    else:  # "auto" or None
        tool_instruction = "Use the tools when needed to answer the user's questions. If you don't need a tool, respond normally."
    
    return f"""You are a helpful assistant with access to the following tools. {tool_instruction}

# Available Tools

{tools_str}

# Tool Call Format

When you need to call a tool, respond with a JSON object in this EXACT format:
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}

For multiple tool calls, use a JSON array:
[{{"name": "func1", "arguments": {{}}}}, {{"name": "func2", "arguments": {{}}}}]

IMPORTANT: Output ONLY the JSON when calling tools, no other text."""


def parse_tool_calls(text: str, available_tools: List[ToolDefinition] = None) -> tuple[str, List[ToolCall]]:
    """
    Parse tool calls from model output with improved parsing.
    
    Features:
    - Parses single JSON objects
    - Parses JSON arrays
    - Handles code blocks
    - Deduplicates calls
    - Validates against available tools
    
    Returns (remaining_text, list_of_tool_calls).
    """
    tool_calls = []
    seen_calls = set()  # For deduplication
    remaining_text = text
    
    # Get list of valid tool names for validation
    valid_tool_names = set()
    if available_tools:
        valid_tool_names = {t.function.name for t in available_tools}
    
    def add_tool_call(name: str, arguments: str, original_match: str = None):
        """Helper to add a tool call with deduplication."""
        # Skip if not a valid tool name (when validation is enabled)
        if valid_tool_names and name not in valid_tool_names:
            return False
        
        # Dedup key
        dedup_key = f"{name}:{arguments}"
        if dedup_key in seen_calls:
            return False
        seen_calls.add(dedup_key)
        
        tool_calls.append(ToolCall(
            id=f"call-{uuid.uuid4().hex[:12]}",
            type="function",
            function=FunctionCall(name=name, arguments=arguments)
        ))
        return True
    
    # Strategy 1: Try to parse as a full JSON array
    # Look for [...] patterns
    array_pattern = r'\[\s*\{[^[\]]*\}\s*(?:,\s*\{[^[\]]*\}\s*)*\]'
    for match in re.finditer(array_pattern, text, re.DOTALL):
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                valid_array = True
                for item in arr:
                    if isinstance(item, dict) and "name" in item:
                        args = item.get("arguments", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        add_tool_call(item["name"], args_str, match.group(0))
                    else:
                        valid_array = False
                if valid_array and arr:
                    remaining_text = remaining_text.replace(match.group(0), "").strip()
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Parse individual JSON objects (more lenient)
    # Match {"name": "...", "arguments": {...}} patterns
    json_patterns = [
        # Standard format
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}',
        # Reversed order
        r'\{\s*"arguments"\s*:\s*(\{[^{}]*\})\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}',
    ]
    
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                if "arguments" in pattern[:30]:  # Reversed pattern
                    fn_args, fn_name = match.group(1), match.group(2)
                else:
                    fn_name, fn_args = match.group(1), match.group(2)
                
                # Validate arguments JSON
                json.loads(fn_args)
                
                if add_tool_call(fn_name, fn_args, match.group(0)):
                    remaining_text = remaining_text.replace(match.group(0), "").strip()
            except (json.JSONDecodeError, Exception):
                continue
    
    # Strategy 3: Parse code blocks with JSON
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    
    for pattern in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                
                # Handle both single object and array
                items = parsed if isinstance(parsed, list) else [parsed]
                
                for item in items:
                    if isinstance(item, dict) and "name" in item:
                        args = item.get("arguments", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        if add_tool_call(item["name"], args_str, match.group(0)):
                            remaining_text = remaining_text.replace(match.group(0), "").strip()
            except (json.JSONDecodeError, Exception):
                continue
    
    # Clean up remaining text
    remaining_text = re.sub(r'\s+', ' ', remaining_text).strip()
    
    return remaining_text, tool_calls


def detect_incomplete_tool_call(text: str) -> bool:
    """
    Detect if the model output contains an incomplete tool call JSON.
    Used for retry logic.
    """
    # Check for unclosed braces/brackets that look like tool calls
    if '{"name"' in text or "[{" in text:
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_braces > 0 or open_brackets > 0:
            return True
    return False


def get_retry_prompt() -> str:
    """Get a prompt to fix malformed tool call output."""
    return """Your previous response contained a malformed tool call. Please try again.
Output ONLY valid JSON in this format:
{"name": "function_name", "arguments": {"param": "value"}}"""


def _format_chatml_message(message: ChatMessage) -> str:
    """Format a single message into a ChatML block."""
    content = message.content or ""

    if message.role == "tool" and message.tool_call_id:
        return (
            f"<|im_start|>tool\n"
            f"Call ID: {message.tool_call_id}\n"
            f"Result: {content}\n"
            f"<|im_end|>\n"
        )

    if message.role == "assistant" and message.tool_calls:
        tool_calls_formatted = []
        for tool_call in message.tool_calls:
            fn = tool_call.get("function", {})
            tool_calls_formatted.append({
                "id": tool_call.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}")
            })
        return f"<|im_start|>assistant\n{json.dumps(tool_calls_formatted)}<|im_end|>\n"

    return f"<|im_start|>{message.role}\n{content}<|im_end|>\n"


def _encode_len(tokenizer, text: str) -> int:
    """Return the token length for a text snippet."""
    encoded = tokenizer.encode(text, return_tensors="pt")
    if hasattr(encoded, "shape"):
        if len(encoded.shape) == 1:
            return int(encoded.shape[0])
        return int(encoded.shape[1])
    return int(len(encoded))


def _truncate_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to the first max_tokens tokens."""
    if max_tokens <= 0:
        return ""

    encoded = tokenizer.encode(text, return_tensors="pt")
    if hasattr(encoded, "shape") and len(encoded.shape) > 1:
        return tokenizer.decode(encoded[0][:max_tokens], skip_special_tokens=False)
    return tokenizer.decode(encoded[:max_tokens], skip_special_tokens=False)


def build_prompt_sliding_window(
    messages: List[ChatMessage],
    tokenizer,
    max_prompt_len: int,
    system_override: str = "",
) -> tuple[str, int]:
    """
    Build a ChatML prompt that fits within max_prompt_len tokens using a
    sliding window strategy.

    Strategy:
    1. Always include the system message (tools injection or user system prompt)
    2. Always include the LAST (most recent) user message
    3. Fill remaining token budget with as many prior turns as possible,
       working backwards from the newest message
    4. Never truncate mid-message — drop whole turns only

    Returns (prompt_string, token_count).
    """
    assistant_marker = "<|im_start|>assistant\n"
    assistant_tokens = _encode_len(tokenizer, assistant_marker)
    remaining_budget = max(max_prompt_len - assistant_tokens, 0)

    system_block = ""
    system_index: Optional[int] = None

    if system_override:
        system_block = f"<|im_start|>system\n{system_override}<|im_end|>\n"
        if messages and messages[0].role == "system":
            system_index = 0
    elif messages and messages[0].role == "system":
        system_index = 0
        system_block = _format_chatml_message(messages[0])

    if system_block:
        system_tokens = _encode_len(tokenizer, system_block)
        if system_tokens > remaining_budget:
            logger.warning(
                "System block exceeded prompt budget; truncating system prompt to fit the reserved assistant marker"
            )
            system_block = _truncate_text_to_tokens(tokenizer, system_block, remaining_budget)
            system_tokens = _encode_len(tokenizer, system_block)
        remaining_budget = max(remaining_budget - system_tokens, 0)

    last_user_index = next(
        (
            idx for idx in range(len(messages) - 1, -1, -1)
            if idx != system_index and messages[idx].role == "user"
        ),
        None,
    )

    last_user_block = ""
    last_user_tokens = 0
    if last_user_index is not None:
        last_user_block = _format_chatml_message(messages[last_user_index])
        last_user_tokens = _encode_len(tokenizer, last_user_block)

        if system_block and last_user_tokens > remaining_budget:
            target_system_budget = max(max_prompt_len - assistant_tokens - last_user_tokens, 0)
            if target_system_budget < _encode_len(tokenizer, system_block):
                logger.warning(
                    "Truncating system prompt further to preserve the most recent user turn in the sliding window"
                )
                system_block = _truncate_text_to_tokens(tokenizer, system_block, target_system_budget)
                remaining_budget = max(max_prompt_len - assistant_tokens - _encode_len(tokenizer, system_block), 0)

    candidate_indices = [
        idx for idx in range(len(messages) - 1, -1, -1)
        if idx != system_index
    ]

    selected_blocks_reversed: List[str] = []
    last_user_included = last_user_index is None
    dropped_turns = 0

    for position, idx in enumerate(candidate_indices):
        block = _format_chatml_message(messages[idx])
        block_tokens = _encode_len(tokenizer, block)
        reserved_budget = 0

        if last_user_index is not None and not last_user_included and idx != last_user_index:
            reserved_budget = last_user_tokens

        if block_tokens + reserved_budget <= remaining_budget:
            selected_blocks_reversed.append(block)
            remaining_budget -= block_tokens
            if idx == last_user_index:
                last_user_included = True
            continue

        dropped_turns += 1

        if last_user_index is not None and not last_user_included and idx > last_user_index:
            continue

        dropped_turns += len(candidate_indices) - position - 1
        break

    selected_blocks = list(reversed(selected_blocks_reversed))
    prompt = system_block + "".join(selected_blocks) + assistant_marker
    input_length = _encode_len(tokenizer, prompt)

    while input_length > max_prompt_len and selected_blocks:
        dropped_turns += 1
        selected_blocks.pop(0)
        prompt = system_block + "".join(selected_blocks) + assistant_marker
        input_length = _encode_len(tokenizer, prompt)

    if input_length > max_prompt_len and system_block:
        system_budget = max(max_prompt_len - assistant_tokens, 0)
        system_block = _truncate_text_to_tokens(tokenizer, system_block, system_budget)
        prompt = system_block + "".join(selected_blocks) + assistant_marker
        input_length = _encode_len(tokenizer, prompt)

    if dropped_turns > 0:
        logger.info(
            f"Sliding window: dropped {dropped_turns} older turn(s) to fit {max_prompt_len} token budget"
        )

    return prompt, input_length


# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model, tokenizer = get_model_and_tokenizer(request.model)
    max_context_len, max_prompt_len = get_model_context_limits(request.model)
    
    MAX_RETRY_ATTEMPTS = 2  # For malformed tool calls
    
    # Check if tools are disabled via tool_choice
    use_tools = request.tools and request.tool_choice != "none"

    system_override = ""
    if use_tools:
        tools_prompt = format_tools_for_prompt(request.tools, request.tool_choice)
        if tools_prompt:
            system_override = tools_prompt

    prompt, input_length = build_prompt_sliding_window(
        messages=request.messages,
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
        system_override=system_override,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Cap max_new_tokens to stay within context limit
    available_tokens = max_context_len - input_length - 10
    max_new_tokens = min(request.max_tokens or 512, available_tokens, 500)
    max_new_tokens = max(max_new_tokens, 10)
    
    # Generation config for NPU
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # --- Streaming Response with Tool Call Detection ---
    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        
        async def generate_with_lock():
            global is_generating
            async with get_model_lock(request.model):
                is_generating = True
                try:
                    await asyncio.get_running_loop().run_in_executor(None, lambda: model.generate(input_ids, **gen_kwargs))
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                finally:
                    is_generating = False
        
        # Start generation in a background task
        asyncio.create_task(generate_with_lock())

        async def stream_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            accumulated_text = ""
            buffered_chunks: List[str] = []
            
            for text in streamer:
                accumulated_text += text
                buffered_chunks.append(text)
            
            finish_reason = "stop"
            _, parsed_tools = (accumulated_text, [])
            if use_tools:
                _, parsed_tools = parse_tool_calls(accumulated_text, request.tools)

            if parsed_tools:
                finish_reason = "tool_calls"

                initial_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": None},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"

                for i, tc in enumerate(parsed_tools):
                    tool_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": i,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(tool_chunk)}\n\n"
            else:
                for text in buffered_chunks:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Calculate completion tokens unconditionally
            completion_tokens = len(tokenizer.encode(accumulated_text))
            
            # Send the normal finish chunk
            end_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            
            # OpenAI specification: yield one final chunk with an empty choices array and the usage object
            if request.stream_options and request.stream_options.include_usage:
                usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": input_length,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_length + completion_tokens
                    }
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
                
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Standard Response with Retry Logic ---
    else:
        generated_text = ""
        retry_count = 0
        current_input_ids = input_ids
        global is_generating  # Must be declared at function scope, not inside loops
        
        while retry_count <= MAX_RETRY_ATTEMPTS:
            async with get_model_lock(request.model):
                is_generating = True
                try:
                    with torch.no_grad():
                        output_ids = await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: model.generate(current_input_ids, **gen_kwargs)
                        )
                finally:
                    is_generating = False
            
            generated_text = tokenizer.decode(output_ids[0][current_input_ids.shape[1]:], skip_special_tokens=True)
            
            # Check if we need tools and got malformed output
            if use_tools and detect_incomplete_tool_call(generated_text):
                retry_count += 1
                if retry_count <= MAX_RETRY_ATTEMPTS:
                    logger.warning(f"Malformed tool call detected, retry {retry_count}/{MAX_RETRY_ATTEMPTS}")
                    retry_messages = list(request.messages) + [
                        ChatMessage(role="assistant", content=generated_text),
                        ChatMessage(role="user", content=get_retry_prompt()),
                    ]
                    retry_prompt, _ = build_prompt_sliding_window(
                        messages=retry_messages,
                        tokenizer=tokenizer,
                        max_prompt_len=max_prompt_len,
                        system_override=system_override,
                    )
                    current_input_ids = tokenizer.encode(retry_prompt, return_tensors="pt")
                    continue
            break
        
        # Parse tool calls if tools were requested
        tool_calls_list = None
        finish_reason = "stop"
        response_content = generated_text
        
        if use_tools:
            remaining_text, parsed_tool_calls = parse_tool_calls(generated_text, request.tools)
            if parsed_tool_calls:
                tool_calls_list = parsed_tool_calls
                finish_reason = "tool_calls"
                response_content = remaining_text if remaining_text else None
                logger.info(f"Parsed {len(parsed_tool_calls)} tool call(s)")

        # Calculate tokens
        prompt_tokens = input_length
        completion_tokens = int(output_ids.shape[1] - current_input_ids.shape[1])
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionMessageWithTools(
                        role="assistant", 
                        content=response_content,
                        tool_calls=tool_calls_list
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    """
    OpenAI Responses API endpoint (for N8N compatibility).
    Converts Responses API format to internal format and returns response.
    """
    model, tokenizer = get_model_and_tokenizer(request.model)
    max_context_len, max_prompt_len = get_model_context_limits(request.model)
    
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
    if input_length > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
        input_length = max_prompt_len
        logger.warning(f"Input truncated to {max_prompt_len} tokens")
    
    # Cap max tokens
    available_tokens = max_context_len - input_length - 10
    max_new_tokens = min(request.max_output_tokens or 512, available_tokens, 500)
    max_new_tokens = max(max_new_tokens, 10)
    
    # Generation config
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    
    # Generate response
    global is_generating
    async with get_model_lock(request.model):
        is_generating = True
        try:
            with torch.no_grad():
                output_ids = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: model.generate(input_ids, **gen_kwargs)
                )
        finally:
            is_generating = False
    
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    completion_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    
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


@app.post("/v1/models/load")
async def load_model(request: ModelLoadRequest):
    """Queue a model for local download/compile/load if needed."""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'")

    task = schedule_model_load(request.model)
    model_entry = get_model_catalog_entry(request.model)

    if task is None and model_entry["is_loaded"]:
        return {
            "status": "loaded",
            "message": f"{model_entry['name']} is already ready.",
            "model": model_entry
        }

    return {
        "status": model_entry["status"],
        "message": f"Loading {model_entry['name']} locally. This can take a while on first run.",
        "model": model_entry
    }

@app.get("/health")
async def health():
    if not models_ready.is_set():
        return {"status": "loading", "models_loaded": 0}
    return {"status": "ok"}

def _dir_size_gb(path: str) -> float:
    """Return total size of a directory in GB, or 0.0 if it doesn't exist."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except Exception:
        pass
    return round(total / (1024**3), 2)

@app.get("/v1/system/status")
async def system_status():
    """Return system resource usage including model disk footprint."""
    vm = psutil.virtual_memory()

    # Disk sizes
    npu_cache_gb = _dir_size_gb(NPU_MODEL_CACHE)
    hf_home = get_hf_home_dir()
    hf_hub_path = os.path.join(hf_home, "hub")
    hf_cache_gb = _dir_size_gb(hf_hub_path)
    available_models = [get_model_catalog_entry(model_id) for model_id in AVAILABLE_MODELS]
    loading_count = sum(1 for model in available_models if model["is_loading"])

    return {
        "memory": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_percent": vm.percent
        },
        "cpu": {
            "percent": psutil.cpu_percent(interval=None)
        },
        "models": {
            "loaded": list(loaded_models.keys()),
            "count": len(loaded_models),
            "loading_count": loading_count,
            "available": available_models
        },
        "npu": {
            "config": os.environ.get("IPEX_LLM_NPU_MTL", "non-MTL"),
            "busy": any(lock.locked() for lock in model_locks.values()) or npu_resource_lock.locked(),
            "model_locks": {
                mid: model_locks[mid].locked()
                for mid in model_locks
            }
        },
        "disk": {
            "npu_cache_gb": npu_cache_gb,
            "hf_cache_gb": hf_cache_gb,
            "total_gb": round(npu_cache_gb + hf_cache_gb, 2)
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel NPU LLM Server")
    parser.add_argument(
        "--models",
        type=str,
        default="qwen1.5-1.8b",
        help="Comma-separated list of models to load (e.g., 'qwen1.5-1.8b,qwen2-1.5b,deepseek-1.5b')"
    )
    
    # Use PORT environment variable as default if available
    default_port = int(os.environ.get("PORT", 8000))
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to run server on (default: {default_port})")
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
    model_ids_to_load = [m.strip() for m in args.models.split(",") if m.strip()]

    logger.info(f"Server starting! Visit: http://localhost:{args.port}")
    logger.info(f"Models requested: {', '.join(model_ids_to_load)}")
    
    # Bind to all interfaces (0.0.0.0) but uvicorn will still log 0.0.0.0 by default.
    # To avoid confusing the user, we print a clear URL above.
    uvicorn.run(app, host="0.0.0.0", port=args.port)
