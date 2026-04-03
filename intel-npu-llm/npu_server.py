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
from pathlib import Path
from threading import Thread
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

app = FastAPI(title="Intel NPU LLM Server")

@app.get("/", response_class=FileResponse)
async def read_index():
    """Serve the built-in test UI."""
    return FileResponse(Path(__file__).parent / "index.html")

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

# --- Global State ---
loaded_models: Dict[str, Any] = {}
default_model_id = "qwen1.5-1.8b"  # Use the verified working model
npu_resource_lock = asyncio.Lock()  # Ensure only one generation at a time
is_generating = False  # Explicit state for tracking

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
    
    logger.info(f"Loading '{model_id}' ({hf_model_path}) for Intel NPU...")
    
    npu_env = os.environ.get("IPEX_LLM_NPU_MTL", "not set")
    logger.info(f"NPU Environment: IPEX_LLM_NPU_MTL={npu_env}")
    
    # Create cache directory for NPU model
    model_cache_dir = os.path.join(NPU_MODEL_CACHE, hf_model_path.replace("/", "_"))
    
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
            max_context_len=1024,
            max_prompt_len=512,
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
    import re
    
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


# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model, tokenizer = get_model_and_tokenizer(request.model)
    
    # NPU model context limits (set during compilation)
    MAX_CONTEXT_LEN = 1024
    MAX_PROMPT_LEN = 512
    MAX_RETRY_ATTEMPTS = 2  # For malformed tool calls
    
    # Check if tools are disabled via tool_choice
    use_tools = request.tools and request.tool_choice != "none"
    
    # Build prompt with tool support
    prompt = ""
    has_system = False
    
    # If tools are provided (and not disabled), inject them into the system prompt
    if use_tools:
        tools_prompt = format_tools_for_prompt(request.tools, request.tool_choice)
        if tools_prompt:  # Will be empty if tool_choice="none"
            prompt += f"<|im_start|>system\n{tools_prompt}<|im_end|>\n"
            has_system = True
    
    # Format messages (ChatML format for Qwen/compatible models)
    for msg in request.messages:
        # Skip system message if we already added tools as system
        if msg.role == "system" and has_system:
            continue
        
        content = msg.content or ""
        
        # Handle tool results with better formatting
        if msg.role == "tool" and msg.tool_call_id:
            prompt += f"<|im_start|>tool\n"
            prompt += f"Call ID: {msg.tool_call_id}\n"
            prompt += f"Result: {content}\n"
            prompt += "<|im_end|>\n"
        # Handle assistant messages with tool calls
        elif msg.role == "assistant" and msg.tool_calls:
            # Format tool calls as clean JSON
            tool_calls_formatted = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {})
                tool_calls_formatted.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}")
                })
            prompt += f"<|im_start|>assistant\n{json.dumps(tool_calls_formatted)}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>{msg.role}\n{content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"

    # Encode and check length
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_ids.shape[1]
    
    # Truncate input if too long (keep last MAX_PROMPT_LEN tokens)
    if input_length > MAX_PROMPT_LEN:
        input_ids = input_ids[:, -MAX_PROMPT_LEN:]
        input_length = MAX_PROMPT_LEN
        logger.warning(f"Input truncated to {MAX_PROMPT_LEN} tokens")
    
    # Cap max_new_tokens to stay within context limit
    available_tokens = MAX_CONTEXT_LEN - input_length - 10
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
            async with npu_resource_lock:
                is_generating = True
                try:
                    await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate(input_ids, **gen_kwargs))
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                finally:
                    is_generating = False
        
        # Start generation in a background task
        asyncio.create_task(generate_with_lock())

        async def stream_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            accumulated_text = ""
            tool_calls_emitted = False
            
            for text in streamer:
                accumulated_text += text
                
                # Check if we should parse tool calls (at the end)
                # For now, stream content normally
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                # When stream_options.include_usage is true, exclude usage from normal chunks
                if request.stream_options and request.stream_options.include_usage:
                    pass # Don't add usage to normal chunks
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # At the end, check if we detected tool calls
            finish_reason = "stop"
            if use_tools:
                _, parsed_tools = parse_tool_calls(accumulated_text, request.tools)
                if parsed_tools:
                    finish_reason = "tool_calls"
                    # Emit tool calls as final chunks
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
            async with npu_resource_lock:
                is_generating = True
                try:
                    with torch.no_grad():
                        output_ids = await asyncio.get_event_loop().run_in_executor(
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
                    # Add retry prompt
                    retry_prompt = prompt + generated_text + "<|im_end|>\n"
                    retry_prompt += f"<|im_start|>user\n{get_retry_prompt()}<|im_end|>\n"
                    retry_prompt += "<|im_start|>assistant\n"
                    current_input_ids = tokenizer.encode(retry_prompt, return_tensors="pt")
                    # Re-truncate if needed
                    if current_input_ids.shape[1] > MAX_PROMPT_LEN:
                        current_input_ids = current_input_ids[:, -MAX_PROMPT_LEN:]
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
        completion_tokens = len(tokenizer.encode(generated_text))
        
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
        logger.warning(f"Input truncated to {MAX_PROMPT_LEN} tokens")
    
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
    
    # Generate response
    global is_generating
    async with npu_resource_lock:
        is_generating = True
        try:
            with torch.no_grad():
                output_ids = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate(input_ids, **gen_kwargs)
                )
        finally:
            is_generating = False
    
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
    return {
        "status": "ok", 
        "models_loaded": len(loaded_models),
        "npu_lock_status": "locked" if npu_resource_lock.locked() else "free"
    }

@app.get("/v1/system/status")
async def system_status():
    """Return system resource usage."""
    vm = psutil.virtual_memory()
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
            "count": len(loaded_models)
        },
        "npu": {
            "config": os.environ.get("IPEX_LLM_NPU_MTL", "non-MTL"),
            "busy": is_generating or npu_resource_lock.locked()
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
    model_ids = [m.strip() for m in args.models.split(",")]
    
    # Load models
    load_all_models(model_ids)
    
    logger.info(f"Server starting on http://0.0.0.0:{args.port}")
    logger.info(f"Models available: {', '.join(loaded_models.keys())}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
