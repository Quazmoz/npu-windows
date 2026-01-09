# Intel NPU LLM Server

Run Large Language Models on your Intel Core Ultra NPU with an OpenAI-compatible API.

## 🎯 Features

- **NPU Acceleration**: Leverage Intel's Neural Processing Unit for power-efficient AI
- **OpenAI-Compatible API**: Works with any OpenAI client, including Open WebUI
- **Local & Private**: All processing happens on your device
- **Power Efficient**: ~3-5x less power than CPU inference

## 📋 Requirements

- **Processor**: Intel Core Ultra (Meteor Lake, Arrow Lake, or Lunar Lake)
- **OS**: Windows 11
- **NPU Driver**: Version 32.0.100.3104 or newer
- **Python**: 3.11 (managed via Miniconda)
- **Docker Desktop**: For Open WebUI frontend (optional)

## 🚀 Quick Start

### 1. Install Dependencies (First Time Only)

```powershell
# Install Miniconda (if not installed)
winget install Anaconda.Miniconda3

# Create Python environment
conda create -n ipex-npu python=3.11 -y
conda activate ipex-npu

# Install ipex-llm with NPU support
pip install --pre --upgrade ipex-llm[npu]
pip install fastapi uvicorn pydantic
```

### 1b. HuggingFace Authentication (For Gated Models)

Some models (Llama 2, Llama 3, Llama 3.2) require HuggingFace authentication:

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co)
2. **Accept the model license** - Visit the model page (e.g., [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)) and accept the terms
3. **Generate an access token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Create a `.env` file** in the project root:

```powershell
# Create .env file with your token
echo 'HF_TOKEN=hf_your_token_here' > .env
```

Or manually create `npu-windows/.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Note**: Without this, gated models will fail to download. Non-gated models (Qwen, DeepSeek, MiniCPM, GLM-Edge, Baichuan2) work without authentication.

### 2. Start the NPU Backend (Multiple Models)

```powershell
# From the project root - loads 2 models by default
.\start_backend.bat
```

Or load specific models:
```powershell
.\start_backend.bat --models "qwen-1.8b,phi3-mini,qwen-4b"
```

List all available models:
```powershell
.\start_backend.bat --list
```

Or manually:
```powershell
$env:IPEX_LLM_NPU_MTL = "1"  # For Meteor Lake (Core Ultra Series 1)
conda activate ipex-npu
cd intel-npu-llm
python npu_server.py
```

### 3. Start Open WebUI (Optional)

```powershell
cd intel-npu-llm
docker compose up -d
```

### 4. Access the Interface

- **Open WebUI**: http://localhost:3000
- **API Endpoints**:
  - `/v1/chat/completions` - OpenAI Chat Completions API (Open WebUI, LangChain)
  - `/v1/responses` - OpenAI Responses API (N8N, newer tools)
  - `/v1/models` - List available models

### 5. Connect Your Own Open WebUI (Optional)

If you already have Open WebUI running elsewhere (e.g., on a homelab server), configure it to use your NPU server:

1. **In Open WebUI**: Go to **Settings → Connections → OpenAI API**
2. **Add a new connection** with these settings:
   - **API Base URL**: `http://<YOUR-WINDOWS-PC-IP>:8000/v1`
   - **API Key**: `sk-dummy` (any value works, the NPU server doesn't validate keys)
3. **Save** and your NPU models will appear in the model dropdown

> **Tip**: Find your Windows IP with `ipconfig` in PowerShell. Use your local network IP (e.g., `192.168.1.x`).

> **Firewall Note**: You may need to allow port 8000 through Windows Firewall for remote connections.

### 6. Connect N8N (Optional)

To use your NPU server with N8N workflows:

1. **In N8N**: Add an **OpenAI** node to your workflow
2. **Configure credentials**:
   - **API Key**: `sk-dummy` (any value)
   - **Base URL**: `http://<YOUR-WINDOWS-PC-IP>:8000/v1`
3. **Select model**: Use one of the loaded model IDs (e.g., `qwen1.5-1.8b`)

> **Note**: N8N uses the `/v1/responses` API endpoint, which is fully supported.

### 7. Tool Calling / Function Calling (Agents)

The server supports OpenAI-compatible tool/function calling for building AI agents:

```json
{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    "tool_choice": "auto"
}
```

#### Tool Choice Options

| `tool_choice` | Behavior |
|---------------|----------|
| `"auto"` | Model decides when to use tools (default) |
| `"none"` | Disable tool calling, respond normally |
| `"required"` | Force the model to call at least one tool |
| `{"type": "function", "function": {"name": "get_weather"}}` | Force specific tool |

#### Advanced Features

- **Parallel tool calls**: Model can call multiple tools in one response
- **Streaming tool calls**: Tool calls are detected and emitted at end of stream
- **Retry logic**: Malformed tool calls are automatically retried (max 2 attempts)
- **Tool validation**: Only defined tools are parsed, invalid calls are ignored

**Recommended models**: `qwen2.5-7b`, `qwen2.5-3b` (larger models work better)

> **Note**: Tool calling works best with 3B+ parameter models. Smaller models may struggle.

---

## 🤖 Supported Models

All models below are **officially verified** for Intel NPU via ipex-llm:

### Qwen Series (Recommended)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `qwen1.5-1.8b` | 1.8B | ~8 tok/s | ✅ **Default** - Verified working |
| `qwen1.5-4b` | 4B | ~5 tok/s | Better quality |
| `qwen1.5-7b` | 7B | ~3 tok/s | Best Qwen1.5 |
| `qwen2-1.5b` | 1.5B | ~10 tok/s | Official NPU verified |
| `qwen2-7b` | 7B | ~3 tok/s | Official NPU verified |
| `qwen2.5-3b` | 3B | ~8 tok/s | 🔥 **Latest Qwen** |
| `qwen2.5-7b` | 7B | ~3 tok/s | 🔥 Best Qwen 2.5 |

### Llama Series
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `llama2-7b` | 7B | ~3 tok/s | Classic, requires HF login |
| `llama3-8b` | 8B | ~2 tok/s | Powerful, requires HF login |
| `llama3.2-1b` | 1B | ~15 tok/s | ⚡ **Fastest Llama**, requires HF login |
| `llama3.2-3b` | 3B | ~10 tok/s | Fast & capable, requires HF login |

### DeepSeek R1 (Reasoning)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `deepseek-1.5b` | 1.5B | ~10 tok/s | Fast reasoning |
| `deepseek-7b` | 7B | ~3 tok/s | Best reasoning |

### GLM-Edge (Bilingual)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `glm-edge-1.5b` | 1.5B | ~10 tok/s | Chinese/English bilingual |
| `glm-edge-4b` | 4B | ~5 tok/s | Larger bilingual model |

### MiniCPM (Ultra-Compact)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `minicpm-1b` | 1B | ~15 tok/s | Ultra-compact, efficient |
| `minicpm-2b` | 2B | ~10 tok/s | Small but capable |

### Baichuan2 (Chinese)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `baichuan2-7b` | 7B | ~3 tok/s | Chinese-focused LLM |

### Load Multiple Models

```powershell
.\start_backend.bat --models "qwen2.5-3b,llama3.2-1b,minicpm-2b"
```

> **Note**: First run downloads and compiles each model (1-3 min). Subsequent loads are instant from cache.

---

## ⚡ NPU vs CPU/GPU

| Metric | NPU | CPU | iGPU |
|--------|-----|-----|------|
| Power Draw | ~5-10W | 15-45W | 20-35W |
| TOPS (INT8) | 11 TOPS | ~2-3 TOPS | ~8 TOPS |
| Battery Life | Hours | ~1 hour | ~2 hours |
| Best For | Efficiency | Fallback | Larger models |

---

## 🔧 Configuration

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `IPEX_LLM_NPU_MTL` | `1` | Required for Meteor Lake (Core Ultra Series 1) |
| `HF_HOME` | path | Hugging Face cache directory |

### Processor-Specific Settings

| Processor Series | Environment Variable |
|------------------|---------------------|
| Core Ultra Series 1 (Meteor Lake) | `IPEX_LLM_NPU_MTL=1` |
| Core Ultra Series 2 (Arrow Lake) | None required |
| Core Ultra (Lunar Lake) | None required |

---

## 🐛 Troubleshooting

### NPU Not Detected
1. Check Device Manager → Neural processors → Intel(R) AI Boost
2. Update NPU driver to latest version
3. Ensure `IPEX_LLM_NPU_MTL=1` is set for Meteor Lake

### Generation Hangs
- First generation takes 1-3 minutes for NPU warmup
- Subsequent generations are fast (~1 second)

### Port Already in Use
```powershell
# Kill existing Python processes
Get-Process python* | Stop-Process -Force
```

---

## 💾 Model Storage

Models are stored in two locations:

| Location | Contents | Path |
|----------|----------|------|
| **HuggingFace Cache** | Original downloaded models | `%USERPROFILE%\.cache\huggingface\hub\` |
| **NPU Cache** | Compiled NPU-optimized models | `intel-npu-llm\npu_model_cache\` |

### Space Usage (Approximate)

| Model Size | HF Cache | NPU Cache | Total |
|------------|----------|-----------|-------|
| 1-2B models | ~2-4 GB | ~1-2 GB | ~3-6 GB |
| 3-4B models | ~6-8 GB | ~2-4 GB | ~8-12 GB |
| 7-8B models | ~14-16 GB | ~4-8 GB | ~18-24 GB |

### Clear Cache

```powershell
# Clear NPU cache only (will recompile on next run)
Remove-Item -Recurse -Force .\intel-npu-llm\npu_model_cache\

# Clear HuggingFace cache (will re-download models)
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub\
```

### Custom Cache Location

Set in your `.env` file to store models on a different drive:
```
HF_HOME=D:\models\huggingface
```

---

## 📁 Project Structure

```
npu-windows/
├── start_backend.bat          # Easy startup script
├── intel-npu-llm/
│   ├── npu_server.py          # NPU-accelerated LLM server
│   ├── docker-compose.yml     # Open WebUI frontend
│   └── npu_model_cache/       # Compiled NPU models (auto-created)
└── README.md
```

---

## 📄 License

MIT License
