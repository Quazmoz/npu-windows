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
- **API Endpoint**: http://localhost:8000/v1/chat/completions

---

## 🤖 Supported Models

Models verified compatible with ipex-llm NPU:

### Qwen Series (Recommended)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `qwen1.5-1.8b` | 1.8B | ~8 tok/s | ✅ **Default** - Verified working |
| `qwen1.5-4b` | 4B | ~5 tok/s | Better quality |
| `qwen1.5-7b` | 7B | ~3 tok/s | Best Qwen1.5 |
| `qwen2-1.5b` | 1.5B | ~10 tok/s | Fast |
| `qwen2-7b` | 7B | ~3 tok/s | Officially supported |

### Llama Series
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `llama2-7b` | 7B | ~3 tok/s | Classic, requires HF login |
| `llama3-8b` | 8B | ~2 tok/s | Best Llama, requires HF login |

### DeepSeek R1 (Reasoning)
| Model ID | Size | NPU Speed | Notes |
|----------|------|-----------|-------|
| `deepseek-1.5b` | 1.5B | ~10 tok/s | Fast reasoning |
| `deepseek-7b` | 7B | ~3 tok/s | Best reasoning |

### Load Multiple Models

```powershell
.\start_backend.bat --models "qwen1.5-1.8b,qwen1.5-4b,deepseek-1.5b"
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

## 📁 Project Structure

```
npu-185H/
├── start_backend.bat          # Easy startup script
├── intel-npu-llm/
│   ├── npu_server.py          # NPU-accelerated LLM server
│   ├── docker-compose.yml     # Open WebUI frontend
│   ├── npu_model_cache/       # Compiled NPU models (auto-created)
│   └── test_npu.py            # NPU test script
└── README.md
```

---

## 📄 License

MIT License
