# Quick Start Guide

Get your Intel NPU running LLMs in 5 minutes.

## Prerequisites

- Intel Core Ultra processor (Series 1, 2, or Lunar Lake)
- Windows 11
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Open WebUI)

## Step 1: Install Dependencies (First Time)

Open PowerShell:

```powershell
# Clone the repo
git clone https://github.com/yourusername/npu-185H.git
cd npu-185H

# Install Miniconda if needed
winget install Anaconda.Miniconda3

# Restart PowerShell, then:
conda create -n ipex-npu python=3.11 -y
conda activate ipex-npu
pip install --pre --upgrade ipex-llm[npu]
pip install fastapi uvicorn pydantic
```

## Step 2: Start the Backend

```powershell
.\start_backend.bat
```

Wait for:
```
Detected CPU: Intel(R) Core(TM) Ultra 9 185H
Processor: Intel Core Ultra Series 1 - Meteor Lake
✓ qwen1.5-1.8b ready on Intel NPU!
Server starting on http://0.0.0.0:8000
```

## Step 3: Start Open WebUI

In a new terminal:
```powershell
cd intel-npu-llm
docker compose up -d
```

## Step 4: Chat!

Open http://localhost:3000

Select a model from the dropdown and start chatting. Check **Task Manager → NPU** to see it working!

---

## Available Models

Models verified compatible with ipex-llm NPU:

| Model ID | Size | Speed | Best For |
|----------|------|-------|----------|
| `qwen1.5-1.8b` | 1.8B | ~8 tok/s | ✅ Default, fast |
| `qwen1.5-4b` | 4B | ~5 tok/s | Better quality |
| `qwen1.5-7b` | 7B | ~3 tok/s | Best Qwen |
| `qwen2-1.5b` | 1.5B | ~10 tok/s | Fast |
| `qwen2-7b` | 7B | ~3 tok/s | Officially supported |
| `llama2-7b` | 7B | ~3 tok/s | Classic (needs HF login) |
| `llama3-8b` | 8B | ~2 tok/s | Best Llama (needs HF login) |
| `deepseek-1.5b` | 1.5B | ~10 tok/s | Reasoning |
| `deepseek-7b` | 7B | ~3 tok/s | Best reasoning |

## Loading Different Models

```powershell
# Load specific models
.\start_backend.bat --models "qwen1.5-1.8b,qwen1.5-4b"

# Load multiple models
.\start_backend.bat --models "qwen1.5-1.8b,deepseek-1.5b,qwen2-7b"

# List all available models
.\start_backend.bat --list
```

---

## Stopping

```powershell
# Stop backend: Ctrl+C in the terminal

# Stop Open WebUI:
cd intel-npu-llm
docker compose down
```

## Next Steps

- See [README.md](README.md) for full documentation
