# Quick Start Guide

Get your Intel NPU running LLMs in under 10 minutes.

- **🎥 Watch the Demo**: [Intel NPU LLM - Quick UI & Performance Overview](https://youtu.be/00RTemT1Bbs)
- **📺 Full walkthrough**: [Building with Intel NPU & OpenAI API](https://youtu.be/6F6LbR2Xjcg)

## Prerequisites

- Intel Core Ultra processor (Series 1 Meteor Lake, Series 2 Arrow Lake, or Lunar Lake)
- Windows 11
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) *(optional — only needed for Open WebUI)*

---

## Step 1: Install Dependencies *(First Time Only)*

```powershell
# 1. Install Miniconda (if not installed)
winget install Anaconda.Miniconda3

# 2. Restart PowerShell, then create the environment
conda create -n ipex-npu python=3.11 -y
conda activate ipex-npu

# 3. Install Intel NPU support
pip install --pre --upgrade ipex-llm[npu]

# 4. Install server dependencies
pip install -r intel-npu-llm/requirements.txt
```

---

## Step 1b: HuggingFace Token *(Only for Llama Models)*

Qwen, DeepSeek, MiniCPM, GLM-Edge, and Baichuan2 work **without** a token.
Llama 2, Llama 3, and Llama 3.2 require accepting the license and a HF token.

```powershell
# Create the .env file with UTF-8 encoding (important!)
'HF_TOKEN=hf_your_token_here' | Out-File -FilePath intel-npu-llm/.env -Encoding utf8
```

---

## Step 2: Start the Backend

```powershell
.\start_backend.bat
```

Wait for the ready message:
```
✓ 'qwen1.5-1.8b' ready on Intel NPU!
Server starting! Visit: http://localhost:8000
```

---

## Step 3: Open the Built-in UI

Open your browser and go to: **http://localhost:8000**

You'll see a full chat interface with:
- Real-time NPU status (Idle / Busy)
- Conversation history (multi-turn context)
- Markdown rendering
- Keyboard shortcuts (`Enter` to send, `Ctrl+L` to clear)

---

## Step 4: *(Optional)* Start Open WebUI

For a more full-featured experience with plugins and user management:

```powershell
cd intel-npu-llm
docker compose up -d
```

Then open **http://localhost:3000**

---

## Common Options

```powershell
# Load a specific model
.\start_backend.bat --models "qwen2.5-3b"

# Load multiple models (selectable from the UI dropdown)
.\start_backend.bat --models "qwen1.5-1.8b,qwen1.5-4b"

# Use a different port
.\start_backend.bat --port 8001

# List all available models
.\start_backend.bat --list
```

---

## Connect to Remote Open WebUI / N8N

If you have Open WebUI or N8N running elsewhere on your network:

1. Find your Windows IP: run `ipconfig` in PowerShell
2. Set the **API Base URL** to: `http://<YOUR-IP>:8000/v1`
3. Set the **API Key** to: `sk-dummy` *(any value works)*

> **Firewall note**: Allow port 8000 through Windows Firewall for remote access.

---

## Recommended Models

| Model ID | Size | Speed | Notes |
|---|---|---|---|
| `qwen1.5-1.8b` | 1.8B | ~8 tok/s | ✅ Default — fast and reliable |
| `qwen2.5-3b`   | 3B   | ~8 tok/s | 🔥 Best quality/speed balance |
| `qwen2.5-7b`   | 7B   | ~3 tok/s | Best quality, needs more RAM |
| `deepseek-1.5b`| 1.5B | ~10 tok/s | Reasoning tasks |
| `llama3.2-1b`  | 1B   | ~15 tok/s | ⚡ Fastest (needs HF token) |

> See [README.md](README.md) for the full model list with all options.

---

## Troubleshooting

**Port already in use?**
```powershell
.\start_backend.bat --port 8001
# or kill all Python processes:
Get-Process python* | Stop-Process -Force
```

**`.env` causes `ValueError: embedded null character`?**
- Re-create it with `| Out-File -Encoding utf8` as shown in Step 1b above
- Or open the file in Notepad → File > Save As → **Encoding: UTF-8**

**NPU not detected?**
1. Open Device Manager → Neural processors → should show "Intel(R) AI Boost"
2. Update driver if missing
3. For Meteor Lake (Series 1): ensure `IPEX_LLM_NPU_MTL=1` is set (the bat does this automatically)
