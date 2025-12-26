# 🔌 Connect External Tools (MCP)

This guide explains how to give your NPU-powered LLM access to external tools (like your file system, Google Drive, or Slack) using the **Model Context Protocol (MCP)**.

## What is MCP?
MCP is a standard that lets AI models talk to external "servers" to get data or perform actions. Open WebUI has built-in support for this.

## 🚀 Quick Setup (Example: File System Access)

Since Open WebUI is running in Docker, it can't directly see your Windows files. We use a bridge (proxy) to let it talk to tools running on your Windows host.

### 1. Prerequisites
You need **Node.js** installed on your Windows machine.

### 2. Start the MCP Bridge (Windows Powershell)
Run this command in a new PowerShell window on your host (not inside Docker). It starts a "Filesystem Server" and exposes it via HTTP so Docker can reach it.

```powershell
npx -y @modelcontextprotocol/server-filesystem c:\Users\Public --port 3001
```
*Note: This specific example command is hypothetical as standard MCP servers use stdio. For Open WebUI in Docker, you often need an "SSE Bridge".*

**Better Method: Using `mcpo` (MCP over OpenAPI)**
The easiest way currently to bridge local tools to Open WebUI is to use the `mcpo` wrapper if available, or simply configuring standard HTTP-based MCP servers.

**Simplest Real-World Example:**
If you have an MCP server running on port 3001 on your Windows machine:

### 3. Configure Open WebUI
1.  Go to **[http://localhost:3000](http://localhost:3000)**.
2.  Click on your profile icon -> **Settings**.
3.  Go to **Admin Settings** -> **Connections** (or **External Tools**).
4.  Look for the **MCP** section.
5.  Add a new connection:
    *   **URL**: `http://host.docker.internal:3001/sse`
    *   (This special address lets Docker talk to your Windows host).

### 4. Use it in Chat
1.  Start a new chat.
2.  Look for the **Tools (+)** button or toggles.
3.  Enable your new MCP tool.
4.  Ask: *"Read the files in the public folder"* (or whatever your tool does).

## 🧩 Recommended Tools

*   **Brave Search**: Give your local model internet access.
*   **Filesystem**: Read/Write local files.
*   **Postgres**: Query your databases.

## ⚠️ Note on Docker
Because Open WebUI is in a container, it creates a boundary.
*   **Stdio Tools**: Tools that run directly in the terminal (standard input/output) will **NOT** work directly if configured inside the Docker container unless they are installed *inside* that container.
*   **Hostname**: Always use `host.docker.internal` to refer to services running on your main Windows Windows machine.
