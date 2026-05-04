# TESTING.md

## Section A — Health & Startup Tests

1. Start the server from the workspace root with one model:

```powershell
python .\intel-npu-llm\npu_server.py --models qwen1.5-1.8b --port 8000
```

2. While the model is still loading, poll the health endpoint and confirm it returns HTTP 200 with `{"status":"loading","models_loaded":0}`:

```powershell
curl http://localhost:8000/health
```

3. Keep polling until the server reports `{"status":"ok"}`:

```powershell
curl http://localhost:8000/health
```

4. Verify the detailed system status endpoint is reachable during and after load:

```powershell
curl http://localhost:8000/v1/system/status
```

5. Confirm the server accepts requests while models are still loading by calling both endpoints before startup completes:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/v1/system/status
```

## Section B — OpenWebUI Integration Tests

1. Open OpenWebUI in a browser.
2. Go to `Settings` -> `Connections` -> `OpenAI API`.
3. Add a custom connection with these values:
   Base URL: `http://<host>:8000/v1`
   API Key: any dummy string such as `test-key`
4. Save the connection and refresh the model list if OpenWebUI does not refresh automatically.
5. Confirm the server model appears in the model selector.
6. Select the model and send `What is 2+2?`.
7. Confirm the response is streamed into the chat UI instead of appearing only at the end.
8. Run a streaming stress test with this prompt: `Write a 200-word story about a robot exploring a forest.`
9. Confirm tokens appear progressively over time and not as one final block.

## Section C — Tool Calling Tests

1. Test non-streaming tool calling with a simple weather tool definition:

```powershell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen1.5-1.8b",
    "messages": [
      {"role": "user", "content": "What is the weather in Seattle? Use the provided function."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "stream": false
  }'
```

2. Verify the JSON response contains `"finish_reason":"tool_calls"` and a non-empty `tool_calls` array inside `choices[0].message`.

3. Test streaming tool calling with the same tool definition:

```powershell
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen1.5-1.8b",
    "messages": [
      {"role": "user", "content": "What is the weather in Seattle? Use the provided function."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

4. Verify the SSE output contains tool call deltas with `tool_calls` entries and does not emit normal `content` deltas before the final `data: [DONE]`.
5. If OpenWebUI exposes tool or function mode, enable it for the connection or chat, provide the same `get_weather` schema, and confirm the assistant emits a tool call instead of plain text.

## Section D — Frontend (index.html) Tests

1. Open the built-in UI in a browser:

```text
http://localhost:8000
```

2. Select a model from the dropdown.
3. Send a message such as `Hello, describe yourself in one sentence.`
4. Confirm the response renders in the chat area.
5. Verify streaming by sending `Write a 200-word story about a robot exploring a forest.` and confirming tokens appear progressively.
6. Open the browser developer tools network tab and confirm the model selector is populated from `/v1/models`.

## Section E — Regression Tests

1. Verify `/v1/models` lists the currently loaded model IDs:

```powershell
curl http://localhost:8000/v1/models
```

2. Verify `/v1/system/status` returns disk and memory stats:

```powershell
curl http://localhost:8000/v1/system/status
```

3. Stop the server and start it with two models:

```powershell
python .\intel-npu-llm\npu_server.py --models qwen1.5-1.8b,qwen2-1.5b --port 8000
```

4. Confirm both models appear in `/v1/models`:

```powershell
curl http://localhost:8000/v1/models
```

5. Confirm startup remains responsive during loading by polling these endpoints before all models finish loading:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/v1/system/status
```
