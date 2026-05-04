# Changes

## 2026-05-04

### Fix 1: Per-model async locks
- Replaced the single shared generation lock path with a `model_locks` registry keyed by model id, while keeping `npu_resource_lock` as a fallback safety lock.
- Registered a dedicated `asyncio.Lock()` for each model in `load_npu_model`, and cleared lock state alongside model state during startup and shutdown.
- Updated `/v1/chat/completions` and `/v1/responses` to acquire `get_model_lock(request.model)` so concurrent requests to different loaded models no longer serialize behind one global lock.
- Updated `/v1/system/status` to expose per-model lock state in `npu.model_locks`, and changed `npu.busy` to derive from lock state instead of `is_generating`.

Why:
- The old single lock forced all models to contend for one shared generation slot.
- Per-model locks let separate loaded models run independently while still preserving single-flight behavior per model.

How to verify:
- Start the server with at least two models loaded, for example `--models qwen1.5-1.8b,qwen2-1.5b`.
- Send two concurrent generation requests, one to each model.
- Call `/v1/system/status` while they are running.
- Confirm `npu.model_locks` shows the active model ids, and that only the lock(s) for the model(s) currently generating are `true`.
- Confirm requests to different models can overlap, while two requests to the same model still serialize.

### Fix 2: Sliding window prompt management
- Added `build_prompt_sliding_window(...)` to construct chat prompts within `max_prompt_len` by preserving the system block, reserving space for the newest user turn, and then filling with as much recent history as fits.
- Tokenized each message independently before assembly so the server can drop whole turns instead of blindly chopping tokens from the left side of the final prompt.
- Added logging when older turns are dropped to fit the prompt budget.
- Updated the malformed tool-call retry path in `/v1/chat/completions` to reuse the same sliding-window logic, so retries keep the same context-preservation behavior.
- Left `/v1/responses` unchanged apart from the per-model lock swap; it still uses simple truncation as requested.

Why:
- The old left-truncation could cut away important system instructions and break conversational continuity.
- A sliding window keeps the prompt logically coherent by prioritizing the preserved system context and the newest conversational turns.

How to verify:
- Run a long multi-turn chat in OpenWebUI against `/v1/chat/completions` until the prompt would previously have exceeded `max_prompt_len`.
- Compare the behavior to the old implementation: the newest turns and the conversation’s governing system/tool context should stay coherent instead of being cut off by raw left-truncation.
- Watch server logs for `Sliding window: dropped ... older turn(s)` entries once the chat history grows beyond the configured prompt budget.
