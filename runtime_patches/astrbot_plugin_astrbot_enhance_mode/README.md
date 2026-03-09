# `astrbot_plugin_astrbot_enhance_mode` Runtime Patch

This directory versions the live plugin patch exported from the NAS `astrbot` container.

Scope:
- Runtime plugin path in container: `/AstrBot/data/plugins/astrbot_plugin_astrbot_enhance_mode/main.py`
- Exported on: `2026-03-09`

Why this exists:
- The plugin is not part of the upstream `AstrBot` repository.
- The live NAS instance contains hotfixes for `active_reply.mode = model_choice`.
- Without exporting the patched file, those fixes would be lost after a container rebuild.

Included fixes:
- prompt-overflow retry for `max_prompt_tokens` style errors
- model-aware soft budget for `model_choice` judge prompts
- timeout-aware shrink-and-retry for slow judge calls
- readable `model_choice` logs for runtime diagnosis

Current budget behavior:
- normal/strong current model: keep full configured window unless retry is needed
- weak fallback chain detected: pre-shrink to `stack=14`, `history=24`
- weak current model detected: pre-shrink to `stack=10`, `history=16`

Apply to a runtime plugin file:

```bash
python scripts/apply_enhance_mode_runtime_patch.py --target /AstrBot/data/plugins/astrbot_plugin_astrbot_enhance_mode/main.py
```

Notes:
- The script creates a timestamped backup beside the target file before overwrite.
- This export is a runtime patch snapshot, not an upstream plugin source mirror.
