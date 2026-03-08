from __future__ import annotations

import re
import time
from typing import Any

from astrbot import logger
from astrbot.api import sp

SESSION_SERVICE_CONFIG_KEY = "session_service_config"
CHAT_PROVIDER_PREF_KEY = "provider_perf_chat_completion"
SUPER_NOEL_STICKY_STATE_KEY = "sub2api_super_noel_sticky_state"
SUPER_NOEL_PERSONA_ID = "?????"
SUPER_NOEL_AGENT_NAME = "super_noel"
SUPER_NOEL_HANDOFF_TOOL_NAME = "transfer_to_super_noel"

DEFAULT_STICKY_ENABLED = True
DEFAULT_STICKY_TTL_SECONDS = 600
DEFAULT_STICKY_MAX_TURNS = 3
DEFAULT_STICKY_RETURN_KEYWORDS = (
    "?????",
    "?????",
    "??????",
    "????",
    "????",
    "??????",
)


def _normalize_settings_source(settings_source: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(settings_source, dict):
        return settings_source
    return {}


def get_super_noel_sticky_settings(
    settings_source: dict[str, Any] | None,
) -> dict[str, Any]:
    root = _normalize_settings_source(settings_source)
    provider_settings = root.get("provider_settings")
    if not isinstance(provider_settings, dict):
        provider_settings = root
    orchestrator_settings = root.get("subagent_orchestrator")
    if not isinstance(orchestrator_settings, dict):
        orchestrator_settings = {}

    ttl_seconds = orchestrator_settings.get(
        "super_noel_sticky_ttl_seconds",
        provider_settings.get(
            "sub2api_super_noel_sticky_ttl_seconds",
            DEFAULT_STICKY_TTL_SECONDS,
        ),
    )
    max_turns = orchestrator_settings.get(
        "super_noel_sticky_max_turns",
        provider_settings.get(
            "sub2api_super_noel_sticky_max_turns",
            DEFAULT_STICKY_MAX_TURNS,
        ),
    )
    enabled = orchestrator_settings.get(
        "super_noel_sticky_enabled",
        provider_settings.get(
            "sub2api_super_noel_sticky_enabled",
            DEFAULT_STICKY_ENABLED,
        ),
    )
    return_keywords = orchestrator_settings.get(
        "super_noel_sticky_return_keywords",
        provider_settings.get(
            "sub2api_super_noel_sticky_return_keywords",
            list(DEFAULT_STICKY_RETURN_KEYWORDS),
        ),
    )

    try:
        ttl_seconds = max(int(ttl_seconds), 0)
    except Exception:
        ttl_seconds = DEFAULT_STICKY_TTL_SECONDS
    try:
        max_turns = max(int(max_turns), 0)
    except Exception:
        max_turns = DEFAULT_STICKY_MAX_TURNS
    if not isinstance(return_keywords, list):
        return_keywords = list(DEFAULT_STICKY_RETURN_KEYWORDS)

    return {
        "enabled": bool(enabled),
        "ttl_seconds": ttl_seconds,
        "max_turns": max_turns,
        "return_keywords": [
            str(item).strip() for item in return_keywords if str(item).strip()
        ],
    }


def resolve_super_noel_binding_from_config(
    cfg: dict[str, Any] | None,
    *,
    agent_name: str = SUPER_NOEL_AGENT_NAME,
) -> tuple[str, str | None]:
    orchestrator = {}
    if isinstance(cfg, dict):
        orchestrator = cfg.get("subagent_orchestrator", {}) or {}
    agents = orchestrator.get("agents", [])
    if isinstance(agents, list):
        for item in agents:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() != str(agent_name or "").strip():
                continue
            persona_id = str(item.get("persona_id") or SUPER_NOEL_PERSONA_ID).strip()
            provider_id = str(item.get("provider_id") or "").strip() or None
            return persona_id or SUPER_NOEL_PERSONA_ID, provider_id
    return SUPER_NOEL_PERSONA_ID, None


async def _get_session_service_config(umo: str) -> dict[str, Any]:
    cfg = await sp.get_async(
        scope="umo",
        scope_id=umo,
        key=SESSION_SERVICE_CONFIG_KEY,
        default={},
    )
    if isinstance(cfg, dict):
        return dict(cfg)
    return {}


async def _set_session_service_config(umo: str, value: dict[str, Any]) -> None:
    if value:
        await sp.put_async(
            scope="umo",
            scope_id=umo,
            key=SESSION_SERVICE_CONFIG_KEY,
            value=value,
        )
    else:
        await sp.remove_async(
            scope="umo",
            scope_id=umo,
            key=SESSION_SERVICE_CONFIG_KEY,
        )


async def _get_chat_provider_pref(umo: str) -> str | None:
    provider_id = await sp.get_async(
        scope="umo",
        scope_id=umo,
        key=CHAT_PROVIDER_PREF_KEY,
        default=None,
    )
    provider_id = str(provider_id or "").strip()
    return provider_id or None


async def _set_chat_provider_pref(umo: str, provider_id: str | None) -> None:
    provider_id = str(provider_id or "").strip()
    if provider_id:
        await sp.put_async(
            scope="umo",
            scope_id=umo,
            key=CHAT_PROVIDER_PREF_KEY,
            value=provider_id,
        )
    else:
        await sp.remove_async(
            scope="umo",
            scope_id=umo,
            key=CHAT_PROVIDER_PREF_KEY,
        )


async def get_super_noel_sticky_state(umo: str) -> dict[str, Any] | None:
    state = await sp.get_async(
        scope="umo",
        scope_id=umo,
        key=SUPER_NOEL_STICKY_STATE_KEY,
        default=None,
    )
    if isinstance(state, dict) and state:
        return dict(state)
    return None


def _should_release_by_text(text: str, return_keywords: list[str]) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in return_keywords)


def _looks_like_command_style_input(text: str | None) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    return bool(re.match(r"^(?:/|\uFF0F)[A-Za-z0-9][A-Za-z0-9_.:-]*(?:\s|$)", first_line))


async def clear_super_noel_sticky_session(
    umo: str,
    *,
    reason: str = "",
) -> None:
    state = await get_super_noel_sticky_state(umo)
    if not state:
        return

    sticky_persona_id = str(state.get("persona_id") or SUPER_NOEL_PERSONA_ID).strip()
    sticky_provider_id = str(state.get("provider_id") or "").strip()
    original_session_cfg = state.get("original_session_service_config")
    if not isinstance(original_session_cfg, dict):
        original_session_cfg = {}
    original_provider_id = str(state.get("original_chat_provider_id") or "").strip() or None

    current_session_cfg = await _get_session_service_config(umo)
    if current_session_cfg.get("persona_id") == sticky_persona_id:
        restored_cfg = dict(original_session_cfg)
        await _set_session_service_config(umo, restored_cfg)

    current_provider_id = await _get_chat_provider_pref(umo)
    if sticky_provider_id and current_provider_id == sticky_provider_id:
        await _set_chat_provider_pref(umo, original_provider_id)

    await sp.remove_async(
        scope="umo",
        scope_id=umo,
        key=SUPER_NOEL_STICKY_STATE_KEY,
    )
    logger.info(
        "super_noel sticky session cleared: umo=%s reason=%s",
        umo,
        reason or "manual",
    )


async def activate_super_noel_sticky_session(
    umo: str,
    *,
    persona_id: str,
    provider_id: str | None,
    settings_source: dict[str, Any] | None = None,
) -> bool:
    settings = get_super_noel_sticky_settings(settings_source)
    if not settings["enabled"]:
        return False

    persona_id = str(persona_id or SUPER_NOEL_PERSONA_ID).strip() or SUPER_NOEL_PERSONA_ID
    provider_id = str(provider_id or "").strip() or None
    now = time.time()

    existing_state = await get_super_noel_sticky_state(umo)
    current_session_cfg = await _get_session_service_config(umo)
    current_provider_id = await _get_chat_provider_pref(umo)

    if existing_state and existing_state.get("active"):
        original_session_cfg = existing_state.get("original_session_service_config")
        if not isinstance(original_session_cfg, dict):
            original_session_cfg = dict(current_session_cfg)
        original_chat_provider_id = (
            str(existing_state.get("original_chat_provider_id") or "").strip() or None
        )
    else:
        original_session_cfg = dict(current_session_cfg)
        original_chat_provider_id = current_provider_id

    state = {
        "active": True,
        "persona_id": persona_id,
        "provider_id": provider_id,
        "entered_at": now,
        "last_active_at": now,
        "expires_at": now + settings["ttl_seconds"],
        "remaining_turns": settings["max_turns"],
        "original_session_service_config": original_session_cfg,
        "original_chat_provider_id": original_chat_provider_id,
    }

    await sp.put_async(
        scope="umo",
        scope_id=umo,
        key=SUPER_NOEL_STICKY_STATE_KEY,
        value=state,
    )

    next_session_cfg = dict(current_session_cfg)
    next_session_cfg["persona_id"] = persona_id
    await _set_session_service_config(umo, next_session_cfg)
    if provider_id:
        await _set_chat_provider_pref(umo, provider_id)

    logger.info(
        "super_noel sticky session activated: umo=%s persona=%s provider=%s ttl=%s turns=%s",
        umo,
        persona_id,
        provider_id or "",
        settings["ttl_seconds"],
        settings["max_turns"],
    )
    return True


async def prepare_super_noel_sticky_session(
    umo: str,
    *,
    user_text: str | None,
    settings_source: dict[str, Any] | None = None,
) -> bool:
    settings = get_super_noel_sticky_settings(settings_source)
    state = await get_super_noel_sticky_state(umo)
    if not state:
        return False
    if not settings["enabled"]:
        await clear_super_noel_sticky_session(umo, reason="disabled")
        return False

    text = str(user_text or "")
    if _looks_like_command_style_input(text):
        await clear_super_noel_sticky_session(umo, reason="command_input")
        return False
    if _should_release_by_text(text, settings["return_keywords"]):
        await clear_super_noel_sticky_session(umo, reason="user_requested_return")
        return False

    now = time.time()
    expires_at = float(state.get("expires_at") or 0)
    if expires_at and now >= expires_at:
        await clear_super_noel_sticky_session(umo, reason="expired")
        return False

    remaining_turns = int(state.get("remaining_turns") or 0)
    if remaining_turns <= 0:
        await clear_super_noel_sticky_session(umo, reason="turn_budget_exhausted")
        return False

    persona_id = str(state.get("persona_id") or SUPER_NOEL_PERSONA_ID).strip() or SUPER_NOEL_PERSONA_ID
    provider_id = str(state.get("provider_id") or "").strip() or None

    current_session_cfg = await _get_session_service_config(umo)
    if current_session_cfg.get("persona_id") != persona_id:
        current_session_cfg["persona_id"] = persona_id
        await _set_session_service_config(umo, current_session_cfg)

    if provider_id:
        current_provider_id = await _get_chat_provider_pref(umo)
        if current_provider_id != provider_id:
            await _set_chat_provider_pref(umo, provider_id)

    state["last_active_at"] = now
    state["remaining_turns"] = max(remaining_turns - 1, 0)
    await sp.put_async(
        scope="umo",
        scope_id=umo,
        key=SUPER_NOEL_STICKY_STATE_KEY,
        value=state,
    )
    logger.info(
        "super_noel sticky session consumed: umo=%s remaining_turns=%s expires_at=%s",
        umo,
        state["remaining_turns"],
        state.get("expires_at"),
    )
    return True
