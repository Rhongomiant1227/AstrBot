import asyncio
import base64
import datetime
import json
import mimetypes
import random
import re
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp import types as mcp_types

from astrbot.api import llm_tool, logger, sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.io import download_image_by_url

from .ban_control import BanStore, parse_duration_seconds
from .memory_rag_store import MemoryRAGStore
from .plugin_config import PluginConfig, parse_plugin_config
from .runtime_state import RuntimeState
from .tag_utils import (
    bounded_chat_history_text,
    build_interaction_instructions,
    chain_has_refuse_tag,
    clean_response_text_for_history,
    has_refuse_tag,
    transform_result_chain,
)
from .webui import RAGWebUIServer

IMAGE_MARKER_PATTERN = re.compile(r"\[Image(?:: [^\]]*)?\]")
MSG_ID_PATTERN = re.compile(r"#msg([^:]+):")


class Main(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.runtime = RuntimeState()
        self._display_timezone = self._resolve_config_timezone()
        plugin_data_dir = (
            Path(get_astrbot_data_path())
            / "plugin_data"
            / "astrbot_plugin_astrbot_enhance_mode"
        )
        self.ban_store = BanStore(plugin_data_dir / "ban_list.db")
        self.memory_rag_store: MemoryRAGStore | None = None
        self.rag_webui_server: RAGWebUIServer | None = None
        try:
            self.memory_rag_store = MemoryRAGStore(
                plugin_data_dir / "memory_rag.db",
                display_timezone=self._display_timezone,
            )
        except Exception as e:
            logger.error(f"enhance-mode | 初始化记忆 RAG 存储失败: {e}", exc_info=True)
        logger.info(
            "enhance-mode | plugin initialized | data_dir=%s memory_rag_store_ready=%s timezone=%s",
            plugin_data_dir,
            self.memory_rag_store is not None,
            self._display_timezone,
        )

    def _cfg(self) -> PluginConfig:
        return parse_plugin_config(self.config)

    def _touch_origin(self, origin: str, cfg: PluginConfig) -> None:
        self.runtime.touch_origin(origin, cfg.global_settings.lru_cache.max_origins)

    def _resolve_config_timezone(self) -> str:
        base_cfg = self.context.get_config()
        if not isinstance(base_cfg, dict):
            return "Asia/Shanghai"
        timezone_name = str(base_cfg.get("timezone") or "").strip()
        return timezone_name or "Asia/Shanghai"

    def _resolve_tzinfo(self) -> datetime.tzinfo:
        timezone_name = self._resolve_config_timezone()
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            try:
                return ZoneInfo("Asia/Shanghai")
            except Exception:
                return datetime.timezone.utc

    def _format_timestamp_iso(self, timestamp: float | int) -> str:
        if self.memory_rag_store is not None:
            return self.memory_rag_store.format_timestamp_iso(timestamp)
        return datetime.datetime.fromtimestamp(
            float(timestamp), tz=self._resolve_tzinfo()
        ).isoformat()

    @staticmethod
    def _provider_label(provider: object | None) -> str:
        if provider is None:
            return "none"
        provider_id = getattr(provider, "provider_id", None) or getattr(
            provider, "id", None
        )
        if provider_id:
            return str(provider_id)
        model = getattr(provider, "model", None)
        cls_name = type(provider).__name__
        return f"{cls_name}({model})" if model else cls_name

    @staticmethod
    def _normalize_message_id(raw: str | int | None) -> str:
        text = str(raw or "").strip()
        if text.startswith("#msg"):
            text = text[4:]
        if text.endswith(":"):
            text = text[:-1]
        return text.strip()

    @staticmethod
    def _extract_message_id_from_history_line(line: str) -> str:
        matched = MSG_ID_PATTERN.search(str(line or ""))
        if not matched:
            return ""
        return str(matched.group(1) or "").strip()

    @staticmethod
    def _replace_image_marker_at_index(
        line: str, image_index: int, caption: str
    ) -> tuple[str, bool]:
        if image_index < 0:
            return line, False
        matches = list(IMAGE_MARKER_PATTERN.finditer(line))
        if image_index >= len(matches):
            return line, False

        target = matches[image_index]
        safe_caption = str(caption or "").strip().replace("]", ")")
        replacement = f"[Image: {safe_caption}]"
        new_line = line[: target.start()] + replacement + line[target.end() :]
        return new_line, new_line != line

    def _apply_image_caption_to_history(
        self,
        origin: str,
        message_id: str,
        image_index: int,
        caption: str,
    ) -> bool:
        chats = self.runtime.session_chats.get(origin)
        if not chats:
            return False

        target_marker = f"#msg{message_id}:"
        for idx, line in enumerate(chats):
            if target_marker not in line:
                continue
            replaced_line, changed = self._replace_image_marker_at_index(
                line, image_index, caption
            )
            if not changed:
                return False
            chats[idx] = replaced_line
            return True
        return False

    async def _resolve_image_ref_to_local_path(self, image_ref: str) -> str:
        clean_ref = str(image_ref or "").strip()
        if not clean_ref:
            return ""

        if clean_ref.startswith("file://"):
            clean_ref = clean_ref[7:]

        candidate = Path(clean_ref)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        if clean_ref.startswith("http://") or clean_ref.startswith("https://"):
            downloaded = await download_image_by_url(clean_ref)
            return str(downloaded or "")

        return ""

    @staticmethod
    def _encode_image_file(image_path: str) -> tuple[str, str]:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        raw_bytes = path.read_bytes()
        if not raw_bytes:
            raise ValueError(f"Image file is empty: {image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return encoded, mime_type

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(0, int(seconds))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)

    @staticmethod
    def _ban_scope_id(event: AstrMessageEvent) -> str:
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return ""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return ""
        platform_id = (
            str(event.get_platform_id() or "").strip()
            or str(event.get_platform_name() or "").strip()
        )
        return f"{platform_id}:{group_id}" if platform_id else group_id

    def _get_admin_sid_set(self) -> set[str]:
        base_cfg = self.context.get_config()
        if not isinstance(base_cfg, dict):
            return set()
        admins = base_cfg.get("admins_id", [])
        if not isinstance(admins, list):
            return set()
        return {str(sid).strip() for sid in admins if str(sid).strip()}

    @staticmethod
    def _parse_role_ids(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []

        parsed_ids: list[str] = []
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parsed_ids = [str(item).strip() for item in payload]
            except json.JSONDecodeError:
                parsed_ids = []

        if not parsed_ids:
            normalized = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
            parsed_ids = [token.strip() for token in normalized.split(",")]

        deduped: list[str] = []
        seen: set[str] = set()
        for role_id in parsed_ids:
            if not role_id or role_id in seen:
                continue
            seen.add(role_id)
            deduped.append(role_id)
        return deduped

    def _parse_optional_timestamp(self, raw: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None

        try:
            numeric = float(text)
            # Heuristic: treat millisecond timestamps as ms.
            if numeric > 1e12:
                numeric /= 1000.0
            return numeric
        except (TypeError, ValueError):
            pass

        tzinfo = self._resolve_tzinfo()
        normalized = text.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.datetime.strptime(normalized, fmt).replace(tzinfo=tzinfo)
                return dt.timestamp()
            except ValueError:
                continue

        try:
            dt = datetime.datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tzinfo)
            return dt.timestamp()
        except ValueError:
            return None

    @staticmethod
    def _parse_extra_metadata(raw: str) -> dict:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return {"raw_metadata": payload}
        except json.JSONDecodeError:
            return {"raw_metadata": text}

    @staticmethod
    def _normalize_sort_order(raw: str) -> str:
        return "asc" if str(raw or "").strip().lower() == "asc" else "desc"

    @staticmethod
    def _normalize_sort_by(raw: str) -> str:
        return "time" if str(raw or "").strip().lower() == "time" else "relevance"

    def _resolve_memory_scope(
        self,
        event: AstrMessageEvent,
        group_scope: str,
        group_id: str,
        platform_id: str,
    ) -> tuple[str, str, str]:
        explicit_scope = str(group_scope or "").strip()
        explicit_group_id = str(group_id or "").strip()
        explicit_platform_id = str(platform_id or "").strip()

        event_group_id = (
            str(event.get_group_id() or "").strip()
            if event.get_message_type() == MessageType.GROUP_MESSAGE
            else ""
        )
        event_platform_id = str(
            event.get_platform_id() or event.get_platform_name() or ""
        ).strip()

        final_group_id = explicit_group_id or event_group_id
        final_platform_id = explicit_platform_id or event_platform_id

        if explicit_scope:
            final_scope = explicit_scope
        elif final_group_id:
            final_scope = (
                f"{final_platform_id}:{final_group_id}"
                if final_platform_id
                else final_group_id
            )
        else:
            final_scope = ""

        return final_scope, final_group_id, final_platform_id

    def _resolve_embedding_provider(
        self, cfg: PluginConfig
    ) -> EmbeddingProvider | None:
        provider_id = str(cfg.memory_rag.embedding_provider_id or "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider and isinstance(provider, EmbeddingProvider):
                logger.debug(
                    "enhance-mode | using configured embedding provider | provider=%s",
                    self._provider_label(provider),
                )
                return provider
            logger.warning(
                f"enhance-mode | 配置的 embedding_provider_id 无效或类型不匹配: {provider_id}"
            )

        all_embedding_providers = self.context.get_all_embedding_providers()
        if all_embedding_providers:
            logger.debug(
                "enhance-mode | using fallback embedding provider | provider=%s",
                self._provider_label(all_embedding_providers[0]),
            )
            return all_embedding_providers[0]
        return None

    def _check_memory_rag_ready(self) -> tuple[bool, str]:
        cfg = self._cfg()
        if not cfg.memory_rag.enable:
            return False, "Memory RAG is disabled in enhance mode config."
        if not self.memory_rag_store:
            return False, "Memory RAG store is not initialized."
        return True, ""

    def _memory_rag_webui_url(self, cfg: PluginConfig) -> str:
        host = str(cfg.memory_rag_webui.host or "127.0.0.1").strip() or "127.0.0.1"
        port = int(cfg.memory_rag_webui.port)
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    async def _start_memory_rag_webui(self) -> None:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        if not webui_cfg.enable:
            return
        if not self.memory_rag_store:
            logger.warning(
                "enhance-mode | memory_rag_webui enabled but memory_rag_store is unavailable"
            )
            return
        if self.rag_webui_server is not None:
            return

        try:
            self.rag_webui_server = RAGWebUIServer(
                store=self.memory_rag_store,
                config={
                    "host": webui_cfg.host,
                    "port": webui_cfg.port,
                    "access_password": webui_cfg.access_password,
                    "session_timeout": webui_cfg.session_timeout,
                },
                plugin_version="0.1.0",
            )
            await self.rag_webui_server.start()
            logger.info(
                "enhance-mode | Memory RAG WebUI started at %s",
                self._memory_rag_webui_url(cfg),
            )
            if self.rag_webui_server.password_generated:
                logger.info(
                    "enhance-mode | Memory RAG WebUI generated password: %s",
                    self.rag_webui_server.access_password,
                )
        except Exception as e:
            logger.error(
                f"enhance-mode | 启动 Memory RAG WebUI 失败: {e}", exc_info=True
            )
            self.rag_webui_server = None

    async def _stop_memory_rag_webui(self) -> None:
        if self.rag_webui_server is None:
            return
        try:
            await self.rag_webui_server.stop()
        except Exception as e:
            logger.warning(
                f"enhance-mode | 停止 Memory RAG WebUI 失败: {e}", exc_info=True
            )
        finally:
            self.rag_webui_server = None

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        cfg = self._cfg()
        self._display_timezone = self._resolve_config_timezone()
        if self.memory_rag_store is not None:
            self.memory_rag_store.set_display_timezone(self._display_timezone)
        logger.info(
            "enhance-mode | loaded | react_mode=%s group_history=%s active_reply=%s memory_rag=%s webui=%s lru_max_origins=%s timezone=%s",
            cfg.group_features.react_mode_enable,
            cfg.group_history_enabled,
            cfg.active_reply_enabled,
            cfg.memory_rag.enable,
            cfg.memory_rag_webui.enable,
            cfg.global_settings.lru_cache.max_origins,
            self._display_timezone,
        )
        await self._start_memory_rag_webui()

    def _allow_active_reply(self, event: AstrMessageEvent, cfg: PluginConfig) -> bool:
        ar = cfg.active_reply
        if not cfg.active_reply_enabled:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar.whitelist and (
            event.unified_msg_origin not in ar.whitelist
            and (event.get_group_id() and event.get_group_id() not in ar.whitelist)
        ):
            return False
        return True

    async def _resolve_persona_mask(self, event: AstrMessageEvent) -> tuple[str, str]:
        persona_id = ""
        try:
            session_service_config = await sp.get_async(
                scope="umo",
                scope_id=event.unified_msg_origin,
                key="session_service_config",
                default={},
            )
            if isinstance(session_service_config, dict):
                persona_id = str(session_service_config.get("persona_id") or "").strip()
        except Exception as e:
            logger.debug(f"enhance-mode | 获取 session persona 失败: {e}")

        if not persona_id:
            try:
                curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if curr_cid:
                    conv = await self.context.conversation_manager.get_conversation(
                        event.unified_msg_origin,
                        curr_cid,
                    )
                    if conv and conv.persona_id:
                        persona_id = str(conv.persona_id).strip()
            except Exception as e:
                logger.debug(f"enhance-mode | 获取 conversation persona 失败: {e}")

        if not persona_id:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            persona_id = str(
                cfg.get("provider_settings", {}).get("default_personality") or ""
            ).strip()

        if persona_id == "[%None]":
            return "none", "No persona mask."

        persona = None
        if persona_id:
            try:
                persona = next(
                    (
                        p
                        for p in self.context.persona_manager.personas_v3
                        if p.get("name") == persona_id
                    ),
                    None,
                )
            except Exception:
                persona = None

        if not persona:
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(
                    event.unified_msg_origin
                )
            except Exception:
                persona = {"name": "default", "prompt": ""}

        persona_name = str(persona.get("name") or "default")
        persona_prompt = str(persona.get("prompt") or "").strip()
        if not persona_prompt:
            persona_prompt = "You are a helpful and friendly assistant."
        return persona_name, persona_prompt

    @staticmethod
    def _extract_prompt_overflow_info(error: Exception | str) -> tuple[int, int] | None:
        text = str(error or "")
        patterns = (
            (r"number of input tokens \((\d+)\) have exceeded max_prompt_tokens \((\d+)\)", "input_limit"),
            (r"maximum context length is (\d+) tokens.*?resulted in (\d+) tokens", "limit_input"),
            (r"requested (\d+) tokens.*?maximum context length is (\d+) tokens", "input_limit_reverse"),
        )
        for pattern, mode in patterns:
            matched = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not matched:
                continue
            first = int(matched.group(1))
            second = int(matched.group(2))
            if mode == "limit_input":
                return second, first
            if mode == "input_limit_reverse":
                return first, second
            return first, second
        return None

    @staticmethod
    def _shrink_model_choice_windows(
        messages: list[str],
        history_lines: list[str],
        *,
        input_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
    ) -> tuple[list[str], list[str]]:
        current_messages = list(messages)
        current_history = list(history_lines)
        if not current_messages and not current_history:
            return current_messages, current_history

        if (
            input_tokens is not None
            and max_prompt_tokens is not None
            and input_tokens > 0
            and max_prompt_tokens > 0
        ):
            keep_ratio = (max_prompt_tokens * 0.82) / input_tokens
            keep_ratio = min(0.92, max(0.25, keep_ratio))
            stack_ratio = min(0.95, max(0.50, keep_ratio + 0.08))
            history_ratio = min(0.85, max(0.15, keep_ratio - 0.12))
        else:
            stack_ratio = 0.75
            history_ratio = 0.50 if current_history else 0.0

        history_keep = len(current_history)
        if current_history:
            history_keep = max(
                0,
                min(history_keep, int(len(current_history) * history_ratio)),
            )
            if history_keep == len(current_history):
                history_keep = max(0, len(current_history) - max(4, len(current_history) // 4))

        stack_keep = len(current_messages)
        if current_messages:
            min_stack_keep = 4 if len(current_messages) > 4 else 1
            stack_keep = max(
                min_stack_keep,
                min(stack_keep, int(len(current_messages) * stack_ratio)),
            )
            if stack_keep == len(current_messages) and len(current_messages) > min_stack_keep:
                stack_keep = max(min_stack_keep, len(current_messages) - max(2, len(current_messages) // 4))

        if history_keep == len(current_history) and stack_keep == len(current_messages):
            if history_keep > 0:
                history_keep -= 1
            elif stack_keep > 1:
                stack_keep -= 1

        new_messages = current_messages[-stack_keep:] if stack_keep > 0 else []
        new_history = current_history[-history_keep:] if history_keep > 0 else []
        return new_messages, new_history


    @staticmethod
    def _build_model_choice_prompt(
        prompt_tmpl: str,
        persona_name: str,
        persona_mask: str,
        messages: list[str],
        history_context_lines: list[str],
    ) -> str:
        history_context = (
            "\n".join(history_context_lines)
            if history_context_lines
            else "(disabled or no additional history)"
        )
        try:
            return prompt_tmpl.format(
                stack_size=len(messages),
                messages="\n".join(messages),
                history_count=len(history_context_lines),
                history_context=history_context,
                persona_name=persona_name,
                persona_mask=persona_mask,
            )
        except Exception:
            return (
                f"{prompt_tmpl}\n\n"
                f"Persona ({persona_name}):\n{persona_mask}\n\n"
                f"Recent messages:\n{chr(10).join(messages)}\n\n"
                f"Extra history ({len(history_context_lines)}):\n{history_context}\n\n"
                "Output only REPLY or SKIP."
            )

    def _get_model_choice_soft_budget(
        self,
        event: AstrMessageEvent,
        provider: Provider,
        messages: list[str],
        history_context_lines: list[str],
    ) -> tuple[int, int, str, str, str]:
        provider_id = str(getattr(provider, "provider_config", {}).get("id", "") or "")
        try:
            model_name = str(provider.get_model() or "")
        except Exception:
            model_name = str(getattr(provider, "model_name", "") or "")

        current_labels = [label.lower() for label in (provider_id, model_name) if label]
        chain_labels = list(current_labels)

        provider_settings = {}
        try:
            config = self.context.get_config(event.unified_msg_origin)
            if hasattr(config, "get"):
                provider_settings = config.get("provider_settings", {}) or {}
        except Exception:
            provider_settings = {}

        fallback_ids = provider_settings.get("fallback_chat_models", [])
        if isinstance(fallback_ids, list):
            for fallback_id in fallback_ids:
                if not isinstance(fallback_id, str) or not fallback_id:
                    continue
                chain_labels.append(fallback_id.lower())
                fallback_provider = self.context.get_provider_by_id(fallback_id)
                if isinstance(fallback_provider, Provider):
                    try:
                        fallback_model = str(fallback_provider.get_model() or "")
                    except Exception:
                        fallback_model = str(
                            getattr(fallback_provider, "model_name", "") or ""
                        )
                    if fallback_model:
                        chain_labels.append(fallback_model.lower())

        stack_limit = len(messages)
        history_limit = len(history_context_lines)
        reason = "full"
        current_chain = " ".join(current_labels)
        full_chain = " ".join(chain_labels)

        weak_tokens = (
            "glm-4.1v-9b-thinking",
            "glm-4.1v",
            "9b-thinking",
        )
        if any(token in current_chain for token in weak_tokens):
            reason = "weak_current_model"
            stack_limit = min(stack_limit, 10)
            history_limit = min(history_limit, 16)
        elif any(token in full_chain for token in weak_tokens):
            reason = "weak_fallback_chain"
            stack_limit = min(stack_limit, 14)
            history_limit = min(history_limit, 24)

        return stack_limit, history_limit, reason, provider_id, model_name

    @staticmethod
    def _shrink_model_choice_windows_for_timeout(
        current_messages: list[str],
        current_history: list[str],
    ) -> tuple[list[str], list[str]]:
        history_keep = len(current_history)
        if current_history:
            min_history_keep = 6 if len(current_history) > 6 else 1
            history_keep = max(
                min_history_keep,
                min(history_keep, int(len(current_history) * 0.60)),
            )
            if history_keep == len(current_history) and len(current_history) > min_history_keep:
                history_keep = max(
                    min_history_keep,
                    len(current_history) - max(2, len(current_history) // 3),
                )

        stack_keep = len(current_messages)
        if current_messages:
            min_stack_keep = 4 if len(current_messages) > 4 else 1
            stack_keep = max(
                min_stack_keep,
                min(stack_keep, int(len(current_messages) * 0.75)),
            )
            if stack_keep == len(current_messages) and len(current_messages) > min_stack_keep:
                stack_keep = max(
                    min_stack_keep,
                    len(current_messages) - max(1, len(current_messages) // 4),
                )

        if history_keep == len(current_history) and stack_keep == len(current_messages):
            if history_keep > 1:
                history_keep -= 1
            elif stack_keep > 1:
                stack_keep -= 1

        new_messages = current_messages[-stack_keep:] if stack_keep > 0 else []
        new_history = current_history[-history_keep:] if history_keep > 0 else []
        return new_messages, new_history

    async def _judge_model_choice(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        origin: str,
        messages: list[str],
        trigger_reason: str,
    ) -> bool:
        ar = cfg.active_reply
        history = self.runtime.model_choice_histories[origin]
        history_context_lines = []
        if ar.model_history_messages > 0:
            history_context_lines = history[-ar.model_history_messages :]

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider or not isinstance(provider, Provider):
            logger.error("enhance-mode | model_choice | provider_unavailable")
            return False

        persona_name, persona_mask = await self._resolve_persona_mask(event)
        prompt_tmpl = ar.model_choice_prompt
        message_window = list(messages)
        history_window = list(history_context_lines)
        (
            budget_stack,
            budget_history,
            budget_reason,
            provider_id,
            model_name,
        ) = self._get_model_choice_soft_budget(
            event,
            provider,
            message_window,
            history_window,
        )
        if budget_stack > 0 and len(message_window) > budget_stack:
            message_window = message_window[-budget_stack:]
        if budget_history > 0 and len(history_window) > budget_history:
            history_window = history_window[-budget_history:]

        logger.info(
            "enhance-mode | model_choice | start_judge | "
            f"origin={origin} trigger={trigger_reason} provider={provider_id or 'unknown'} "
            f"model={model_name or 'unknown'} budget={budget_reason} "
            f"stack={len(message_window)}/{len(messages)} history={len(history_window)}/{len(history_context_lines)}"
        )

        judge_resp = None
        for attempt in range(3):
            judge_prompt = self._build_model_choice_prompt(
                prompt_tmpl,
                persona_name,
                persona_mask,
                message_window,
                history_window,
            )
            try:
                judge_resp = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=judge_prompt,
                        session_id=uuid.uuid4().hex,
                        persist=False,
                    ),
                    timeout=cfg.global_settings.timeouts.model_choice_sec,
                )
                break
            except asyncio.TimeoutError:
                next_messages, next_history = self._shrink_model_choice_windows_for_timeout(
                    message_window,
                    history_window,
                )
                if next_messages == message_window and next_history == history_window:
                    logger.error(
                        "enhance-mode | model_choice | timeout_no_retry_budget | "
                        f"origin={origin} attempt={attempt + 1}/3 stack={len(message_window)} history={len(history_window)}"
                    )
                    return False
                logger.warning(
                    "enhance-mode | model_choice | timeout_retry | "
                    f"origin={origin} attempt={attempt + 1}/3 stack={len(message_window)}->{len(next_messages)} "
                    f"history={len(history_window)}->{len(next_history)}"
                )
                message_window = next_messages
                history_window = next_history
                continue
            except Exception as e:
                overflow = self._extract_prompt_overflow_info(e)
                if not overflow:
                    logger.error(
                        "enhance-mode | model_choice | request_failed | "
                        f"origin={origin} attempt={attempt + 1}/3 error={e}"
                    )
                    return False
                input_tokens, max_prompt_tokens = overflow
                next_messages, next_history = self._shrink_model_choice_windows(
                    message_window,
                    history_window,
                    input_tokens=input_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
                if next_messages == message_window and next_history == history_window:
                    logger.error(
                        "enhance-mode | model_choice | overflow_no_retry_budget | "
                        f"origin={origin} input_tokens={input_tokens} limit={max_prompt_tokens}"
                    )
                    return False
                logger.warning(
                    "enhance-mode | model_choice | overflow_retry | "
                    f"origin={origin} attempt={attempt + 1}/3 input_tokens={input_tokens} "
                    f"limit={max_prompt_tokens} stack={len(message_window)}->{len(next_messages)} "
                    f"history={len(history_window)}->{len(next_history)}"
                )
                message_window = next_messages
                history_window = next_history

        if judge_resp is None:
            logger.info(
                "enhance-mode | model_choice | skip_after_retries | "
                f"origin={origin} trigger={trigger_reason}"
            )
            return False

        decision_raw = (judge_resp.completion_text or "").strip().upper()
        decision = decision_raw.split()[0] if decision_raw else ""
        if decision.startswith("REPLY"):
            logger.info(
                "enhance-mode | model_choice | decision_reply | "
                f"origin={origin} trigger={trigger_reason} persona={persona_name}"
            )
            return True
        if decision and not decision.startswith("SKIP"):
            logger.info(
                "enhance-mode | model_choice | decision_unknown_skip | "
                f"origin={origin} trigger={trigger_reason} output={decision_raw}"
            )
            return False
        logger.info(
            "enhance-mode | model_choice | decision_skip | "
            f"origin={origin} trigger={trigger_reason} persona={persona_name}"
        )
        return False


    async def _need_active_reply_model_choice(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        origin = event.unified_msg_origin
        self._touch_origin(origin, cfg)

        ar = cfg.active_reply
        text = (event.message_str or "").strip() or "[Empty]"
        nickname = event.message_obj.sender.nickname
        sender_id = event.get_sender_id()
        stack = self.runtime.active_reply_stacks[origin]
        history = self.runtime.model_choice_histories[origin]

        stack.append(f"[{nickname}/{sender_id}]: {text}")
        history_line = (
            f"[{nickname}/{sender_id}/"
            f"{datetime.datetime.now().strftime('%H:%M:%S')}]: {text}"
        )
        history.append(history_line)

        history_limit = max(
            60,
            ar.model_stack_size * 6,
            ar.model_history_messages * 6,
        )
        if len(history) > history_limit:
            del history[:-history_limit]

        logger.info(
            "enhance-mode | model_choice | 栈填充 | "
            f"origin={origin} progress={len(stack)}/{ar.model_stack_size} "
            f"sender={sender_id}"
        )

        if len(stack) < ar.model_stack_size:
            return False

        messages = stack[-ar.model_stack_size :]
        stack.clear()
        return await self._judge_model_choice(
            event,
            cfg,
            origin,
            messages,
            trigger_reason="stack_full",
        )

    async def _get_image_caption(
        self,
        image_url: str,
        provider_id: str,
        prompt: str,
        timeout_sec: float,
    ) -> str:
        if not provider_id:
            provider = self.context.get_using_provider()
        else:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise Exception(f"没有找到 ID 为 {provider_id} 的提供商")

        if not isinstance(provider, Provider):
            raise Exception(f"提供商类型错误({type(provider)})，无法获取图片描述")

        start_ts = time.perf_counter()
        logger.debug(
            "enhance-mode | image_caption start | provider=%s timeout=%.1fs",
            self._provider_label(provider),
            timeout_sec,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                image_urls=[image_url],
                persist=False,
            ),
            timeout=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        logger.debug(
            "enhance-mode | image_caption done | provider=%s elapsed_ms=%.1f caption_len=%s",
            self._provider_label(provider),
            elapsed_ms,
            len(response.completion_text or ""),
        )
        return response.completion_text

    async def _need_active_reply(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        if not self._allow_active_reply(event, cfg):
            return False

        ar = cfg.active_reply
        if ar.mode == "model_choice":
            return await self._need_active_reply_model_choice(event, cfg)
        sample = random.random()
        decision = sample < ar.possibility
        logger.debug(
            "enhance-mode | active_reply probability | origin=%s sample=%.4f threshold=%.4f decision=%s",
            event.unified_msg_origin,
            sample,
            ar.possibility,
            decision,
        )
        return decision

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        cfg = self._cfg()
        if not cfg.group_features.role_display:
            return

        base_cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not base_cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                if "Nickname: " in part.text and role_line not in part.text:
                    nickname_idx = part.text.index("Nickname: ")
                    rest = part.text[nickname_idx:]
                    newline_idx = rest.find("\n")
                    if newline_idx != -1:
                        insert_pos = nickname_idx + newline_idx
                    else:
                        close_idx = part.text.find("</system_reminder>")
                        insert_pos = close_idx if close_idx != -1 else len(part.text)
                    part.text = (
                        part.text[:insert_pos] + role_line + part.text[insert_pos:]
                    )
                return

        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def guard_banned_user(self, event: AstrMessageEvent) -> None:
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        if released_count > 0:
            logger.info(
                "enhance-mode | 自动解封过期用户数量 | "
                f"count={released_count} scope={scope_id}"
            )

        sender_id = str(event.get_sender_id() or "").strip()
        if not sender_id:
            return
        if (
            not cfg.group_features.ban_allow_admin
            and sender_id in self._get_admin_sid_set()
        ):
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=sender_id
            )
            if active_ban:
                logger.info(
                    "enhance-mode | 管理员保护生效，命中封禁但按 bypass 放行(不修改数据库) | "
                    f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
                    f"scope={scope_id} origin={event.unified_msg_origin}"
                )
            return

        active_ban = self.ban_store.get_active_ban(scope_id=scope_id, user_id=sender_id)
        if not active_ban:
            return

        logger.info(
            "enhance-mode | 命中封禁名单，已拦截消息 | "
            f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
            f"scope={scope_id} origin={event.unified_msg_origin}"
        )
        event.stop_event()

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        cfg = self._cfg()
        if not cfg.group_history_enabled and not cfg.active_reply_enabled:
            return

        has_content = any(
            isinstance(comp, (Plain, Image, Reply))
            for comp in event.message_obj.message
        )
        if not has_content:
            return

        need_active = await self._need_active_reply(event, cfg)

        if cfg.group_history_enabled:
            try:
                await self._record_message(event, cfg)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
                logger.info(
                    "enhance-mode | active_reply triggered | origin=%s mode=%s provider=%s",
                    event.unified_msg_origin,
                    cfg.active_reply.mode,
                    self._provider_label(provider),
                )
                if hasattr(event, "set_extra"):
                    event.set_extra("_enhance_active_reply_triggered", True)
                    event.set_extra("_enhance_active_reply_mode", cfg.active_reply.mode)

                session_curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if not session_curr_cid:
                    logger.error(
                        "enhance-mode | 当前未处于对话状态，无法主动回复，"
                        "请使用 /switch 或 /new 创建一个会话。"
                    )
                    return

                conv = await self.context.conversation_manager.get_conversation(
                    event.unified_msg_origin,
                    session_curr_cid,
                )
                if not conv:
                    logger.error("enhance-mode | 未找到对话，无法主动回复")
                    return

                yield event.request_llm(
                    prompt=event.message_str,
                    session_id=event.session_id,
                    conversation=conv,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"enhance-mode | 主动回复失败: {e}")

    async def _record_message(self, event: AstrMessageEvent, cfg: PluginConfig) -> None:
        history_cfg = cfg.group_history

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname
        msg_id = event.message_obj.message_id
        normalized_msg_id = self._normalize_message_id(msg_id)
        image_urls: list[str] = []

        if history_cfg.include_sender_id and history_cfg.include_role_tag:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{sender_id}/{datetime_str}]{role_tag} #msg{msg_id}:"
        elif history_cfg.include_sender_id:
            sender_id = event.get_sender_id()
            header = f"[{nickname}/{sender_id}/{datetime_str}] #msg{msg_id}:"
        elif history_cfg.include_role_tag:
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{datetime_str}]{role_tag} #msg{msg_id}:"
        else:
            header = f"[{nickname}/{datetime_str}] #msg{msg_id}:"

        parts = [header]
        for comp in event.get_messages():
            if isinstance(comp, Reply):
                quote_nick = comp.sender_nickname or "Unknown"
                quote_text = (comp.message_str or "").strip() or "..."
                quote_id = self._normalize_message_id(getattr(comp, "id", ""))
                if quote_id:
                    parts.append(f" [Quote #msg{quote_id} {quote_nick}: {quote_text}]")
                else:
                    parts.append(f" [Quote {quote_nick}: {quote_text}]")
            elif isinstance(comp, Plain):
                parts.append(f" {comp.text}")
            elif isinstance(comp, Image):
                image_url = str(comp.url or comp.file or "").strip()
                image_urls.append(image_url)
                parts.append(" [Image]")
            elif isinstance(comp, At):
                parts.append(f" [At: {comp.name}]")

        final_message = "".join(parts)
        logger.debug(f"enhance-mode | {event.unified_msg_origin} | {final_message}")

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > history_cfg.max_messages:
            removed_line = chats.pop(0)
            removed_msg_id = self._extract_message_id_from_history_line(removed_line)
            if removed_msg_id:
                self.runtime.image_message_registry[event.unified_msg_origin].pop(
                    removed_msg_id, None
                )
        if normalized_msg_id and image_urls:
            self.runtime.image_message_registry[event.unified_msg_origin][
                normalized_msg_id
            ] = {"urls": image_urls, "captions": {}}
            logger.debug(
                "enhance-mode | image message registered | origin=%s msg_id=%s image_count=%s deferred_caption=%s",
                event.unified_msg_origin,
                normalized_msg_id,
                len(image_urls),
                history_cfg.image_caption,
            )
        logger.debug(
            "enhance-mode | group history updated | origin=%s size=%s",
            event.unified_msg_origin,
            len(chats),
        )

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return

        self._touch_origin(event.unified_msg_origin, cfg)
        bounded_chats = bounded_chat_history_text(
            self.runtime.session_chats[event.unified_msg_origin]
        )
        logger.debug(
            "enhance-mode | injecting group context | origin=%s history_size=%s",
            event.unified_msg_origin,
            len(self.runtime.session_chats[event.unified_msg_origin]),
        )
        interaction_instructions = build_interaction_instructions(
            cfg.group_features.mention_parse,
            cfg.group_history.include_sender_id,
        )
        interaction_instructions += (
            "\nIf a history message contains `[Image]` and visual details are necessary, "
            "you may call `enhance_use_image(message_id, image_index, attach_to_model, write_to_history, prompt)`. "
            "By default it does both: attach image to this run context and write description back into chat history. "
            "Set `attach_to_model=false` for history-only. "
            "Set `write_to_history=false` for attach-only."
        )

        if (
            cfg.group_features.react_mode_enable
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        ):
            is_active_triggered = event.get_extra(
                "_enhance_active_reply_triggered", False
            )
            active_mode = event.get_extra("_enhance_active_reply_mode", "")
            if is_active_triggered and active_mode == "model_choice":
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    "You decided to actively join this conversation because some recent messages are worth replying to.\n"
                    "Choose the message(s) you want to respond to from the chat history above, "
                    "and compose a natural reply. Quote the message you choose in most cases.\n"
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            else:
                prompt = req.prompt
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    f"Now, a new message is coming: `{prompt}`. "
                    "Please react to it. Your entire output is your reply to this message. "
                    "Quote the message which is coming in most cases. "
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            req.contexts = []
        else:
            req.system_prompt += (
                "You are now in a chatroom. The chat history is as follows: \n"
            )
            req.system_prompt += bounded_chats
            req.system_prompt += interaction_instructions

    @filter.on_decorating_result()
    async def parse_tags(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if not result or not result.chain:
            return

        if result.result_content_type in (
            ResultContentType.STREAMING_RESULT,
            ResultContentType.STREAMING_FINISH,
        ):
            logger.debug(
                "enhance-mode | streaming finish detected, skip parse_tags to avoid late-stage mutation",
            )
            return

        # 全局拦截：模型输出 <refuse/> 时直接清空结果链，阻止后续发送到平台。
        if chain_has_refuse_tag(result.chain):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，已取消发送 | "
                f"origin={event.unified_msg_origin}"
            )
            if hasattr(event, "set_extra"):
                event.set_extra("_enhance_refused_reply", True)
            result.chain = []
            return

        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        transformed = transform_result_chain(
            result.chain,
            parse_mention=self._cfg().group_features.mention_parse,
        )
        if transformed is None:
            return

        result.chain = transformed

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return
        if not resp.completion_text:
            return

        if has_refuse_tag(resp.completion_text):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，跳过机器人回复历史记录 | "
                f"origin={event.unified_msg_origin}"
            )
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        text = clean_response_text_for_history(resp.completion_text)
        if not text:
            return
        final_message = f"[You/{datetime_str}]: {text}"

        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > cfg.group_history.max_messages:
            removed_line = chats.pop(0)
            removed_msg_id = self._extract_message_id_from_history_line(removed_line)
            if removed_msg_id:
                self.runtime.image_message_registry[event.unified_msg_origin].pop(
                    removed_msg_id, None
                )
        logger.debug(
            "enhance-mode | bot response recorded | origin=%s size=%s",
            event.unified_msg_origin,
            len(chats),
        )

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return
        self.runtime.cleanup_origin(event.unified_msg_origin)
        logger.info(
            "enhance-mode | runtime session cache cleaned | origin=%s",
            event.unified_msg_origin,
        )

    @llm_tool(name="enhance_get_ban_list_status")
    async def get_ban_list_status(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        max_results: int = 20,
    ) -> str:
        """Get ban-list status maintained by enhance mode.

        Args:
            user_id(string): Optional. Target user ID to query. Empty means list active bans.
            max_results(number): Optional. Maximum number of active bans to return when user_id is empty.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban list is group-scoped. "
                "Please call this tool in current group chat context."
            )

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        target_user_id = str(user_id or "").strip().lstrip("@")
        if (
            target_user_id
            and not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id}, auto released this round: {released_count})."
            )

        if target_user_id:
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=target_user_id
            )
            if not active_ban:
                return (
                    f"User `{target_user_id}` is not banned now. "
                    f"(scope={scope_id}, auto released this round: {released_count})"
                )

            expires_at_text = datetime.datetime.fromtimestamp(
                active_ban.expires_at
            ).strftime("%Y-%m-%d %H:%M:%S")
            return (
                f"User `{target_user_id}` is currently banned.\n"
                f"- Remaining: {self._format_duration(active_ban.remaining_seconds)}\n"
                f"- Expires at: {expires_at_text}\n"
                f"- Scope: {scope_id}\n"
                f"- Auto released this round: {released_count}"
            )

        try:
            parsed_max_results = int(max_results)
        except (TypeError, ValueError):
            parsed_max_results = 20
        limit = max(1, min(parsed_max_results, 200))
        records = self.ban_store.list_active_bans(scope_id=scope_id, limit=limit)
        if not records:
            return (
                f"No active bans in scope `{scope_id}`. "
                f"(auto released this round: {released_count})"
            )

        lines = [
            f"Active bans in scope `{scope_id}` ({len(records)} shown, limit={limit}). "
            f"Auto released this round: {released_count}",
        ]
        for idx, record in enumerate(records, start=1):
            expire_text = datetime.datetime.fromtimestamp(record.expires_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            lines.append(
                f"{idx}. user_id={record.user_id}, remaining={self._format_duration(record.remaining_seconds)}, "
                f"expires_at={expire_text}"
            )
        return "\n".join(lines)

    @llm_tool(name="enhance_ban_user")
    async def ban_user(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration: str = "10m",
    ) -> str:
        """Temporarily ban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to ban.
            duration(string): Optional. Ban duration like 60s, 10m, 2h, 1d. Default is 10m.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        duration_seconds = parse_duration_seconds(duration)
        if duration_seconds is None:
            return "Invalid `duration`. Use formats like `60s`, `10m`, `2h`, `1d`."

        if (
            not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id})."
            )

        final_duration = max(
            1,
            min(duration_seconds, cfg.group_features.ban_max_duration_sec),
        )
        expires_at = self.ban_store.ban_user(
            scope_id=scope_id,
            user_id=target_user_id,
            duration_seconds=final_duration,
            source_origin=event.unified_msg_origin,
        )
        expire_time = datetime.datetime.fromtimestamp(expires_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logger.info(
            "enhance-mode | ban applied | scope=%s user_id=%s duration=%s origin=%s",
            scope_id,
            target_user_id,
            self._format_duration(final_duration),
            event.unified_msg_origin,
        )
        return (
            f"Banned user `{target_user_id}` in scope `{scope_id}`.\n"
            f"- Duration: {self._format_duration(final_duration)}\n"
            f"- Expires at: {expire_time}"
        )

    @llm_tool(name="enhance_unban_user")
    async def unban_user(self, event: AstrMessageEvent, user_id: str) -> str:
        """Unban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to unban.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Unban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        removed = self.ban_store.unban_user(scope_id=scope_id, user_id=target_user_id)
        if not removed:
            return f"User `{target_user_id}` is not banned in scope `{scope_id}`."
        logger.info(
            "enhance-mode | unban applied | scope=%s user_id=%s origin=%s",
            scope_id,
            target_user_id,
            event.unified_msg_origin,
        )
        return f"Unbanned user `{target_user_id}` in scope `{scope_id}`."

    @staticmethod
    def _make_text_tool_result(text: str) -> mcp_types.CallToolResult:
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=str(text or ""))]
        )

    @llm_tool(name="enhance_use_image")
    async def use_image(
        self,
        event: AstrMessageEvent,
        message_id: str,
        image_index: int = 1,
        attach_to_model: bool = True,
        write_to_history: bool = True,
        prompt: str = "",
    ) -> AsyncGenerator[mcp_types.CallToolResult, None]:
        """Use one image from runtime history for multimodal input and/or history backfill.

        Args:
            message_id(string): Required. Message ID in history header, for example `123456`.
            image_index(number): Optional. One-based index of image in that message. Default is 1.
            attach_to_model(bool): Optional. Attach image bytes to this run context. Default is true.
            write_to_history(bool): Optional. Replace `[Image]` with description in history. Default is true.
            prompt(string): Optional. Override image caption prompt for this call.
        """
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            yield self._make_text_tool_result(
                "Group history enhancement is disabled in enhance mode config."
            )
            return

        normalized_message_id = self._normalize_message_id(message_id)
        if not normalized_message_id:
            yield self._make_text_tool_result("Invalid `message_id`: empty.")
            return

        try:
            index_number = int(image_index)
        except (TypeError, ValueError):
            yield self._make_text_tool_result(
                "Invalid `image_index`: must be a positive integer."
            )
            return
        if index_number <= 0:
            yield self._make_text_tool_result("Invalid `image_index`: must be >= 1.")
            return
        image_idx = index_number - 1

        attach_requested = bool(attach_to_model)
        history_requested = bool(write_to_history)
        if not attach_requested and not history_requested:
            yield self._make_text_tool_result(
                "Invalid mode: `attach_to_model` and `write_to_history` cannot both be false."
            )
            return

        origin = event.unified_msg_origin
        message_registry = self.runtime.image_message_registry.get(origin, {})
        message_entry = message_registry.get(normalized_message_id)
        if not isinstance(message_entry, dict):
            yield self._make_text_tool_result(
                f"Image message `{normalized_message_id}` not found in current runtime history. "
                "Try a newer message ID from current chat context."
            )
            return

        urls_raw = message_entry.get("urls")
        if not isinstance(urls_raw, list) or not urls_raw:
            yield self._make_text_tool_result(
                f"No image records found for message `{normalized_message_id}`."
            )
            return
        if image_idx >= len(urls_raw):
            yield self._make_text_tool_result(
                f"`image_index` out of range. message `{normalized_message_id}` has "
                f"{len(urls_raw)} image(s)."
            )
            return

        image_url = str(urls_raw[image_idx] or "").strip()
        if not image_url:
            yield self._make_text_tool_result(
                f"Image URL is unavailable for message `{normalized_message_id}` "
                f"at index {index_number}."
            )
            return

        captions_raw = message_entry.get("captions")
        if isinstance(captions_raw, dict):
            captions_map = captions_raw
        else:
            captions_map = {}
            message_entry["captions"] = captions_map

        caption = ""
        caption_cached = False
        cached_caption = captions_map.get(image_idx)
        if isinstance(cached_caption, str) and cached_caption.strip():
            caption = cached_caption.strip()
            caption_cached = True
        elif history_requested:
            try:
                if not cfg.group_history.image_caption:
                    yield self._make_text_tool_result(
                        "Image caption is disabled in enhance mode config."
                    )
                    return
                final_prompt = (
                    str(prompt or "").strip() or cfg.group_history.image_caption_prompt
                )
                caption = await self._get_image_caption(
                    image_url=image_url,
                    provider_id=cfg.group_history.image_caption_provider_id,
                    prompt=final_prompt,
                    timeout_sec=cfg.global_settings.timeouts.image_caption_sec,
                )
                caption = str(caption or "").strip()
                if caption:
                    captions_map[image_idx] = caption
            except Exception as e:
                logger.exception(
                    "enhance-mode | use_image caption failed | origin=%s msg_id=%s image_index=%s error=%s",
                    origin,
                    normalized_message_id,
                    index_number,
                    e,
                )
                yield self._make_text_tool_result(
                    f"Failed to get image description: {e}"
                )
                return

        attach_success = False
        attach_error = ""
        if attach_requested:
            resolved_paths_raw = message_entry.get("resolved_paths")
            if not isinstance(resolved_paths_raw, list) or len(
                resolved_paths_raw
            ) != len(urls_raw):
                resolved_paths_raw = [""] * len(urls_raw)
                message_entry["resolved_paths"] = resolved_paths_raw

            selected_ref = str(
                resolved_paths_raw[image_idx] or urls_raw[image_idx] or ""
            ).strip()
            if not selected_ref:
                attach_error = (
                    f"Image reference is unavailable for message `{normalized_message_id}` "
                    f"at index {index_number}."
                )
            else:
                try:
                    local_path = await self._resolve_image_ref_to_local_path(
                        selected_ref
                    )
                    if not local_path:
                        attach_error = (
                            f"Failed to resolve image path for message `{normalized_message_id}` "
                            f"index {index_number}."
                        )
                    else:
                        resolved_paths_raw[image_idx] = local_path
                        image_b64, mime_type = self._encode_image_file(local_path)
                        yield mcp_types.CallToolResult(
                            content=[
                                mcp_types.ImageContent(
                                    type="image",
                                    data=image_b64,
                                    mimeType=mime_type,
                                )
                            ]
                        )
                        attach_success = True
                        logger.info(
                            "enhance-mode | use_image attach success | origin=%s msg_id=%s image_index=%s mime=%s",
                            origin,
                            normalized_message_id,
                            index_number,
                            mime_type,
                        )
                except Exception as e:
                    attach_error = str(e)
                    logger.exception(
                        "enhance-mode | use_image attach failed | origin=%s msg_id=%s image_index=%s error=%s",
                        origin,
                        normalized_message_id,
                        index_number,
                        e,
                    )

        history_success = False
        history_error = ""
        if history_requested:
            if not caption:
                history_error = "Image description is empty."
            else:
                history_success = self._apply_image_caption_to_history(
                    origin=origin,
                    message_id=normalized_message_id,
                    image_index=image_idx,
                    caption=caption,
                )
                if history_success:
                    logger.info(
                        "enhance-mode | use_image history write success | origin=%s msg_id=%s image_index=%s",
                        origin,
                        normalized_message_id,
                        index_number,
                    )
                else:
                    history_error = (
                        "Failed to apply image description back to runtime history."
                    )

        if attach_requested and history_requested:
            success = attach_success and history_success
        elif attach_requested:
            success = attach_success
        else:
            success = history_success

        payload: dict[str, object] = {
            "status": "ok" if success else "failed",
            "success": success,
            "origin": origin,
            "message_id": normalized_message_id,
            "image_index": index_number,
            "attach_requested": attach_requested,
            "write_to_history_requested": history_requested,
            "attach_success": attach_success if attach_requested else None,
            "write_to_history_success": history_success if history_requested else None,
            "description_cached": caption_cached,
        }
        if attach_requested and not history_requested:
            payload["description"] = caption
        if attach_error:
            payload["attach_error"] = attach_error
        if history_error:
            payload["write_to_history_error"] = history_error

        logger.info(
            "enhance-mode | use_image done | origin=%s msg_id=%s image_index=%s attach=%s/%s write=%s/%s success=%s",
            origin,
            normalized_message_id,
            index_number,
            attach_success,
            attach_requested,
            history_success,
            history_requested,
            success,
        )
        yield self._make_text_tool_result(json.dumps(payload, ensure_ascii=False))

    @llm_tool(name="enhance_memory_rag_write")
    async def memory_rag_write(
        self,
        event: AstrMessageEvent,
        content: str,
        related_role_ids: str,
        memory_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        extra_metadata_json: str = "{}",
    ) -> str:
        """Write a memory record into enhance-mode memory RAG store.

        Args:
            content(string): Required. Memory content text.
            related_role_ids(string): Required. Role IDs in JSON array or comma-separated string.
            memory_time(string): Optional. Memory time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            extra_metadata_json(string): Optional. JSON object string for extra metadata.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        clean_content = str(content or "").strip()
        if not clean_content:
            return "Invalid `content`: empty."

        role_ids = self._parse_role_ids(related_role_ids)
        if not role_ids:
            return "Invalid `related_role_ids`: at least one role ID is required."

        ts = self._parse_optional_timestamp(memory_time)
        if memory_time and ts is None:
            return (
                "Invalid `memory_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        memory_ts = ts if ts is not None else time.time()

        final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
            event, group_scope, group_id, platform_id
        )

        cfg = self._cfg()
        embedding_provider = self._resolve_embedding_provider(cfg)
        if not embedding_provider:
            return (
                "No embedding provider is available. "
                "Please configure one in AstrBot provider settings."
            )

        start_ts = time.perf_counter()
        logger.info(
            "enhance-mode | memory_rag_write start | origin=%s roles=%s content_len=%s scope=%s",
            event.unified_msg_origin,
            len(role_ids),
            len(clean_content),
            final_scope,
        )
        try:
            embedding = await embedding_provider.get_embedding(clean_content)
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_write embedding failed: %s", e)
            return f"Failed to generate embedding: {e}"

        metadata = self._parse_extra_metadata(extra_metadata_json)
        try:
            memory_id = self.memory_rag_store.add_memory(
                content=clean_content,
                embedding=embedding,
                role_ids=role_ids,
                memory_time=memory_ts,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                extra_metadata=metadata,
            )
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_write store failed: %s", e)
            return f"Failed to write memory: {e}"

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        logger.info(
            "enhance-mode | memory_rag_write done | memory_id=%s provider=%s embedding_dim=%s elapsed_ms=%.1f",
            memory_id,
            self._provider_label(embedding_provider),
            len(embedding),
            elapsed_ms,
        )

        return json.dumps(
            {
                "status": "ok",
                "memory_id": memory_id,
                "memory_time": memory_ts,
                "memory_time_iso": self._format_timestamp_iso(memory_ts),
                "group_scope": final_scope,
                "group_id": final_group_id,
                "platform_id": final_platform_id,
                "related_role_ids": role_ids,
            },
            ensure_ascii=False,
        )

    @llm_tool(name="enhance_memory_rag_read")
    async def memory_rag_read(
        self,
        event: AstrMessageEvent,
        query: str = "",
        related_role_ids: str = "",
        role_match_mode: str = "any",
        start_time: str = "",
        end_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        sort_by: str = "relevance",
        sort_order: str = "desc",
        max_results: int = 10,
        embedding_recall_k: int = 0,
        ignore_group_id: bool = False,
    ) -> str:
        """Read memory records from enhance-mode memory RAG store.

        Args:
            query(string): Optional. Query text for embedding retrieval.
            related_role_ids(string): Optional. Role IDs in JSON array or comma-separated string.
            role_match_mode(string): Optional. `any` or `all`.
            start_time(string): Optional. Start time as unix timestamp or ISO datetime.
            end_time(string): Optional. End time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            sort_by(string): Optional. `relevance` or `time`.
            sort_order(string): Optional. `desc` or `asc`.
            max_results(number): Optional. Number of results to return.
            embedding_recall_k(number): Optional. Top-K cutoff for embedding recall before final sorting.
            ignore_group_id(boolean): Optional. When true, do not auto-apply current group filter, allowing cross-group read.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        start_time_ts = self._parse_optional_timestamp(start_time)
        end_time_ts = self._parse_optional_timestamp(end_time)
        if start_time and start_time_ts is None:
            return (
                "Invalid `start_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if end_time and end_time_ts is None:
            return (
                "Invalid `end_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if (
            start_time_ts is not None
            and end_time_ts is not None
            and start_time_ts > end_time_ts
        ):
            return "Invalid time range: `start_time` cannot be greater than `end_time`."

        normalized_ignore_group = (
            ignore_group_id
            if isinstance(ignore_group_id, bool)
            else str(ignore_group_id or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if normalized_ignore_group:
            final_scope = str(group_scope or "").strip()
            final_group_id = str(group_id or "").strip()
            final_platform_id = str(platform_id or "").strip()
            if not final_scope and final_group_id:
                final_scope = (
                    f"{final_platform_id}:{final_group_id}"
                    if final_platform_id
                    else final_group_id
                )
        else:
            final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
                event, group_scope, group_id, platform_id
            )
        normalized_roles = self._parse_role_ids(related_role_ids)
        normalized_mode = (
            "all" if str(role_match_mode or "").strip().lower() == "all" else "any"
        )
        normalized_sort_by = self._normalize_sort_by(sort_by)
        normalized_sort_order = self._normalize_sort_order(sort_order)

        cfg = self._cfg()
        max_allowed = max(1, cfg.memory_rag.max_return_results)
        try:
            requested_max = int(max_results)
        except (TypeError, ValueError):
            requested_max = 10
        if requested_max <= 0:
            final_max_results = max_allowed
        else:
            final_max_results = min(requested_max, max_allowed)

        try:
            parsed_recall_k = int(embedding_recall_k)
        except (TypeError, ValueError):
            parsed_recall_k = 0
        effective_recall_k = (
            parsed_recall_k
            if parsed_recall_k > 0
            else int(cfg.memory_rag.default_recall_k)
        )

        query_embedding: list[float] | None = None
        clean_query = str(query or "").strip()
        read_start_ts = time.perf_counter()
        logger.info(
            "enhance-mode | memory_rag_read start | origin=%s query_len=%s role_count=%s scope=%s max_results=%s sort=%s/%s",
            event.unified_msg_origin,
            len(clean_query),
            len(normalized_roles),
            final_scope,
            final_max_results,
            normalized_sort_by,
            normalized_sort_order,
        )
        if clean_query:
            embedding_provider = self._resolve_embedding_provider(cfg)
            if not embedding_provider:
                return (
                    "No embedding provider is available. "
                    "Please configure one in AstrBot provider settings."
                )
            try:
                query_embedding = await embedding_provider.get_embedding(clean_query)
            except Exception as e:
                logger.exception(
                    "enhance-mode | memory_rag_read embedding failed: %s", e
                )
                return f"Failed to generate query embedding: {e}"

        try:
            records = self.memory_rag_store.search_memories(
                query_embedding=query_embedding,
                embedding_recall_k=effective_recall_k,
                role_ids=normalized_roles,
                role_match_mode=normalized_mode,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                start_time=start_time_ts,
                end_time=end_time_ts,
                sort_by=normalized_sort_by,
                sort_order=normalized_sort_order,
                max_results=final_max_results,
            )
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_read search failed: %s", e)
            return f"Failed to read memories: {e}"

        elapsed_ms = (time.perf_counter() - read_start_ts) * 1000
        logger.info(
            "enhance-mode | memory_rag_read done | origin=%s result_count=%s elapsed_ms=%.1f",
            event.unified_msg_origin,
            len(records),
            elapsed_ms,
        )

        return json.dumps(
            {
                "status": "ok",
                "count": len(records),
                "query": clean_query,
                "filters": {
                    "related_role_ids": normalized_roles,
                    "role_match_mode": normalized_mode,
                    "start_time": start_time_ts,
                    "end_time": end_time_ts,
                    "ignore_group_id": normalized_ignore_group,
                    "group_scope": final_scope,
                    "group_id": final_group_id,
                    "platform_id": final_platform_id,
                },
                "sort": {"by": normalized_sort_by, "order": normalized_sort_order},
                "max_results": final_max_results,
                "embedding_recall_k": effective_recall_k if clean_query else None,
                "records": records,
            },
            ensure_ascii=False,
        )

    @filter.command_group("enhance")
    def enhance(self) -> None:
        pass

    @permission_type(PermissionType.ADMIN)
    @enhance.command("rag-webui")
    async def rag_webui(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        logger.info(
            "enhance-mode | rag-webui command received | origin=%s webui_enable=%s",
            event.unified_msg_origin,
            webui_cfg.enable,
        )
        if not webui_cfg.enable:
            yield event.plain_result(
                "Memory RAG WebUI is disabled.\n"
                "Please enable `memory_rag_webui.enable` in enhance plugin config."
            )
            return

        if self.rag_webui_server is None:
            await self._start_memory_rag_webui()

        if self.rag_webui_server is None:
            yield event.plain_result(
                "Memory RAG WebUI failed to start.\n"
                "Please check AstrBot logs for error details."
            )
            return

        message = (
            "Enhance Memory RAG WebUI\n\n"
            f"URL: {self._memory_rag_webui_url(cfg)}\n"
            "Functions: login, filters, pagination, detail view, delete memory.\n"
        )
        if self.rag_webui_server.password_generated:
            message += (
                "\nPassword is auto generated for this run. "
                "Please check AstrBot logs for the generated password."
            )
        yield event.plain_result(message)

    async def terminate(self) -> None:
        logger.info("enhance-mode | plugin terminating")
        await self._stop_memory_rag_webui()
        logger.info("enhance-mode | plugin terminated")
