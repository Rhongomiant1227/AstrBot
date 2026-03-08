import ast
import asyncio
import base64
import inspect
import json
import random
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import NotFoundError
from openai.lib.streaming.chat._completions import ChatCompletionStreamState
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.api.provider import Provider
from astrbot.core.agent.message import ContentPart, ImageURLPart, Message, TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, TokenUsage, ToolCallsResult
from astrbot.core.utils.io import download_image_by_url
from astrbot.core.utils.network_utils import (
    create_proxy_client,
    is_connection_error,
    log_connection_failure,
)
from astrbot.core.utils.string_utils import normalize_and_dedupe_strings

from ..register import register_provider_adapter


@register_provider_adapter(
    "openai_chat_completion",
    "OpenAI API Chat Completion 提供商适配器",
)
class ProviderOpenAIOfficial(Provider):
    _ERROR_TEXT_CANDIDATE_MAX_CHARS = 4096

    @classmethod
    def _truncate_error_text_candidate(cls, text: str) -> str:
        if len(text) <= cls._ERROR_TEXT_CANDIDATE_MAX_CHARS:
            return text
        return text[: cls._ERROR_TEXT_CANDIDATE_MAX_CHARS]

    @staticmethod
    def _safe_json_dump(value: Any) -> str | None:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return None

    def _get_image_moderation_error_patterns(self) -> list[str]:
        """Return configured moderation patterns (case-insensitive substring match, not regex)."""
        configured = self.provider_config.get("image_moderation_error_patterns", [])
        patterns: list[str] = []
        if isinstance(configured, str):
            configured = [configured]
        if isinstance(configured, list):
            for pattern in configured:
                if not isinstance(pattern, str):
                    continue
                pattern = pattern.strip()
                if pattern:
                    patterns.append(pattern)
        return patterns

    @staticmethod
    def _extract_error_text_candidates(error: Exception) -> list[str]:
        candidates: list[str] = []

        def _append_candidate(candidate: Any):
            if candidate is None:
                return
            text = str(candidate).strip()
            if not text:
                return
            candidates.append(
                ProviderOpenAIOfficial._truncate_error_text_candidate(text)
            )

        _append_candidate(str(error))

        body = getattr(error, "body", None)
        if isinstance(body, dict):
            err_obj = body.get("error")
            body_text = ProviderOpenAIOfficial._safe_json_dump(
                {"error": err_obj} if isinstance(err_obj, dict) else body
            )
            _append_candidate(body_text)
            if isinstance(err_obj, dict):
                for field in ("message", "type", "code", "param"):
                    value = err_obj.get(field)
                    if value is not None:
                        _append_candidate(value)
        elif isinstance(body, str):
            _append_candidate(body)

        response = getattr(error, "response", None)
        if response is not None:
            response_text = getattr(response, "text", None)
            if isinstance(response_text, str):
                _append_candidate(response_text)

        return normalize_and_dedupe_strings(candidates)

    def _is_content_moderated_upload_error(self, error: Exception) -> bool:
        patterns = [
            pattern.lower() for pattern in self._get_image_moderation_error_patterns()
        ]
        if not patterns:
            return False
        candidates = [
            candidate.lower()
            for candidate in self._extract_error_text_candidates(error)
        ]
        for pattern in patterns:
            if any(pattern in candidate for candidate in candidates):
                return True
        return False

    @staticmethod
    def _context_contains_image(contexts: list[dict]) -> bool:
        for context in contexts:
            content = context.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
        return False

    async def _fallback_to_text_only_and_retry(
        self,
        payloads: dict,
        context_query: list,
        chosen_key: str,
        available_api_keys: list[str],
        func_tool: ToolSet | None,
        reason: str,
        *,
        image_fallback_used: bool = False,
    ) -> tuple:
        logger.warning(
            "检测到图片请求失败（%s），已移除图片并重试（保留文本内容）。",
            reason,
        )
        new_contexts = await self._remove_image_from_context(context_query)
        payloads["messages"] = new_contexts
        return (
            False,
            chosen_key,
            available_api_keys,
            payloads,
            new_contexts,
            func_tool,
            image_fallback_used,
        )

    def _create_http_client(self, provider_config: dict) -> httpx.AsyncClient | None:
        """创建带代理的 HTTP 客户端"""
        proxy = provider_config.get("proxy", "")
        return create_proxy_client("OpenAI", proxy)

    def __init__(self, provider_config, provider_settings) -> None:
        super().__init__(provider_config, provider_settings)
        self.chosen_api_key = None
        self.api_keys: list = super().get_keys()
        self.chosen_api_key = self.api_keys[0] if len(self.api_keys) > 0 else None
        self.timeout = provider_config.get("timeout", 120)
        self.custom_headers = provider_config.get("custom_headers", {})
        if isinstance(self.timeout, str):
            self.timeout = int(self.timeout)

        if not isinstance(self.custom_headers, dict) or not self.custom_headers:
            self.custom_headers = None
        else:
            for key in self.custom_headers:
                self.custom_headers[key] = str(self.custom_headers[key])

        wire_api = str(provider_config.get("wire_api", "chat_completions")).lower()
        client_max_retries = provider_config.get("client_max_retries")
        if client_max_retries is None and wire_api in {"responses", "sub2api"}:
            # Responses/sub2api already has provider-level retry handling. Avoid SDK retries
            # stacking on top and turning one bad request into a multi-minute stall.
            client_max_retries = 0
        if isinstance(client_max_retries, str):
            client_max_retries = int(client_max_retries)
        if client_max_retries is not None:
            client_max_retries = max(0, int(client_max_retries))

        shared_client_kwargs = {
            "api_key": self.chosen_api_key,
            "default_headers": self.custom_headers,
            "timeout": self.timeout,
            "http_client": self._create_http_client(provider_config),
        }
        if client_max_retries is not None:
            shared_client_kwargs["max_retries"] = client_max_retries

        if "api_version" in provider_config:
            # Using Azure OpenAI API
            self.client = AsyncAzureOpenAI(
                api_version=provider_config.get("api_version", None),
                base_url=provider_config.get("api_base", ""),
                **shared_client_kwargs,
            )
        else:
            # Using OpenAI Official API
            self.client = AsyncOpenAI(
                base_url=provider_config.get("api_base", None),
                **shared_client_kwargs,
            )

        self.wire_api = wire_api
        self.sub2api_mode = self.wire_api == "sub2api"
        self.use_responses_api = self.wire_api in {"responses", "sub2api"}
        # sub2api compatibility mode has unstable stream semantics on some upstreams.
        self.responses_stream_supported = not self.sub2api_mode
        self.sub2api_tool_adapter_enabled = self.sub2api_mode and bool(
            provider_config.get("sub2api_tool_adapter", True),
        )
        self.sub2api_force_python_for_deterministic = self.sub2api_mode and bool(
            provider_config.get(
                "sub2api_force_python_for_deterministic",
                False,
            ),
        )
        self.sub2api_force_python_retry_once = self.sub2api_mode and bool(
            provider_config.get("sub2api_force_python_retry_once", False)
        )

        self.chat_default_params = set(
            inspect.signature(
                self.client.chat.completions.create,
            ).parameters.keys()
        )
        self.responses_default_params: set[str] = set()
        if hasattr(self.client, "responses") and hasattr(
            self.client.responses, "create"
        ):
            self.responses_default_params = set(
                inspect.signature(
                    self.client.responses.create,
                ).parameters.keys()
            )
        self.default_params = (
            self.responses_default_params
            if self.use_responses_api and self.responses_default_params
            else self.chat_default_params
        )

        model = provider_config.get("model", "unknown")
        self.set_model(model)

        self.reasoning_key = "reasoning_content"

    async def get_models(self):
        try:
            models_str = []
            models = await self.client.models.list()
            models = sorted(models.data, key=lambda x: x.id)
            for model in models:
                models_str.append(model.id)
            return models_str
        except NotFoundError as e:
            raise Exception(f"获取模型列表失败：{e}")

    @staticmethod
    def _to_plain_data(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool, dict, list)):
            return obj
        for method_name in ("model_dump", "to_dict"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    continue
        return obj

    @classmethod
    def _to_plain_dict(cls, obj: Any) -> dict:
        data = cls._to_plain_data(obj)
        return data if isinstance(data, dict) else {}

    @classmethod
    def _extract_error_status_code(cls, error: Exception) -> int | None:
        code = getattr(error, "status_code", None)
        if isinstance(code, int):
            return code
        response = getattr(error, "response", None)
        if response is not None:
            response_code = getattr(response, "status_code", None)
            if isinstance(response_code, int):
                return response_code
        body = getattr(error, "body", None)
        body_dict = cls._to_plain_dict(body)
        maybe_code = body_dict.get("status_code")
        if isinstance(maybe_code, int):
            return maybe_code
        return None

    @classmethod
    def _is_upstream_server_error(cls, error: Exception) -> bool:
        status_code = cls._extract_error_status_code(error)
        if status_code in {500, 502, 503, 504}:
            return True
        text = str(error).lower()
        if "upstream request failed" in text:
            return True
        if "bad gateway" in text:
            return True
        if '"type":"server_error"' in text or "'type': 'server_error'" in text:
            return True
        if "event: error" in text and "response.failed" in text:
            return True
        if "an error occurred while processing your request" in text:
            return True
        if '"status":"failed"' in text and "server_error" in text:
            return True
        return False

    def _split_payload_and_extra_body(
        self,
        payloads: dict,
        default_params: set[str],
    ) -> tuple[dict, dict]:
        request_payload = {}
        extra_body = {}
        for key, value in payloads.items():
            if key in default_params:
                request_payload[key] = value
            else:
                extra_body[key] = value

        custom_extra_body = self.provider_config.get("custom_extra_body", {})
        if isinstance(custom_extra_body, dict):
            extra_body.update(custom_extra_body)

        return request_payload, extra_body

    @staticmethod
    def _convert_tools_for_responses(tool_list: list[dict]) -> list[dict]:
        converted: list[dict] = []
        for tool in tool_list:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(
                tool.get("function"), dict
            ):
                function_payload = tool["function"]
                converted_tool = {
                    "type": "function",
                    "name": function_payload.get("name", ""),
                    "parameters": function_payload.get("parameters")
                    or {"type": "object", "properties": {}},
                    "strict": function_payload.get("strict", False),
                }
                if function_payload.get("description") is not None:
                    converted_tool["description"] = function_payload.get("description")
                converted.append(converted_tool)
                continue
            converted.append(tool)
        return converted

    def _build_sub2api_tool_adapter_prompt(
        self,
        tool_list: list[dict],
        *,
        post_tool_phase: bool = False,
        force_python_tool_name: str | None = None,
        force_handoff_tool_name: str | None = None,
        force_reason: str = "",
        python_retry_reason: str = "",
    ) -> str:
        tool_defs: list[dict] = []
        for raw_tool in tool_list:
            tool = self._to_plain_dict(raw_tool)
            if tool.get("type") != "function":
                continue
            function_payload = self._to_plain_dict(tool.get("function"))
            name = function_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            parameters = function_payload.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            tool_defs.append(
                {
                    "name": name.strip(),
                    "description": str(function_payload.get("description") or ""),
                    "parameters": parameters,
                },
            )

        if not tool_defs:
            return ""

        phase_name = "phase_2_post_tool" if post_tool_phase else "phase_1_pre_tool"
        if post_tool_phase:
            phase_rules = (
                "You are in phase 2 (post-tool synthesis). Tool outputs already exist.\n"
                "If the existing tool outputs are enough, answer the user directly and do NOT output tool tags.\n"
                "If additional tool data is still required, output ONLY tool tags and no extra text.\n"
                "Do not call the same tool repeatedly with the same arguments unless new evidence is needed.\n"
            )
        else:
            phase_rules = (
                "You are in phase 1 (tool planning).\n"
                "If a listed tool can help with the user request, output ONLY tool tags and no extra text.\n"
                "If user explicitly mentions a listed tool name, you MUST call that tool in this turn.\n"
                "If no listed tool is suitable, answer normally.\n"
            )

        forced_rules = ""
        if force_python_tool_name and not post_tool_phase:
            forced_rules = (
                "This request appears deterministic / computation-heavy.\n"
                f"Before answering, call `{force_python_tool_name}` with executable Python code to verify the exact result.\n"
                "Do not provide final numeric/list results before seeing tool output.\n"
                "Keep the Python script narrowly scoped to the user's exact input.\n"
                "Avoid brute-force enumeration of global state spaces, broad precomputation, large recursion trees, or oversized caches unless the search space is provably tiny.\n"
                "Prefer direct formulas, candidate checking, bounded loops, and compact data structures.\n"
                f"Trigger reason: {force_reason or 'deterministic_compute'}.\n"
            )

        forced_handoff_rules = ""
        if force_handoff_tool_name and not post_tool_phase:
            forced_handoff_rules = (
                "This request is a structured, multi-step reasoning / derivation task.\n"
                f"Prefer calling `{force_handoff_tool_name}` before answering directly.\n"
                "If the user asks for full derivation, rigorous proof, multi-part solving, exam-style writeup, or careful verification, do NOT answer directly in this turn.\n"
                f"Output ONLY a tool tag for `{force_handoff_tool_name}` and no extra text.\n"
                f"Trigger reason: {force_reason or 'complex_reasoning_handoff'}.\n"
            )

        resource_failure_rules = ""
        if python_retry_reason == "python_resource_memory":
            resource_failure_rules = (
                "The previous Python attempt failed with a sandbox memory limit error.\n"
                "If exact computation is still needed, retry Python with a dramatically smaller script that only solves the current input.\n"
                "Do not enumerate all possibilities, precompute large tables, build huge sets/dicts/lists, or keep broad lru_cache state.\n"
                "Do not tell the user that you will retry; either output ONLY tool tags for the lighter retry or answer directly if the existing tool output is already sufficient.\n"
            )
        elif python_retry_reason == "python_resource_timeout":
            resource_failure_rules = (
                "The previous Python attempt failed due to sandbox timeout / CPU limits.\n"
                "If exact computation is still needed, retry Python with a much smaller and more targeted script for only the current input.\n"
                "Avoid exhaustive search, expensive recursion, broad precomputation, and unnecessary intermediate data.\n"
                "Do not tell the user that you will retry; either output ONLY tool tags for the lighter retry or answer directly if the existing tool output is already sufficient.\n"
            )

        tools_json = self._safe_json_dump(tool_defs) or "[]"
        return (
            "<sub2api_tool_adapter>\n"
            "You are running through a tool compatibility adapter for sub2api.\n"
            f"Current adapter stage: {phase_name}.\n"
            'Tool tag format: <astrbot_tool_call>{"name":"tool_name","arguments":{...}}</astrbot_tool_call>\n'
            "For multiple tools, output multiple <astrbot_tool_call>...</astrbot_tool_call> blocks.\n"
            f"{phase_rules}"
            f"{forced_rules}"
            f"{forced_handoff_rules}"
            f"{resource_failure_rules}"
            "Tool results from previous rounds may appear as role=tool or as [tool_output call_id=...] blocks.\n"
            "Tool definitions (JSON):\n"
            f"{tools_json}\n"
            "</sub2api_tool_adapter>"
        )

    @staticmethod
    def _pick_sub2api_python_tool_name_from_names(tool_names: list[str]) -> str | None:
        preferred_names = ["run_python", "astrbot_execute_python"]
        normalized_names = [
            name for name in tool_names if isinstance(name, str) and name
        ]
        for preferred in preferred_names:
            if preferred in normalized_names:
                return preferred
        for name in normalized_names:
            if "python" in name.lower():
                return name
        return None

    def _pick_sub2api_python_tool_name(self, tool_list: list[dict]) -> str | None:
        names: list[str] = []
        for raw_tool in tool_list:
            tool = self._to_plain_dict(raw_tool)
            function_payload = self._to_plain_dict(tool.get("function"))
            name = function_payload.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return self._pick_sub2api_python_tool_name_from_names(names)

    def _extract_latest_user_text_for_sub2api(self, messages: list[Any]) -> str:
        for raw_message in reversed(messages):
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = self._normalize_content(content, strip=False)
            if text:
                return text
        return ""

    @staticmethod
    def _looks_like_deterministic_compute_query(user_text: str) -> bool:
        if not user_text:
            return False
        text = user_text.lower().replace(" ", "")

        digit_count = sum(ch.isdigit() for ch in text)
        operator_count = len(re.findall(r"[+\-*/=<>%^()\[\]{}]", text))
        structured_tokens = len(
            re.findall(r"(?:\d+[a-z]+|[a-z]+\d+)", text, re.IGNORECASE),
        )
        question_like = "?" in text or "\uff1f" in user_text

        compute_keywords = (
            "\u8ba1\u7b97",  # ??
            "\u7b97\u4e00\u4e0b",  # ???
            "\u6c42",  # ?
            "\u679a\u4e3e",  # ??
            "\u7edf\u8ba1",  # ??
            "\u7ec4\u5408",  # ??
            "\u6392\u5217",  # ??
            "\u6982\u7387",  # ??
            "\u6700\u4f18",  # ??
            "\u63a8\u5bfc",  # ??
            "\u516c\u5f0f",  # ??
            "\u65b9\u7a0b",  # ??
            "calculate",
            "compute",
            "solve",
            "count",
            "enumerate",
            "probability",
            "optimize",
            "exact",
        )

        if any(keyword in text for keyword in compute_keywords):
            if digit_count >= 2 or operator_count >= 1 or structured_tokens >= 1:
                return True

        if operator_count >= 1 and digit_count >= 2:
            return True

        if digit_count >= 8 and (operator_count >= 1 or structured_tokens >= 2):
            return True

        if structured_tokens >= 2 and question_like:
            return True

        # Detect structured deterministic questions without explicit '?'.
        # This keeps the rule domain-agnostic (not Mahjong-specific).
        interrogative_keywords = (
            "什么",
            "多少",
            "几个",
            "几种",
            "哪些",
            "哪个",
            "怎么",
            "what",
            "which",
            "howmany",
            "how much",
            "list all",
            "enumerate",
        )
        has_interrogative_word = any(
            keyword in user_text.lower() for keyword in interrogative_keywords
        )
        if has_interrogative_word and (
            (digit_count >= 9 and structured_tokens >= 1)
            or re.search(r"\d{5,}[a-z]", text, re.IGNORECASE)
        ):
            return True

        return False

    def _detect_sub2api_force_python_for_compute(
        self,
        messages: list[Any],
        tool_list: list[dict],
        *,
        post_tool_phase: bool,
    ) -> tuple[str | None, str]:
        if not self.sub2api_force_python_for_deterministic:
            return None, ""

        if post_tool_phase:
            return None, ""

        latest_user_text = self._extract_latest_user_text_for_sub2api(messages)
        if not self._looks_like_deterministic_compute_query(latest_user_text):
            return None, ""

        python_tool_name = self._pick_sub2api_python_tool_name(tool_list)
        if not python_tool_name:
            return None, ""

        return python_tool_name, "deterministic_compute"

    def _extract_latest_user_request_for_sub2api(
        self,
        messages: list[Any],
    ) -> tuple[str, list[str]]:
        for raw_message in reversed(messages):
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            if str(message.get("role", "")).lower() != "user":
                continue

            content = message.get("content")
            image_urls: list[str] = []
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    part_type = str(item.get("type", "")).lower()
                    if part_type in {"text", "input_text"}:
                        text_val = item.get("text")
                        if text_val is None:
                            text_val = item.get("content") or item.get("value")
                        if text_val:
                            text_parts.append(str(text_val))
                        continue
                    if part_type in {"image_url", "input_image"}:
                        image_url = None
                        if part_type == "image_url":
                            image_payload = item.get("image_url")
                            if isinstance(image_payload, dict):
                                image_url = image_payload.get(
                                    "url"
                                ) or image_payload.get("image_url")
                            elif isinstance(image_payload, str):
                                image_url = image_payload
                        if image_url is None:
                            image_url = item.get("image_url") or item.get("url")
                        if image_url:
                            image_urls.append(str(image_url))
                text = (
                    "".join(text_parts)
                    if text_parts
                    else self._normalize_content(content, strip=False)
                )
            else:
                text = self._normalize_content(content, strip=False)
            if text or image_urls:
                return text, image_urls
        return "", []

    @staticmethod
    def _looks_like_command_style_input(user_text: str | None) -> bool:
        text = str(user_text or "").strip()
        if not text:
            return False
        first_line = text.splitlines()[0].strip()
        return bool(
            re.match(r"^(?:/|\uFF0F)[A-Za-z0-9][A-Za-z0-9_.:-]*(?:\s|$)", first_line)
        )

    @staticmethod
    def _looks_like_complex_reasoning_handoff_query(
        user_text: str,
        image_urls: list[str] | None = None,
    ) -> bool:
        text = str(user_text or "")
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        has_image = bool(image_urls)
        if not text.strip() and not has_image:
            return False
        if ProviderOpenAIOfficial._looks_like_command_style_input(text):
            return False

        signal_score = 0

        complexity_keywords = (
            "证明",
            "推导",
            "详细过程",
            "不要跳步",
            "逐步",
            "验算",
            "完整解答",
            "按卷面",
            "数列",
            "递推",
            "极限",
            "微积分",
            "导数",
            "积分",
            "微分方程",
            "一致收敛",
            "函数项级数",
            "级数",
            "线性代数",
            "特征值",
            "概率",
            "组合",
            "计数",
            "分类讨论",
            "物理",
            "受力",
            "电磁",
            "压轴",
            "竞赛",
            "并发",
            "死锁",
            "线程",
            "调试",
            "规则判断",
            "牌理",
            "向听",
            "听什么",
            "番符",
            "点数",
            "三麻",
            "解题",
            "解这题",
            "解这道题",
            "给出理由",
            "prove",
            "derivation",
            "detailed",
            "step by step",
            "limit",
            "sequence",
            "recurrence",
            "calculus",
            "integral",
            "derivative",
            "uniform convergence",
            "series",
            "eigenvalue",
            "probability",
            "combinatorics",
            "counting",
            "physics",
            "deadlock",
            "thread",
            "concurrency",
            "debug",
        )
        if any(
            keyword and (keyword in text or keyword in lowered)
            for keyword in complexity_keywords
        ):
            signal_score += 2

        math_notation_patterns = (
            r"(?:^|[^a-z])(lim|sup|inf|max|min)(?:[^a-z]|$)",
            r"(?:sin|cos|tan|log|ln|sqrt)(?:\s*\(|\s+[a-z0-9])",
            r"[∑∫√∞→≤≥≈≠]",
            r"[a-z]_[{(]?[a-z0-9+\-*/]+[})]?",
            r"\b[a-z]\s*=\s*[a-z0-9_+\-*/()]+",
            r"\bdy/dx\b|\bd/dx\b|y''|y'",
            r"\b[0-9]{3,}[mpsz]\b",
        )
        notation_hits = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for pattern in math_notation_patterns
        )
        if notation_hits >= 2:
            signal_score += 2
        elif notation_hits == 1:
            signal_score += 1

        section_count = len(
            re.findall(
                r"(?:^|[\n\r])\s*[\(（]?[1-9][\)）.、]",
                text,
            )
        )
        if section_count >= 2:
            signal_score += 2
        if len(re.findall(r"\n\s*-\s*", text)) >= 2:
            signal_score += 1

        if len(compact) >= 80 and any(ch.isdigit() for ch in compact):
            signal_score += 1

        image_reasoning_markers = (
            "完整解答",
            "详细过程",
            "不要跳步",
            "逐步",
            "按卷面",
            "证明",
            "推导",
            "解题",
            "解这题",
            "解这道题",
            "受力",
            "物理",
            "数学",
            "试卷",
            "竞赛",
            "牌理",
            "向听",
            "番符",
            "点数",
            "规则判断",
        )
        casual_image_chat_markers = (
            "今日运势",
            "明日运势",
            "运势",
            "表情包",
            "梗图",
            "头像",
            "壁纸",
            "自拍",
            "颜值",
            "可爱",
            "好看",
            "漂亮",
            "吐槽",
            "看看这张图",
            "看看这个图",
            "看下这张图",
            "看下这个图",
            "这张图怎么样",
            "这个图怎么样",
        )
        if has_image:
            has_strong_image_reasoning = any(
                marker in text for marker in image_reasoning_markers
            )
            has_casual_image_chat = any(
                marker in text for marker in casual_image_chat_markers
            )
            if (
                has_casual_image_chat
                and not has_strong_image_reasoning
                and notation_hits == 0
            ):
                return False
            if has_strong_image_reasoning:
                signal_score += 2
            elif not text.strip():
                return False

        riichi_markers = (
            "向听",
            "听什么",
            "番符",
            "点数",
            "和牌",
            "牌理",
            "三麻",
        )
        if re.search(r"\b[0-9]{3,}[mpsz]\b", lowered) and any(
            marker in text for marker in riichi_markers
        ):
            signal_score += 2

        if any(marker in text for marker in ("死锁", "并发", "线程")):
            signal_score += 2

        return signal_score >= 2

    @staticmethod
    def _looks_like_manual_super_noel_request(text: str | None) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        markers = (
            "另一个你",
            "另一个南条酱",
            "另外的那个南条酱",
            "里南条酱",
            "超级南条酱",
            "更冷静一点的南条酱",
            "更严谨一点的南条酱",
            "叫另一个你",
            "叫另一个南条酱",
            "叫里南条酱",
            "让另一个你来",
            "让另一个南条酱来",
            "让里南条酱来",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _pick_sub2api_super_handoff_tool_name_from_names(
        tool_names: list[str],
    ) -> str | None:
        normalized_names = [
            name for name in tool_names if isinstance(name, str) and name
        ]
        if "transfer_to_super_noel" in normalized_names:
            return "transfer_to_super_noel"
        return None

    def _detect_sub2api_force_super_handoff(
        self,
        messages: list[Any],
        tool_list: list[dict],
        *,
        post_tool_phase: bool,
    ) -> tuple[str | None, str]:
        if post_tool_phase:
            return None, ""

        tool_names: list[str] = []
        for raw_tool in tool_list:
            tool = self._to_plain_dict(raw_tool)
            function_payload = self._to_plain_dict(tool.get("function"))
            name = function_payload.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)

        handoff_tool_name = self._pick_sub2api_super_handoff_tool_name_from_names(
            tool_names
        )
        if not handoff_tool_name:
            return None, ""

        latest_user_text, image_urls = self._extract_latest_user_request_for_sub2api(
            messages
        )
        if self._looks_like_manual_super_noel_request(latest_user_text):
            return handoff_tool_name, "manual_super_noel_request"
        if not self._looks_like_complex_reasoning_handoff_query(
            latest_user_text,
            image_urls,
        ):
            return None, ""

        return handoff_tool_name, "complex_reasoning_handoff"

    def _should_force_sub2api_super_handoff_once(
        self,
        request_messages: list[Any] | None,
        llm_response: LLMResponse,
        tools: ToolSet | None,
    ) -> tuple[bool, str]:
        if not self.sub2api_tool_adapter_enabled:
            return False, ""
        if tools is None or not isinstance(request_messages, list):
            return False, ""
        if self._is_sub2api_post_tool_phase(request_messages):
            return False, ""

        called_tools = getattr(llm_response, "tools_call_name", None)
        if isinstance(called_tools, list) and called_tools:
            return False, ""

        forced_handoff = self._build_sub2api_super_handoff_response(
            request_messages,
            tools,
            retry_reason="complex_reasoning_handoff",
        )
        if forced_handoff is None:
            return False, ""
        return True, forced_handoff.tools_call_name[0]

    def _build_sub2api_super_handoff_response(
        self,
        request_messages: list[Any] | None,
        tools: ToolSet | None,
        *,
        retry_reason: str,
    ) -> LLMResponse | None:
        if tools is None or not isinstance(request_messages, list):
            return None
        if self._is_sub2api_post_tool_phase(request_messages):
            return None
        handoff_tool_name = self._pick_sub2api_super_handoff_tool_name_from_names(
            [tool.name for tool in tools.func_list],
        )
        if not handoff_tool_name:
            return None
        user_text, image_urls = self._extract_latest_user_request_for_sub2api(
            request_messages
        )
        if not self._looks_like_complex_reasoning_handoff_query(user_text, image_urls):
            return None

        handoff_args: dict[str, Any] = {
            "input": user_text,
            "background_task": False,
        }
        if image_urls:
            handoff_args["image_urls"] = image_urls

        llm_response = LLMResponse("tool")
        llm_response.role = "tool"
        llm_response.tools_call_name = [handoff_tool_name]
        llm_response.tools_call_args = [handoff_args]
        llm_response.tools_call_ids = [f"sub2api_forced_handoff_{retry_reason}"]
        llm_response.result_chain = None
        llm_response.completion_text = None
        return llm_response

    def _should_retry_sub2api_python_compute_once(
        self,
        request_messages: list[Any] | None,
        llm_response: LLMResponse,
        tools: ToolSet | None,
    ) -> tuple[bool, str, str]:
        if not self.sub2api_tool_adapter_enabled:
            return False, "", ""
        if not self.sub2api_force_python_for_deterministic:
            return False, "", ""
        if not self.sub2api_force_python_retry_once:
            return False, "", ""
        if tools is None or not isinstance(request_messages, list):
            return False, "", ""

        python_tool_name = self._pick_sub2api_python_tool_name_from_names(
            [tool.name for tool in tools.func_list],
        )
        if not python_tool_name:
            return False, "", ""

        called_tools = getattr(llm_response, "tools_call_name", None)
        if isinstance(called_tools, list) and called_tools:
            return False, "", ""

        if self._is_sub2api_post_tool_phase(request_messages):
            retry_reason = self._get_sub2api_python_failure_reason(request_messages)
            if retry_reason not in {
                "python_resource_memory",
                "python_resource_timeout",
            }:
                return False, "", ""

            completion_text = str(getattr(llm_response, "completion_text", "") or "")
            if completion_text and not self._looks_like_sub2api_retry_chatter(
                completion_text
            ):
                return False, "", ""
            return True, python_tool_name, retry_reason

        latest_user_text = self._extract_latest_user_text_for_sub2api(request_messages)
        if not self._looks_like_deterministic_compute_query(latest_user_text):
            return False, "", ""

        return True, python_tool_name, "missing_required_python_call"

    def _sub2api_messages_have_tool_outputs(self, messages: list[Any]) -> bool:
        marker = "[tool_output call_id="
        for raw_message in messages:
            message = self._to_plain_dict(raw_message)
            if not message:
                continue

            role = str(message.get("role", "")).lower()
            if role == "tool":
                return True
            if role in {"system", "developer"}:
                continue

            content = message.get("content")
            if isinstance(content, str):
                content_text = content
            else:
                content_text = self._normalize_content(content, strip=False)

            if marker in content_text:
                return True
        return False

    def _extract_sub2api_tool_outputs(self, messages: list[Any]) -> list[str]:
        marker = "[tool_output call_id="
        latest_user_index = -1
        parsed_messages: list[tuple[int, str, str]] = []

        for idx, raw_message in enumerate(messages):
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if isinstance(content, str):
                content_text = content
            else:
                content_text = self._normalize_content(content, strip=False)
            parsed_messages.append((idx, role, content_text))
            if role == "user":
                latest_user_index = idx

        start_index = latest_user_index + 1 if latest_user_index >= 0 else 0
        outputs: list[str] = []
        for idx, role, content_text in parsed_messages:
            if idx < start_index:
                continue
            if role == "tool":
                if content_text:
                    outputs.append(content_text)
                continue
            if role in {"system", "developer"}:
                continue
            if marker in content_text:
                _, _, suffix = content_text.partition("]")
                outputs.append(suffix.strip() if suffix else content_text)
        return outputs

    @staticmethod
    def _classify_sub2api_python_failure(tool_output_text: str) -> str:
        if not tool_output_text:
            return ""
        lowered = tool_output_text.lower()
        if "memoryerror" in lowered or "out of memory" in lowered:
            return "python_resource_memory"
        if (
            "timeout" in lowered
            or "timed out" in lowered
            or "time limit" in lowered
            or "cpu limit" in lowered
        ):
            return "python_resource_timeout"
        return ""

    def _get_sub2api_python_failure_reason(self, messages: list[Any]) -> str:
        for output_text in reversed(self._extract_sub2api_tool_outputs(messages)):
            failure_reason = self._classify_sub2api_python_failure(output_text)
            if failure_reason:
                return failure_reason
        return ""

    @staticmethod
    def _looks_like_sub2api_retry_chatter(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        markers = (
            "retry",
            "try again",
            "rerun",
            "re-run",
            "lighter",
            "lightweight",
            "memory",
            "timeout",
            "??",
            "??",
            "??",
            "??",
            "???",
            "??",
            "??",
            "??",
        )
        return any(marker in lowered or marker in text for marker in markers)

    def _is_sub2api_post_tool_phase(self, messages: list[Any]) -> bool:
        marker = "[tool_output call_id="
        latest_user_index = -1
        parsed_messages: list[tuple[int, str, str]] = []

        for idx, raw_message in enumerate(messages):
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if isinstance(content, str):
                content_text = content
            else:
                content_text = self._normalize_content(content, strip=False)
            parsed_messages.append((idx, role, content_text))
            if role == "user":
                latest_user_index = idx

        start_index = latest_user_index + 1 if latest_user_index >= 0 else 0
        for idx, role, content_text in parsed_messages:
            if idx < start_index:
                continue
            if role == "tool":
                return True
            if role in {"system", "developer"}:
                continue
            if marker in content_text:
                return True
        return False

    def _inject_sub2api_tool_adapter_prompt(
        self,
        payloads: dict,
        tool_list: list[dict],
    ) -> None:
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return

        marker = "<sub2api_tool_adapter>"
        for raw_message in messages:
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            content = message.get("content")
            if isinstance(content, str) and marker in content:
                return

        post_tool_phase = self._is_sub2api_post_tool_phase(messages)
        force_python_tool_name, force_reason = (
            self._detect_sub2api_force_python_for_compute(
                messages,
                tool_list,
                post_tool_phase=post_tool_phase,
            )
        )
        force_handoff_tool_name, handoff_reason = (
            self._detect_sub2api_force_super_handoff(
                messages,
                tool_list,
                post_tool_phase=post_tool_phase,
            )
        )
        python_retry_reason = ""
        if post_tool_phase:
            python_retry_reason = self._get_sub2api_python_failure_reason(messages)

        adapter_prompt = self._build_sub2api_tool_adapter_prompt(
            tool_list,
            post_tool_phase=post_tool_phase,
            force_python_tool_name=force_python_tool_name,
            force_handoff_tool_name=force_handoff_tool_name,
            force_reason=handoff_reason or force_reason,
            python_retry_reason=python_retry_reason,
        )
        if not adapter_prompt:
            return

        insert_index = 0
        for idx, raw_message in enumerate(messages):
            message = self._to_plain_dict(raw_message)
            if str(message.get("role", "")) == "system":
                insert_index = idx + 1

        messages.insert(
            insert_index,
            {"role": "system", "content": adapter_prompt},
        )

        tool_names: list[str] = []
        for raw_tool in tool_list:
            tool = self._to_plain_dict(raw_tool)
            function_payload = self._to_plain_dict(tool.get("function"))
            name = function_payload.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)

        logger.info(
            "sub2api adapter injected tool prompt: tool_count=%s phase=%s tools=%s force_python=%s force_handoff=%s",
            len(tool_list),
            "post_tool" if post_tool_phase else "pre_tool",
            ",".join(tool_names[:8]),
            force_python_tool_name or "",
            force_handoff_tool_name or "",
        )

    @staticmethod
    def _build_sub2api_retry_force_tool_prompt(
        python_tool_name: str,
        retry_reason: str = "",
    ) -> str:
        reason_rules = ""
        if retry_reason == "python_resource_memory":
            reason_rules = (
                "The previous Python attempt failed with MemoryError under sandbox limits.\n"
                "Rewrite the code to be dramatically smaller and scoped only to the current input.\n"
            )
        elif retry_reason == "python_resource_timeout":
            reason_rules = (
                "The previous Python attempt failed due to timeout / CPU limits.\n"
                "Rewrite the code to be much smaller and more targeted for only the current input.\n"
            )
        return (
            "<sub2api_tool_adapter_retry>\n"
            "A Python retry is required in this turn.\n"
            f"{reason_rules}"
            "Now output ONLY <astrbot_tool_call>...</astrbot_tool_call> tags.\n"
            f"You MUST call `{python_tool_name}` with executable Python code that computes the exact answer.\n"
            "Avoid exhaustive enumeration, broad precomputation, large recursion trees, and large caches or intermediate containers.\n"
            "Prefer direct formulas, candidate checking, bounded loops, and compact data structures.\n"
            "Do NOT output markdown, explanation, apology, retry plans, or final answer text in this turn.\n"
            "</sub2api_tool_adapter_retry>"
        )

    def _inject_sub2api_retry_prompt(
        self,
        payloads: dict,
        python_tool_name: str,
        retry_reason: str = "",
    ) -> None:
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return

        marker = "<sub2api_tool_adapter_retry>"
        for raw_message in messages:
            message = self._to_plain_dict(raw_message)
            if not message:
                continue
            content = message.get("content")
            if isinstance(content, str) and marker in content:
                return

        retry_prompt = self._build_sub2api_retry_force_tool_prompt(
            python_tool_name,
            retry_reason,
        )

        insert_index = 0
        for idx, raw_message in enumerate(messages):
            message = self._to_plain_dict(raw_message)
            if str(message.get("role", "")) == "system":
                insert_index = idx + 1

        messages.insert(
            insert_index,
            {"role": "system", "content": retry_prompt},
        )
        logger.info(
            "sub2api compute retry prompt injected: tool=%s",
            python_tool_name,
        )

    @staticmethod
    def _strip_markdown_code_fence(raw_text: str) -> str:
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(
                r"^\s*```(?:json|JSON)?\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
            stripped = re.sub(r"\s*```\s*$", "", stripped)
        return stripped.strip()

    def _try_parse_adjacent_json_values(self, raw_payload: str) -> list[Any] | None:
        if not isinstance(raw_payload, str):
            return None

        payload = self._strip_markdown_code_fence(raw_payload)
        if not payload:
            return None

        decoder = json.JSONDecoder()
        values: list[Any] = []
        index = 0
        length = len(payload)

        while index < length:
            while index < length and payload[index] in " \t\r\n,":
                index += 1
            if index >= length:
                break

            try:
                value, next_index = decoder.raw_decode(payload, index)
            except Exception:
                return None

            values.append(value)
            index = next_index

        if len(values) > 1:
            return values
        return None

    def _try_parse_json_like_payload(self, raw_payload: str) -> Any:
        if not isinstance(raw_payload, str):
            return None

        payload = self._strip_markdown_code_fence(raw_payload)
        if not payload:
            return None

        candidates = [payload]
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = payload.find(open_char)
            end = payload.rfind(close_char)
            if start >= 0 and end > start:
                snippet = payload[start : end + 1]
                if snippet not in candidates:
                    candidates.append(snippet)

        for candidate in candidates:
            variants = [candidate]
            sanitized = candidate.replace("\\'", "'")
            if sanitized not in variants:
                variants.append(sanitized)

            for variant in variants:
                try:
                    return json.loads(variant)
                except Exception:
                    pass

            for variant in variants:
                parsed_values = self._try_parse_adjacent_json_values(variant)
                if parsed_values is not None:
                    return parsed_values

        for candidate in candidates:
            for py_like in (candidate, candidate.replace("\\'", "'")):
                try:
                    parsed = ast.literal_eval(py_like)
                except Exception:
                    continue
                if isinstance(parsed, (dict, list)):
                    return parsed

        # Fallback: recover minimal call schema from noisy payload.
        name_match = re.search(r'"name"\s*:\s*"([^"\n\r]+)"', payload)
        args_match = re.search(r'"arguments"\s*:\s*(\{[\s\S]*\})\s*$', payload)
        if name_match:
            reconstructed: dict[str, Any] = {"name": name_match.group(1)}
            if args_match:
                raw_args = args_match.group(1)
                parsed_args = None
                for candidate_args in (raw_args, raw_args.replace("\\'", "'")):
                    try:
                        parsed_args = json.loads(candidate_args)
                        break
                    except Exception:
                        continue
                if parsed_args is None:
                    parsed_args = {"_raw": raw_args}
                reconstructed["arguments"] = parsed_args
            return reconstructed

        return None

    def _iter_sub2api_adapter_payload_segments(
        self,
        text: str,
    ) -> list[tuple[int, int, str]]:
        segments: list[tuple[int, int, str]] = []
        if not text:
            return segments

        tag_patterns = [
            r"<astrbot_tool_call>\s*(.*?)\s*</astrbot_tool_call>",
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            r"<function_call>\s*(.*?)\s*</function_call>",
        ]
        seen_spans: set[tuple[int, int]] = set()
        for tag_pattern in tag_patterns:
            for match in re.finditer(tag_pattern, text, re.IGNORECASE | re.DOTALL):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                payload = match.group(1).strip()
                if payload:
                    segments.append((span[0], span[1], payload))

        dangling_open_tag_pattern = re.compile(
            r"<(?P<tag>astrbot_tool_call|tool_call|function_call)>",
            re.IGNORECASE,
        )
        for match in dangling_open_tag_pattern.finditer(text):
            tag_name = match.group("tag")
            if re.search(
                rf"</{re.escape(tag_name)}>", text[match.end() :], re.IGNORECASE
            ):
                continue

            span = (match.start(), len(text))
            if span in seen_spans:
                continue

            payload = text[match.end() :].strip()
            if not payload:
                continue
            if (
                '"name"' not in payload
                and "'name'" not in payload
                and '"tool_calls"' not in payload
                and "'tool_calls'" not in payload
            ):
                continue

            seen_spans.add(span)
            segments.append((span[0], span[1], payload))

        if segments:
            segments.sort(key=lambda item: item[0])
            return segments

        fence_pattern = re.compile(
            r"```(?:json|JSON)?\s*([\s\S]*?)\s*```",
            re.IGNORECASE,
        )
        for match in fence_pattern.finditer(text):
            payload = match.group(1).strip()
            if not payload:
                continue
            if ('"name"' in payload or "'name'" in payload) and (
                '"arguments"' in payload
                or "'arguments'" in payload
                or '"args"' in payload
                or "'args'" in payload
                or '"tool_calls"' in payload
            ):
                segments.append((match.start(), match.end(), payload))

        if segments:
            segments.sort(key=lambda item: item[0])
            return segments

        stripped = text.strip()
        if (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ) and (
            '"name"' in stripped
            or "'name'" in stripped
            or '"tool_calls"' in stripped
            or "'tool_calls'" in stripped
        ):
            segments.append((0, len(text), stripped))
        return segments

    def _strip_sub2api_tool_tag_blocks(self, text: str) -> str:
        if not text:
            return ""

        cleaned = str(text)
        cleaned = re.sub(
            r"<astrbot_tool_call>\s*[\s\S]*?\s*</astrbot_tool_call>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<tool_call>\s*[\s\S]*?\s*</tool_call>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<function_call>\s*[\s\S]*?\s*</function_call>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        dangling_patterns = [
            r"<astrbot_tool_call>\s*(?=[\s\S]*(?:\"name\"|'name'|\"tool_calls\"|'tool_calls'))[\s\S]*$",
            r"<tool_call>\s*(?=[\s\S]*(?:\"name\"|'name'|\"tool_calls\"|'tool_calls'))[\s\S]*$",
            r"<function_call>\s*(?=[\s\S]*(?:\"name\"|'name'|\"tool_calls\"|'tool_calls'))[\s\S]*$",
        ]
        for dangling_pattern in dangling_patterns:
            cleaned = re.sub(
                dangling_pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(
            r"```(?:json|JSON)?\s*```",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _extract_sub2api_tool_adapter_calls(
        self,
        text: str,
        tools: ToolSet | None,
    ) -> tuple[str, list[dict], list[str], list[str]]:
        if not text or tools is None:
            return text, [], [], []

        segments = self._iter_sub2api_adapter_payload_segments(text)
        if not segments:
            return text, [], [], []

        allowed_names = {tool.name for tool in tools.func_list}
        args_ls: list[dict] = []
        func_name_ls: list[str] = []
        tool_call_ids: list[str] = []
        seen_calls: set[tuple[str, str]] = set()
        parse_failures = 0
        parsed_spans: list[tuple[int, int]] = []

        for start, end, raw_payload in segments:
            parsed_payload = self._try_parse_json_like_payload(raw_payload)
            if parsed_payload is None:
                parse_failures += 1
                continue

            if isinstance(parsed_payload, list):
                candidate_calls = parsed_payload
            elif isinstance(parsed_payload, dict):
                if isinstance(parsed_payload.get("calls"), list):
                    candidate_calls = parsed_payload.get("calls", [])
                elif isinstance(parsed_payload.get("tool_calls"), list):
                    candidate_calls = parsed_payload.get("tool_calls", [])
                else:
                    candidate_calls = [parsed_payload]
            else:
                continue

            parsed_this_segment = False
            for raw_call in candidate_calls:
                call = self._to_plain_data(raw_call)
                if not isinstance(call, dict):
                    continue

                nested_function = self._to_plain_dict(call.get("function"))
                function_name = (
                    call.get("name") or call.get("tool") or call.get("function")
                )
                if isinstance(function_name, dict):
                    function_name = function_name.get("name")
                if not function_name and nested_function:
                    function_name = nested_function.get("name")
                if not isinstance(function_name, str):
                    continue

                function_name = function_name.strip()
                if function_name not in allowed_names:
                    continue

                raw_arguments = call.get("arguments", call.get("args"))
                if raw_arguments is None and nested_function:
                    raw_arguments = nested_function.get("arguments")
                if raw_arguments is None:
                    raw_arguments = {}

                if isinstance(raw_arguments, str):
                    parsed_arguments = self._try_parse_json_like_payload(raw_arguments)
                    if parsed_arguments is None:
                        parsed_arguments = {"_raw": raw_arguments}
                elif isinstance(raw_arguments, dict):
                    parsed_arguments = raw_arguments
                else:
                    parsed_arguments = {"value": raw_arguments}

                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {"value": parsed_arguments}

                call_signature = (
                    function_name,
                    self._safe_json_dump(parsed_arguments) or str(parsed_arguments),
                )
                if call_signature in seen_calls:
                    continue
                seen_calls.add(call_signature)

                call_id = call.get("call_id") or call.get("id")
                if call_id is None:
                    call_id = f"call_adapter_{len(tool_call_ids) + 1}"

                args_ls.append(parsed_arguments)
                func_name_ls.append(function_name)
                tool_call_ids.append(str(call_id))
                parsed_this_segment = True

            if parsed_this_segment:
                parsed_spans.append((start, end))

        # Only strip segments that were parsed into valid tool calls.
        merged_spans: list[list[int]] = []
        for start, end in sorted(parsed_spans, key=lambda item: item[0]):
            if not merged_spans or start > merged_spans[-1][1]:
                merged_spans.append([start, end])
                continue
            merged_spans[-1][1] = max(merged_spans[-1][1], end)

        if merged_spans:
            cleaned_chunks: list[str] = []
            cursor = 0
            for start, end in merged_spans:
                cleaned_chunks.append(text[cursor:start])
                cursor = end
            cleaned_chunks.append(text[cursor:])
            cleaned_text = "".join(cleaned_chunks)
            cleaned_text = re.sub(
                r"```(?:json|JSON)?\s*```",
                "",
                cleaned_text,
                flags=re.IGNORECASE,
            ).strip()
        else:
            cleaned_text = text.strip()

        cleaned_text = self._strip_sub2api_tool_tag_blocks(cleaned_text)

        if func_name_ls:
            logger.info(
                "sub2api adapter parsed tool calls: %s",
                ",".join(func_name_ls),
            )
        elif parse_failures:
            logger.warning(
                "sub2api adapter found tool tags but failed to parse payloads: count=%s",
                parse_failures,
            )
        return cleaned_text, args_ls, func_name_ls, tool_call_ids

    def _convert_message_content_for_responses(
        self,
        content: Any,
        *,
        role: str | None = None,
    ) -> str | list[dict]:
        role = (role or "").lower()
        if isinstance(content, str):
            return content if content else " "
        if isinstance(content, dict):
            return self._convert_message_content_for_responses([content], role=role)
        if isinstance(content, list):
            converted_parts: list[dict] = []
            text_fragments: list[str] = []
            for raw_part in content:
                part = self._to_plain_data(raw_part)
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"text", "input_text", "output_text"}:
                    text = part.get("text", "")
                    text = str(text) if text is not None else ""
                    if self.sub2api_mode:
                        text_fragments.append(text)
                    else:
                        converted_parts.append(
                            {"type": "input_text", "text": text or " "}
                        )
                    continue
                if part_type == "think":
                    think_text = part.get("think")
                    if think_text is not None:
                        if self.sub2api_mode:
                            text_fragments.append(str(think_text))
                        else:
                            converted_parts.append(
                                {"type": "input_text", "text": str(think_text)}
                            )
                    continue
                if part_type in {"image_url", "input_image"}:
                    image_url = None
                    detail = "auto"
                    if part_type == "image_url":
                        image_payload = part.get("image_url")
                        if isinstance(image_payload, dict):
                            image_url = image_payload.get("url") or image_payload.get(
                                "image_url"
                            )
                            if image_payload.get("detail"):
                                detail = str(image_payload.get("detail"))
                        elif isinstance(image_payload, str):
                            image_url = image_payload
                    else:
                        image_url = part.get("image_url") or part.get("url")
                        if part.get("detail"):
                            detail = str(part.get("detail"))
                    if image_url:
                        image_item = {
                            "type": "input_image",
                            "image_url": str(image_url),
                            "detail": detail,
                        }
                        if self.sub2api_mode:
                            # sub2api upstream only accepts multimodal blocks on user role.
                            if role == "user":
                                converted_parts.append(image_item)
                        else:
                            converted_parts.append(image_item)

            if self.sub2api_mode:
                plain_text = "".join(text_fragments).strip()
                if converted_parts:
                    mixed_content: list[dict] = [
                        {"type": "input_text", "text": plain_text or " "}
                    ]
                    mixed_content.extend(converted_parts)
                    return mixed_content
                return plain_text or " "

            if converted_parts:
                return converted_parts
            return " "
        return str(content) if content is not None else " "

    def _convert_messages_to_responses_input(self, messages: list[dict]) -> list[dict]:
        valid_call_ids: set[str] = set()
        for raw_message in messages:
            message = self._to_plain_data(raw_message)
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")) != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for raw_tool_call in tool_calls:
                tool_call = self._to_plain_data(raw_tool_call)
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                if call_id:
                    valid_call_ids.add(str(call_id))

        response_input: list[dict] = []
        for raw_message in messages:
            message = self._to_plain_data(raw_message)
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            content = message.get("content")

            if role == "tool":
                call_id = message.get("tool_call_id") or message.get("id")
                if not call_id:
                    continue
                call_id = str(call_id)
                output = content
                if isinstance(output, (dict, list)):
                    output = self._safe_json_dump(output) or str(output)
                elif output is None:
                    output = ""

                if self.sub2api_mode:
                    if self.sub2api_tool_adapter_enabled:
                        response_input.append(
                            {
                                "role": "user",
                                "content": f"[tool_output call_id={call_id}] {str(output)}",
                            }
                        )
                    continue

                if call_id not in valid_call_ids:
                    logger.debug(
                        "Skip orphan tool output without matching function_call: %s",
                        call_id,
                    )
                    continue
                response_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(output),
                    }
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                if content not in (None, "", []):
                    response_input.append(
                        {
                            "role": "assistant",
                            "content": self._convert_message_content_for_responses(
                                content,
                                role=role,
                            ),
                        }
                    )
                if self.sub2api_mode:
                    continue
                for raw_tool_call in message.get("tool_calls", []):
                    tool_call = self._to_plain_data(raw_tool_call)
                    if not isinstance(tool_call, dict):
                        continue
                    if tool_call.get("type") != "function":
                        continue
                    function_payload = tool_call.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    function_name = function_payload.get("name")
                    if not function_name:
                        continue
                    arguments = function_payload.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = self._safe_json_dump(arguments) or "{}"
                    response_input.append(
                        {
                            "type": "function_call",
                            "call_id": str(
                                tool_call.get("id")
                                or f"call_{len(response_input) + 1}",
                            ),
                            "name": str(function_name),
                            "arguments": arguments,
                        }
                    )
                continue

            if role not in {"user", "assistant", "system", "developer"}:
                continue

            response_input.append(
                {
                    "role": role,
                    "content": self._convert_message_content_for_responses(
                        content,
                        role=role,
                    ),
                }
            )
        return response_input

    def _convert_chat_payload_to_responses(self, payloads: dict) -> dict:
        converted = dict(payloads)
        messages = converted.pop("messages", None)
        if isinstance(messages, list):
            converted["input"] = self._convert_messages_to_responses_input(messages)
        if "max_tokens" in converted and "max_output_tokens" not in converted:
            converted["max_output_tokens"] = converted.pop("max_tokens")
        tools = converted.get("tools")
        if self.sub2api_mode:
            converted.pop("tools", None)
        elif isinstance(tools, list):
            converted["tools"] = self._convert_tools_for_responses(tools)
        return converted

    async def _query(self, payloads: dict, tools: ToolSet | None) -> LLMResponse:
        payloads = dict(payloads)
        adapter_tool_list: list[dict] = []
        if tools:
            model = payloads.get("model", "").lower()
            omit_empty_param_field = "gemini" in model
            tool_list = tools.get_func_desc_openai_style(
                omit_empty_parameter_field=omit_empty_param_field,
            )
            if tool_list:
                if self.sub2api_tool_adapter_enabled:
                    adapter_tool_list = tool_list
                else:
                    payloads["tools"] = tool_list

        if self.sub2api_tool_adapter_enabled and adapter_tool_list:
            self._inject_sub2api_tool_adapter_prompt(payloads, adapter_tool_list)

        if self.use_responses_api:
            if not self.responses_default_params:
                raise Exception("Responses API 未初始化成功。")
            responses_payload = self._convert_chat_payload_to_responses(payloads)
            request_payload, extra_body = self._split_payload_and_extra_body(
                responses_payload,
                self.responses_default_params,
            )
            completion = await self.client.responses.create(
                **request_payload,
                stream=False,
                extra_body=extra_body,
            )
            logger.debug(f"responses completion: {completion}")
            llm_response = await self._parse_openai_response_completion(
                completion,
                tools,
                request_messages=payloads.get("messages"),
            )
            should_retry, python_tool_name, retry_reason = (
                self._should_retry_sub2api_python_compute_once(
                    payloads.get("messages"),
                    llm_response,
                    tools,
                )
            )
            if should_retry:
                logger.info(
                    "sub2api compute retry triggered: tool=%s reason=%s",
                    python_tool_name,
                    retry_reason,
                )
                retry_payloads = dict(payloads)
                retry_messages = retry_payloads.get("messages")
                if isinstance(retry_messages, list):
                    retry_payloads["messages"] = list(retry_messages)
                self._inject_sub2api_retry_prompt(
                    retry_payloads,
                    python_tool_name,
                    retry_reason,
                )
                retry_responses_payload = self._convert_chat_payload_to_responses(
                    retry_payloads,
                )
                retry_request_payload, retry_extra_body = (
                    self._split_payload_and_extra_body(
                        retry_responses_payload,
                        self.responses_default_params,
                    )
                )
                retry_completion = await self.client.responses.create(
                    **retry_request_payload,
                    stream=False,
                    extra_body=retry_extra_body,
                )
                logger.debug(f"sub2api retry completion: {retry_completion}")
                retry_llm_response = await self._parse_openai_response_completion(
                    retry_completion,
                    tools,
                    request_messages=retry_payloads.get("messages"),
                )
                retry_called_tools = getattr(
                    retry_llm_response, "tools_call_name", None
                )
                if isinstance(retry_called_tools, list) and retry_called_tools:
                    logger.info(
                        "sub2api compute retry succeeded with tool call: tool=%s reason=%s",
                        python_tool_name,
                        retry_reason,
                    )
                    return retry_llm_response
                logger.warning(
                    "sub2api compute retry returned without tool call: tool=%s reason=%s",
                    python_tool_name,
                    retry_reason,
                )
                forced_handoff_response = self._build_sub2api_super_handoff_response(
                    retry_payloads.get("messages"),
                    tools,
                    retry_reason=retry_reason,
                )
                if forced_handoff_response is not None:
                    logger.info(
                        "sub2api compute retry escalated to super handoff: tool=%s reason=%s",
                        forced_handoff_response.tools_call_name[0],
                        retry_reason,
                    )
                    return forced_handoff_response
            return llm_response

        request_payload, extra_body = self._split_payload_and_extra_body(
            payloads,
            self.chat_default_params,
        )

        completion = await self.client.chat.completions.create(
            **request_payload,
            stream=False,
            extra_body=extra_body,
        )

        if not isinstance(completion, ChatCompletion):
            raise Exception(
                f"API 返回的 completion 类型错误：{type(completion)}: {completion}。",
            )

        logger.debug(f"completion: {completion}")

        llm_response = await self._parse_openai_completion(completion, tools)

        return llm_response

    async def _query_stream(
        self,
        payloads: dict,
        tools: ToolSet | None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """流式查询API，逐步返回结果"""
        payloads = dict(payloads)
        if self.use_responses_api and not self.responses_stream_supported:
            yield await self._query(payloads, tools)
            return

        adapter_tool_list: list[dict] = []
        if tools:
            model = payloads.get("model", "").lower()
            omit_empty_param_field = "gemini" in model
            tool_list = tools.get_func_desc_openai_style(
                omit_empty_parameter_field=omit_empty_param_field,
            )
            if tool_list:
                if self.sub2api_tool_adapter_enabled:
                    adapter_tool_list = tool_list
                else:
                    payloads["tools"] = tool_list

        if self.sub2api_tool_adapter_enabled and adapter_tool_list:
            self._inject_sub2api_tool_adapter_prompt(payloads, adapter_tool_list)

        if self.use_responses_api:
            if not self.responses_default_params:
                raise Exception("Responses API 未初始化成功。")
            responses_payload = self._convert_chat_payload_to_responses(payloads)
            request_payload, extra_body = self._split_payload_and_extra_body(
                responses_payload,
                self.responses_default_params,
            )
            try:
                stream = await self.client.responses.create(
                    **request_payload,
                    stream=True,
                    extra_body=extra_body,
                )
            except Exception as stream_error:
                if self._is_upstream_server_error(stream_error):
                    self.responses_stream_supported = False
                    logger.warning(
                        "Responses 流式请求被上游拒绝，已切换为非流式兼容模式。",
                    )
                    yield await self._query(payloads, tools)
                    return
                raise

            llm_response = LLMResponse("assistant", is_chunk=True)
            final_response_obj: Any = None

            async for event in stream:
                event_dict = self._to_plain_dict(event)
                event_type = str(
                    event_dict.get("type") or getattr(event, "type", ""),
                )
                should_yield = False

                if event_type == "response.output_text.delta":
                    delta = event_dict.get("delta")
                    if delta:
                        llm_response.result_chain = MessageChain(
                            chain=[Comp.Plain(str(delta))],
                        )
                        should_yield = True
                elif event_type in {
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                }:
                    delta = event_dict.get("delta")
                    if delta:
                        llm_response.reasoning_content = str(delta)
                        should_yield = True
                elif event_type == "response.completed":
                    final_response_obj = event_dict.get("response")
                    if final_response_obj is None:
                        final_response_obj = self._to_plain_data(
                            getattr(event, "response", None),
                        )
                    if isinstance(final_response_obj, dict):
                        llm_response.id = final_response_obj.get("id", llm_response.id)
                elif event_type == "response.error":
                    error_obj = event_dict.get("error")
                    raise Exception(f"Responses API 流式调用失败: {error_obj}")

                if should_yield:
                    yield llm_response

            if final_response_obj is not None:
                yield await self._parse_openai_response_completion(
                    final_response_obj,
                    tools,
                    request_messages=payloads.get("messages"),
                )
                return

            if llm_response.result_chain is not None or llm_response.reasoning_content:
                final_fallback = LLMResponse("assistant")
                final_fallback.id = llm_response.id
                final_fallback.result_chain = llm_response.result_chain
                final_fallback.reasoning_content = llm_response.reasoning_content
                yield final_fallback
                return

            raise Exception("Responses API 流式调用未返回可解析结果。")

        request_payload, extra_body = self._split_payload_and_extra_body(
            payloads,
            self.chat_default_params,
        )

        stream = await self.client.chat.completions.create(
            **request_payload,
            stream=True,
            extra_body=extra_body,
        )

        llm_response = LLMResponse("assistant", is_chunk=True)

        state = ChatCompletionStreamState()

        async for chunk in stream:
            try:
                state.handle_chunk(chunk)
            except Exception as e:
                logger.warning("Saving chunk state error: " + str(e))
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            # logger.debug(f"chunk delta: {delta}")
            # handle the content delta
            reasoning = self._extract_reasoning_content(chunk)
            _y = False
            llm_response.id = chunk.id
            if reasoning:
                llm_response.reasoning_content = reasoning
                _y = True
            if delta.content:
                # Don't strip streaming chunks to preserve spaces between words
                completion_text = self._normalize_content(delta.content, strip=False)
                llm_response.result_chain = MessageChain(
                    chain=[Comp.Plain(completion_text)],
                )
                _y = True
            if chunk.usage:
                llm_response.usage = self._extract_usage(chunk.usage)
            if _y:
                yield llm_response

        final_completion = state.get_final_completion()
        llm_response = await self._parse_openai_completion(final_completion, tools)

        yield llm_response

    def _extract_reasoning_content(
        self,
        completion: ChatCompletion | ChatCompletionChunk,
    ) -> str:
        """Extract reasoning content from OpenAI ChatCompletion if available."""
        reasoning_text = ""
        if len(completion.choices) == 0:
            return reasoning_text
        if isinstance(completion, ChatCompletion):
            choice = completion.choices[0]
            reasoning_attr = getattr(choice.message, self.reasoning_key, None)
            if reasoning_attr:
                reasoning_text = str(reasoning_attr)
        elif isinstance(completion, ChatCompletionChunk):
            delta = completion.choices[0].delta
            reasoning_attr = getattr(delta, self.reasoning_key, None)
            if reasoning_attr:
                reasoning_text = str(reasoning_attr)
        return reasoning_text

    def _extract_usage(self, usage: CompletionUsage) -> TokenUsage:
        ptd = usage.prompt_tokens_details
        cached = ptd.cached_tokens if ptd and ptd.cached_tokens else 0
        prompt_tokens = 0 if usage.prompt_tokens is None else usage.prompt_tokens
        completion_tokens = (
            0 if usage.completion_tokens is None else usage.completion_tokens
        )
        return TokenUsage(
            input_other=prompt_tokens - cached,
            input_cached=cached,
            output=completion_tokens,
        )

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _extract_usage_from_responses(self, usage_data: Any) -> TokenUsage | None:
        usage_dict = self._to_plain_dict(usage_data)
        if not usage_dict:
            return None
        input_tokens = self._to_int(usage_dict.get("input_tokens"), 0)
        output_tokens = self._to_int(usage_dict.get("output_tokens"), 0)
        input_details = usage_dict.get("input_tokens_details")
        if not isinstance(input_details, dict):
            input_details = self._to_plain_dict(input_details)
        cached_tokens = self._to_int(input_details.get("cached_tokens"), 0)
        return TokenUsage(
            input_other=max(input_tokens - cached_tokens, 0),
            input_cached=cached_tokens,
            output=output_tokens,
        )

    @classmethod
    def _try_parse_responses_sse_transcript(
        cls,
        raw_text: Any,
    ) -> tuple[dict | None, dict | None]:
        if not isinstance(raw_text, str):
            return None, None

        text = raw_text.strip()
        if not text or "event:" not in text or "data:" not in text:
            return None, None

        final_response: dict | None = None
        latest_error: dict | None = None
        output_text_done: str | None = None

        for block in re.split(r"\r?\n\r?\n+", text):
            block = block.strip()
            if not block:
                continue

            event_name = ""
            data_lines: list[str] = []
            for line in block.splitlines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())

            if not data_lines:
                continue

            raw_data = "\n".join(data_lines).strip()
            if not raw_data:
                continue

            try:
                payload = json.loads(raw_data)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            payload_type = str(payload.get("type") or event_name)
            if payload_type == "response.completed":
                response_obj = payload.get("response")
                if isinstance(response_obj, dict):
                    final_response = response_obj
                continue

            if payload_type in {"response.failed", "error", "response.error"}:
                error_obj = payload.get("error")
                if not isinstance(error_obj, dict):
                    response_obj = payload.get("response")
                    if isinstance(response_obj, dict):
                        maybe_error = response_obj.get("error")
                        if isinstance(maybe_error, dict):
                            error_obj = maybe_error
                if isinstance(error_obj, dict):
                    latest_error = error_obj
                continue

            if payload_type == "response.output_text.done":
                text_value = payload.get("text")
                if isinstance(text_value, str):
                    output_text_done = text_value

        if final_response is not None:
            return final_response, latest_error

        if output_text_done is not None:
            return (
                {
                    "id": "responses_sse_transcript",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": output_text_done,
                                }
                            ],
                        }
                    ],
                },
                latest_error,
            )

        return None, latest_error

    async def _parse_openai_response_completion(
        self,
        completion: Any,
        tools: ToolSet | None,
        request_messages: list[Any] | None = None,
    ) -> LLMResponse:
        llm_response = LLMResponse("assistant")
        completion_dict = self._to_plain_dict(completion)
        if not completion_dict:
            transcript_completion, transcript_error = (
                self._try_parse_responses_sse_transcript(
                    completion,
                )
            )
            if transcript_completion:
                completion_dict = transcript_completion
            elif transcript_error:
                raise Exception(f"Responses API ???????{transcript_error}")
            else:
                raise Exception(f"API ??? response ?????{completion}")

        error_obj = completion_dict.get("error")
        if error_obj:
            raise Exception(f"Responses API 返回错误：{error_obj}")

        output_items = completion_dict.get("output")
        if not isinstance(output_items, list):
            output_items = []

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        args_ls: list[dict] = []
        func_name_ls: list[str] = []
        tool_call_ids: list[str] = []

        for raw_item in output_items:
            item = self._to_plain_dict(raw_item)
            if not item:
                continue
            item_type = item.get("type")

            if item_type == "message":
                content_list = item.get("content")
                if not isinstance(content_list, list):
                    continue
                for raw_part in content_list:
                    part = self._to_plain_dict(raw_part)
                    if not part:
                        continue
                    part_type = part.get("type")
                    if part_type in {"output_text", "text", "input_text"}:
                        text = part.get("text")
                        if text is not None:
                            text_parts.append(str(text))
                continue

            if item_type == "reasoning":
                summary_list = item.get("summary")
                if isinstance(summary_list, list):
                    for raw_summary in summary_list:
                        summary = self._to_plain_dict(raw_summary)
                        text = summary.get("text")
                        if text is not None:
                            reasoning_parts.append(str(text))
                content_list = item.get("content")
                if isinstance(content_list, list):
                    for raw_content in content_list:
                        content = self._to_plain_dict(raw_content)
                        text = content.get("text")
                        if text is not None:
                            reasoning_parts.append(str(text))
                continue

            if item_type == "function_call" and tools is not None:
                function_name = item.get("name")
                if not isinstance(function_name, str):
                    continue
                if not any(tool.name == function_name for tool in tools.func_list):
                    continue

                raw_arguments = item.get("arguments", "{}")
                if isinstance(raw_arguments, str):
                    try:
                        parsed_arguments = json.loads(raw_arguments)
                    except Exception:
                        parsed_arguments = {"_raw": raw_arguments}
                elif isinstance(raw_arguments, dict):
                    parsed_arguments = raw_arguments
                else:
                    parsed_arguments = {"value": raw_arguments}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {"value": parsed_arguments}

                tool_call_id = item.get("call_id") or item.get("id")
                if tool_call_id is None:
                    tool_call_id = f"call_{len(tool_call_ids) + 1}"

                args_ls.append(parsed_arguments)
                func_name_ls.append(function_name)
                tool_call_ids.append(str(tool_call_id))

        completion_text = "".join(text_parts)
        if not completion_text:
            raw_output_text = completion_dict.get("output_text")
            if raw_output_text is None:
                raw_output_text = getattr(completion, "output_text", None)
            if raw_output_text is not None:
                completion_text = self._normalize_content(raw_output_text)

        if completion_text:
            completion_text = self._normalize_content(completion_text)
            reasoning_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
            matches = reasoning_pattern.findall(completion_text)
            if matches and not reasoning_parts:
                reasoning_parts.extend([match.strip() for match in matches])
            completion_text = reasoning_pattern.sub("", completion_text).strip()
            completion_text = re.sub(r"</think>\s*$", "", completion_text).strip()
            if self.sub2api_tool_adapter_enabled:
                if tools is not None:
                    (
                        completion_text,
                        adapter_args,
                        adapter_func_names,
                        adapter_tool_call_ids,
                    ) = self._extract_sub2api_tool_adapter_calls(completion_text, tools)
                    if adapter_func_names:
                        args_ls.extend(adapter_args)
                        func_name_ls.extend(adapter_func_names)
                        tool_call_ids.extend(adapter_tool_call_ids)
                completion_text = self._strip_sub2api_tool_tag_blocks(completion_text)
            completion_text = self._plainify_common_latex_markup(completion_text)
            completion_text = self._trim_optional_followup_offer(completion_text)
            if completion_text:
                llm_response.result_chain = MessageChain().message(completion_text)

        if (
            self.sub2api_tool_adapter_enabled
            and self.sub2api_force_python_for_deterministic
            and tools is not None
            and not func_name_ls
            and isinstance(request_messages, list)
            and not self._is_sub2api_post_tool_phase(request_messages)
        ):
            latest_user_text = self._extract_latest_user_text_for_sub2api(
                request_messages,
            )
            if self._looks_like_deterministic_compute_query(latest_user_text):
                python_tool_name = self._pick_sub2api_python_tool_name_from_names(
                    [tool.name for tool in tools.func_list],
                )
                if python_tool_name:
                    logger.info(
                        "sub2api compute guard: deterministic query detected without tool call; tool=%s",
                        python_tool_name,
                    )

        if reasoning_parts:
            llm_response.reasoning_content = "\n".join(
                [part for part in reasoning_parts if part],
            ).strip()

        forced_handoff_response = None
        if (
            self.sub2api_tool_adapter_enabled
            and not func_name_ls
            and isinstance(request_messages, list)
        ):
            forced_handoff_response = self._build_sub2api_super_handoff_response(
                request_messages,
                tools,
                retry_reason="pre_answer_structured_reasoning",
            )
            if forced_handoff_response is not None:
                logger.info(
                    "sub2api hard bias escalated structured query to super handoff: tool=%s",
                    forced_handoff_response.tools_call_name[0],
                )
                return forced_handoff_response

        if func_name_ls:
            llm_response.role = "tool"
            llm_response.tools_call_args = args_ls
            llm_response.tools_call_name = func_name_ls
            llm_response.tools_call_ids = tool_call_ids

        if llm_response.tools_call_args:
            # Prevent pre-tool assistant text from being emitted as a user-visible reply.
            llm_response.result_chain = None
            llm_response.completion_text = None

        if llm_response.completion_text is None and not llm_response.tools_call_args:
            logger.error(f"API 返回的 response 无法解析：{completion_dict}")
            raise Exception(f"API 返回的 response 无法解析：{completion_dict}")

        llm_response.raw_completion = completion
        llm_response.id = completion_dict.get("id")

        usage = self._extract_usage_from_responses(completion_dict.get("usage"))
        if usage:
            llm_response.usage = usage

        return llm_response

    @staticmethod
    def _normalize_content(raw_content: Any, strip: bool = True) -> str:
        """Normalize content from various formats to plain string.

        Some LLM providers return content as list[dict] format
        like [{'type': 'text', 'text': '...'}] instead of
        plain string. This method handles both formats.

        Args:
            raw_content: The raw content from LLM response, can be str, list, dict, or other.
            strip: Whether to strip whitespace from the result. Set to False for
                   streaming chunks to preserve spaces between words.

        Returns:
            Normalized plain text string.
        """
        # Handle dict format (e.g., {"type": "text", "text": "..."})
        if isinstance(raw_content, dict):
            if "text" in raw_content:
                text_val = raw_content.get("text", "")
                return str(text_val) if text_val is not None else ""
            # For other dict formats, return empty string and log
            logger.warning(f"Unexpected dict format content: {raw_content}")
            return ""

        if isinstance(raw_content, list):
            # Check if this looks like OpenAI content-part format
            # Only process if at least one item has {'type': 'text', 'text': ...} structure
            has_content_part = any(
                isinstance(part, dict) and part.get("type") == "text"
                for part in raw_content
            )
            if has_content_part:
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_val = part.get("text", "")
                        # Coerce to str in case text is null or non-string
                        text_parts.append(str(text_val) if text_val is not None else "")
                return "".join(text_parts)
            # Not content-part format, return string representation
            return str(raw_content)

        if isinstance(raw_content, str):
            content = raw_content.strip() if strip else raw_content
            # Check if the string is a JSON-encoded list (e.g., "[{'type': 'text', ...}]")
            # This can happen when streaming concatenates content that was originally list format
            # Only check if it looks like a complete JSON array (requires strip for check)
            check_content = raw_content.strip()
            if (
                check_content.startswith("[")
                and check_content.endswith("]")
                and len(check_content) < 8192
            ):
                try:
                    # First try standard JSON parsing
                    parsed = json.loads(check_content)
                except json.JSONDecodeError:
                    # If that fails, try parsing as Python literal (handles single quotes)
                    # This is safer than blind replace("'", '"') which corrupts apostrophes
                    try:
                        import ast

                        parsed = ast.literal_eval(check_content)
                    except (ValueError, SyntaxError):
                        parsed = None

                if isinstance(parsed, list):
                    # Only convert if it matches OpenAI content-part schema
                    # i.e., at least one item has {'type': 'text', 'text': ...}
                    has_content_part = any(
                        isinstance(part, dict) and part.get("type") == "text"
                        for part in parsed
                    )
                    if has_content_part:
                        text_parts = []
                        for part in parsed:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_val = part.get("text", "")
                                # Coerce to str in case text is null or non-string
                                text_parts.append(
                                    str(text_val) if text_val is not None else ""
                                )
                        if text_parts:
                            return "".join(text_parts)
            return content

        # Fallback for other types (int, float, etc.)
        return str(raw_content) if raw_content is not None else ""

    @staticmethod
    def _plainify_common_latex_markup(text: str | None) -> str:
        body = str(text or "")
        if not body or "\\" not in body:
            return body

        cleaned = body.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"\\\[\s*", "", cleaned)
        cleaned = re.sub(r"\s*\\\]", "", cleaned)
        cleaned = re.sub(r"\\\(\s*", "", cleaned)
        cleaned = re.sub(r"\s*\\\)", "", cleaned)

        unwrap_patterns = (
            r"\\text\s*\{([^{}]*)\}",
            r"\\mathrm\s*\{([^{}]*)\}",
            r"\\operatorname\s*\{([^{}]*)\}",
            r"\\mathbf\s*\{([^{}]*)\}",
            r"\\mathit\s*\{([^{}]*)\}",
            r"\\mathtt\s*\{([^{}]*)\}",
            r"\\mathbb\s*\{([^{}]*)\}",
            r"\\mathcal\s*\{([^{}]*)\}",
        )
        for _ in range(6):
            previous = cleaned
            for pattern in unwrap_patterns:
                cleaned = re.sub(pattern, r"\1", cleaned)
            if cleaned == previous:
                break

        cleaned = re.sub(r"\\n?Longleftrightarrow", "<=>", cleaned)
        cleaned = re.sub(r"\\n?Leftrightarrow", "<=>", cleaned)
        cleaned = re.sub(r"\\n?Longrightarrow", "=>", cleaned)
        cleaned = re.sub(r"\\n?Rightarrow", "=>", cleaned)

        replacements = (
            ("\\implies", "=>"),
            ("\\rightarrow", "->"),
            ("\\to", "->"),
            ("\\mapsto", "->"),
            ("\\cdot", "*"),
            ("\\times", "x"),
            ("\\leq", "<="),
            ("\\le", "<="),
            ("\\geq", ">="),
            ("\\ge", ">="),
            ("\\neq", "!="),
            ("\\approx", "~="),
            ("\\infty", "inf"),
            ("\\pm", "+/-"),
            ("\\sqrt", "sqrt"),
        )
        for src, dst in replacements:
            cleaned = cleaned.replace(src, dst)

        cleaned = re.sub(r"\\([{}%#$&_])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _looks_like_optional_followup_start(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        starters = (
            "\u5982\u679c\u4f60\u613f\u610f",
            "\u5982\u679c\u4f60\u60f3",
            "\u5982\u679c\u4f60\u9700\u8981",
            "\u5982\u679c\u4f60\u8fd8\u60f3",
            "\u5982\u679c\u4f60\u8981\u662f\u60f3",
            "\u5982\u679c\u4f60\u5176\u5b9e\u662f\u60f3",
            "\u8981\u4e0d\u8981",
            "\u4f60\u8981\u54ea\u4e2a",
            "\u4f60\u60f3\u8981\u54ea\u4e2a",
            "\u4f60\u9009\u54ea\u4e2a",
        )
        if any(stripped.startswith(prefix) for prefix in starters):
            return True
        if stripped.startswith("\u6211\u53ef\u4ee5") and any(
            token in stripped
            for token in ("\u7ee7\u7eed", "\u518d", "\u5e2e\u4f60", "\u7ed9\u4f60")
        ):
            return True
        if stripped.startswith("\u6211\u8fd8\u80fd") and any(
            token in stripped
            for token in ("\u7ee7\u7eed", "\u5e2e\u4f60", "\u7ed9\u4f60")
        ):
            return True
        return False

    @classmethod
    def _trim_optional_followup_offer(cls, text: str | None) -> str:
        body = str(text or "").strip()
        if not body:
            return ""
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized.split("\n")]
        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if len(non_empty_indices) < 2:
            return body

        for idx in non_empty_indices[1:]:
            candidate = lines[idx].strip()
            if not cls._looks_like_optional_followup_start(candidate):
                continue
            head = "\n".join(lines[:idx]).rstrip()
            tail_lines = [line.strip() for line in lines[idx:] if line.strip()]
            tail_text = "\n".join(tail_lines)
            if len(tail_text) > 320:
                continue
            if not any(
                token in tail_text
                for token in (
                    "\u6211\u53ef\u4ee5",
                    "\u6211\u8fd8\u80fd",
                    "\u6211\u4e5f\u53ef\u4ee5",
                    "\u7ee7\u7eed\u5e2e\u4f60",
                    "\u7ee7\u7eed\u7ed9\u4f60",
                    "\u6b63\u5f0f\u8bc1\u660e",
                    "\u8bc1\u660e\u9898\u683c\u5f0f",
                    "\u4e25\u683c\u7b97",
                    "\u4e25\u683c\u8bc1\u660e",
                    "\u4f60\u8981\u54ea\u4e2a",
                    "\u4f60\u60f3\u8981\u54ea\u4e2a",
                    "\u4f60\u9009\u54ea\u4e2a",
                    "1.",
                    "2.",
                    "1\u3001",
                    "2\u3001",
                )
            ):
                continue
            return head
        return body

    async def _parse_openai_completion(
        self, completion: ChatCompletion, tools: ToolSet | None
    ) -> LLMResponse:
        """Parse OpenAI ChatCompletion into LLMResponse"""
        llm_response = LLMResponse("assistant")

        if len(completion.choices) == 0:
            raise Exception("API 返回的 completion 为空。")
        choice = completion.choices[0]

        # parse the text completion
        if choice.message.content is not None:
            completion_text = self._normalize_content(choice.message.content)
            # specially, some providers may set <think> tags around reasoning content in the completion text,
            # we use regex to remove them, and store then in reasoning_content field
            reasoning_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
            matches = reasoning_pattern.findall(completion_text)
            if matches:
                llm_response.reasoning_content = "\n".join(
                    [match.strip() for match in matches],
                )
                completion_text = reasoning_pattern.sub("", completion_text).strip()
            # Also clean up orphan </think> tags that may leak from some models
            completion_text = re.sub(r"</think>\s*$", "", completion_text).strip()

            if self.sub2api_tool_adapter_enabled:
                if tools is not None:
                    (
                        completion_text,
                        adapter_args,
                        adapter_func_names,
                        adapter_tool_call_ids,
                    ) = self._extract_sub2api_tool_adapter_calls(completion_text, tools)
                    if adapter_func_names:
                        llm_response.role = "tool"
                        llm_response.tools_call_args = adapter_args
                        llm_response.tools_call_name = adapter_func_names
                        llm_response.tools_call_ids = adapter_tool_call_ids
                completion_text = self._strip_sub2api_tool_tag_blocks(completion_text)

            completion_text = self._plainify_common_latex_markup(completion_text)
            completion_text = self._trim_optional_followup_offer(completion_text)
            if completion_text:
                llm_response.result_chain = MessageChain().message(completion_text)

        # parse the reasoning content if any
        # the priority is higher than the <think> tag extraction
        llm_response.reasoning_content = self._extract_reasoning_content(completion)

        # parse tool calls if any
        if choice.message.tool_calls and tools is not None:
            args_ls = list(llm_response.tools_call_args or [])
            func_name_ls = list(llm_response.tools_call_name or [])
            tool_call_ids = list(llm_response.tools_call_ids or [])
            tool_call_extra_content_dict = dict(
                llm_response.tools_call_extra_content or {}
            )
            for tool_call in choice.message.tool_calls:
                if isinstance(tool_call, str):
                    # workaround for #1359
                    tool_call = json.loads(tool_call)
                if tools is None:
                    # 工具集未提供
                    # Should be unreachable
                    raise Exception("工具集未提供")
                for tool in tools.func_list:
                    if (
                        tool_call.type == "function"
                        and tool.name == tool_call.function.name
                    ):
                        # workaround for #1454
                        if isinstance(tool_call.function.arguments, str):
                            args = json.loads(tool_call.function.arguments)
                        else:
                            args = tool_call.function.arguments
                        args_ls.append(args)
                        func_name_ls.append(tool_call.function.name)
                        tool_call_ids.append(tool_call.id)

                        # gemini-2.5 / gemini-3 series extra_content handling
                        extra_content = getattr(tool_call, "extra_content", None)
                        if extra_content is not None:
                            tool_call_extra_content_dict[tool_call.id] = extra_content
            llm_response.role = "tool"
            llm_response.tools_call_args = args_ls
            llm_response.tools_call_name = func_name_ls
            llm_response.tools_call_ids = tool_call_ids
            llm_response.tools_call_extra_content = tool_call_extra_content_dict
        # specially handle finish reason
        if choice.finish_reason == "content_filter":
            raise Exception(
                "API 返回的 completion 由于内容安全过滤被拒绝(非 AstrBot)。",
            )

        if llm_response.tools_call_args:
            # Prevent pre-tool assistant text from being emitted as a user-visible reply.
            llm_response.result_chain = None
            llm_response.completion_text = None

        if llm_response.completion_text is None and not llm_response.tools_call_args:
            logger.error(f"API 返回的 completion 无法解析：{completion}。")
            raise Exception(f"API 返回的 completion 无法解析：{completion}。")

        llm_response.raw_completion = completion
        llm_response.id = completion.id

        if completion.usage:
            llm_response.usage = self._extract_usage(completion.usage)

        return llm_response

    async def _prepare_chat_payload(
        self,
        prompt: str | None,
        image_urls: list[str] | None = None,
        contexts: list[dict] | list[Message] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
        **kwargs,
    ) -> tuple:
        """准备聊天所需的有效载荷和上下文"""
        if contexts is None:
            contexts = []
        new_record = None
        if prompt is not None:
            new_record = await self.assemble_context(
                prompt, image_urls, extra_user_content_parts
            )
        context_query = self._ensure_message_to_dicts(contexts)
        if new_record:
            context_query.append(new_record)
        if system_prompt:
            context_query.insert(0, {"role": "system", "content": system_prompt})

        for part in context_query:
            if "_no_save" in part:
                del part["_no_save"]

        # tool calls result
        if tool_calls_result:
            if isinstance(tool_calls_result, ToolCallsResult):
                context_query.extend(tool_calls_result.to_openai_messages())
            else:
                for tcr in tool_calls_result:
                    context_query.extend(tcr.to_openai_messages())

        model = model or self.get_model()

        payloads = {"messages": context_query, "model": model}

        self._finally_convert_payload(payloads)

        return payloads, context_query

    def _finally_convert_payload(self, payloads: dict) -> None:
        """Finally convert the payload. Such as think part conversion, tool inject."""
        for message in payloads.get("messages", []):
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), list
            ):
                reasoning_content = ""
                new_content = []  # not including think part
                for part in message["content"]:
                    if part.get("type") == "think":
                        reasoning_content += str(part.get("think"))
                    else:
                        new_content.append(part)
                message["content"] = new_content
                # reasoning key is "reasoning_content"
                if reasoning_content:
                    message["reasoning_content"] = reasoning_content

    async def _handle_api_error(
        self,
        e: Exception,
        payloads: dict,
        context_query: list,
        func_tool: ToolSet | None,
        chosen_key: str,
        available_api_keys: list[str],
        retry_cnt: int,
        max_retries: int,
        image_fallback_used: bool = False,
    ) -> tuple:
        """处理API错误并尝试恢复"""
        if "429" in str(e):
            logger.warning(
                f"API 调用过于频繁，尝试使用其他 Key 重试。当前 Key: {chosen_key[:12]}",
            )
            # 最后一次不等待
            if retry_cnt < max_retries - 1:
                await asyncio.sleep(1)
            available_api_keys.remove(chosen_key)
            if len(available_api_keys) > 0:
                chosen_key = random.choice(available_api_keys)
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                )
            raise e
        if "maximum context length" in str(e):
            logger.warning(
                f"上下文长度超过限制。尝试弹出最早的记录然后重试。当前记录条数: {len(context_query)}",
            )
            await self.pop_record(context_query)
            payloads["messages"] = context_query
            return (
                False,
                chosen_key,
                available_api_keys,
                payloads,
                context_query,
                func_tool,
                image_fallback_used,
            )
        if "The model is not a VLM" in str(e):  # siliconcloud
            if image_fallback_used or not self._context_contains_image(context_query):
                raise e
            # 尝试删除所有 image
            return await self._fallback_to_text_only_and_retry(
                payloads,
                context_query,
                chosen_key,
                available_api_keys,
                func_tool,
                "model_not_vlm",
                image_fallback_used=True,
            )
        if self._is_content_moderated_upload_error(e):
            if image_fallback_used or not self._context_contains_image(context_query):
                raise e
            return await self._fallback_to_text_only_and_retry(
                payloads,
                context_query,
                chosen_key,
                available_api_keys,
                func_tool,
                "image_content_moderated",
                image_fallback_used=True,
            )

        if self._is_upstream_server_error(e):
            # ????????? schema ???? 5xx??????????
            if func_tool is not None:
                logger.warning(
                    "???? 5xx??????? schema ??????: %s",
                    self.get_model(),
                )
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    None,
                    image_fallback_used,
                )

            if payloads.get("tools"):
                logger.warning(
                    "???? 5xx????? tools ??????: %s",
                    self.get_model(),
                )
                payloads.pop("tools", None)
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    None,
                    image_fallback_used,
                )

            if not image_fallback_used and self._context_contains_image(context_query):
                return await self._fallback_to_text_only_and_retry(
                    payloads,
                    context_query,
                    chosen_key,
                    available_api_keys,
                    func_tool,
                    "upstream_5xx",
                    image_fallback_used=True,
                )

            tool_like_count = sum(
                1
                for msg in context_query
                if isinstance(msg, dict) and msg.get("role") in {"tool", "function"}
            )
            if tool_like_count > 0:
                sanitized_context = [
                    msg
                    for msg in context_query
                    if not (
                        isinstance(msg, dict)
                        and msg.get("role") in {"tool", "function"}
                    )
                ]
                if 0 < len(sanitized_context) < len(context_query):
                    logger.warning(
                        "???? 5xx?????? tool/function ??????????????: %s",
                        len(context_query) - len(sanitized_context),
                    )
                    payloads["messages"] = sanitized_context
                    return (
                        False,
                        chosen_key,
                        available_api_keys,
                        payloads,
                        sanitized_context,
                        func_tool,
                        image_fallback_used,
                    )

            if len(context_query) > 4:
                shrink_steps = min(4, len(context_query) - 4)
                logger.warning(
                    "???? 5xx????????????????????: %s, ????: %s",
                    len(context_query),
                    shrink_steps,
                )
                for _ in range(shrink_steps):
                    await self.pop_record(context_query)
                payloads["messages"] = context_query
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                )
        if (
            "Function calling is not enabled" in str(e)
            or ("tool" in str(e).lower() and "support" in str(e).lower())
            or ("function" in str(e).lower() and "support" in str(e).lower())
        ):
            # openai, ollama, gemini openai, siliconcloud 的错误提示与 code 不统一，只能通过字符串匹配
            logger.info(
                f"{self.get_model()} 不支持函数工具调用，已自动去除，不影响使用。",
            )
            payloads.pop("tools", None)
            return (
                False,
                chosen_key,
                available_api_keys,
                payloads,
                context_query,
                None,
                image_fallback_used,
            )
        # logger.error(f"发生了错误。Provider 配置如下: {self.provider_config}")

        if "tool" in str(e).lower() and "support" in str(e).lower():
            logger.error("疑似该模型不支持函数调用工具调用。请输入 /tool off_all")

        if is_connection_error(e):
            proxy = self.provider_config.get("proxy", "")
            log_connection_failure("OpenAI", e, proxy)

        raise e

    async def text_chat(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        **kwargs,
    ) -> LLMResponse:
        payloads, context_query = await self._prepare_chat_payload(
            prompt,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            extra_user_content_parts=extra_user_content_parts,
            **kwargs,
        )

        llm_response = None
        max_retries = int(self.provider_config.get("max_retries", 10) or 10)
        if max_retries < 1:
            max_retries = 1
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys)
        image_fallback_used = False

        last_exception = None
        completed = False
        for retry_cnt in range(max_retries):
            try:
                self.client.api_key = chosen_key
                llm_response = await self._query(payloads, func_tool)
                completed = True
                break
            except Exception as e:
                last_exception = e
                (
                    success,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                ) = await self._handle_api_error(
                    e,
                    payloads,
                    context_query,
                    func_tool,
                    chosen_key,
                    available_api_keys,
                    retry_cnt,
                    max_retries,
                    image_fallback_used=image_fallback_used,
                )
                if success:
                    completed = True
                    break

        if not completed or llm_response is None:
            logger.error(f"API 调用失败，重试 {max_retries} 次仍然失败。")
            if last_exception is None:
                raise Exception("未知错误")
            raise last_exception
        return llm_response

    async def text_chat_stream(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        """流式对话，与服务商交互并逐步返回结果"""
        payloads, context_query = await self._prepare_chat_payload(
            prompt,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            **kwargs,
        )

        max_retries = int(self.provider_config.get("max_retries", 10) or 10)
        if max_retries < 1:
            max_retries = 1
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys)
        image_fallback_used = False

        last_exception = None
        completed = False
        for retry_cnt in range(max_retries):
            try:
                self.client.api_key = chosen_key
                async for response in self._query_stream(payloads, func_tool):
                    yield response
                completed = True
                break
            except Exception as e:
                last_exception = e
                (
                    success,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                ) = await self._handle_api_error(
                    e,
                    payloads,
                    context_query,
                    func_tool,
                    chosen_key,
                    available_api_keys,
                    retry_cnt,
                    max_retries,
                    image_fallback_used=image_fallback_used,
                )
                if success:
                    completed = True
                    break

        if not completed:
            logger.error(f"API 调用失败，重试 {max_retries} 次仍然失败。")
            if last_exception is None:
                raise Exception("未知错误")
            raise last_exception

    async def _remove_image_from_context(self, contexts: list):
        """从上下文中删除所有带有 image 的记录"""
        new_contexts = []

        for context in contexts:
            if "content" in context and isinstance(context["content"], list):
                # continue
                new_content = []
                for item in context["content"]:
                    if isinstance(item, dict) and "image_url" in item:
                        continue
                    new_content.append(item)
                if not new_content:
                    # 用户只发了图片
                    new_content = [{"type": "text", "text": "[图片]"}]
                context["content"] = new_content
            new_contexts.append(context)
        return new_contexts

    def get_current_key(self) -> str:
        return self.client.api_key

    def get_keys(self) -> list[str]:
        return self.api_keys

    def set_key(self, key) -> None:
        self.client.api_key = key

    async def assemble_context(
        self,
        text: str,
        image_urls: list[str] | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
    ) -> dict:
        """组装成符合 OpenAI 格式的 role 为 user 的消息段"""

        async def resolve_image_part(image_url: str) -> dict | None:
            if image_url.startswith("http"):
                image_path = await download_image_by_url(image_url)
                image_data = await self.encode_image_bs64(image_path)
            elif image_url.startswith("file:///"):
                image_path = image_url.replace("file:///", "")
                image_data = await self.encode_image_bs64(image_path)
            else:
                image_data = await self.encode_image_bs64(image_url)
            if not image_data:
                logger.warning(f"图片 {image_url} 得到的结果为空，将忽略。")
                return None
            return {
                "type": "image_url",
                "image_url": {"url": image_data},
            }

        # 构建内容块列表
        content_blocks = []

        # 1. 用户原始发言（OpenAI 建议：用户发言在前）
        if text:
            content_blocks.append({"type": "text", "text": text})
        elif image_urls:
            # 如果没有文本但有图片，添加占位文本
            content_blocks.append({"type": "text", "text": "[图片]"})
        elif extra_user_content_parts:
            # 如果只有额外内容块，也需要添加占位文本
            content_blocks.append({"type": "text", "text": " "})

        # 2. 额外的内容块（系统提醒、指令等）
        if extra_user_content_parts:
            for part in extra_user_content_parts:
                if isinstance(part, TextPart):
                    content_blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageURLPart):
                    image_part = await resolve_image_part(part.image_url.url)
                    if image_part:
                        content_blocks.append(image_part)
                else:
                    raise ValueError(f"不支持的额外内容块类型: {type(part)}")

        # 3. 图片内容
        if image_urls:
            for image_url in image_urls:
                image_part = await resolve_image_part(image_url)
                if image_part:
                    content_blocks.append(image_part)

        # 如果只有主文本且没有额外内容块和图片，返回简单格式以保持向后兼容
        if (
            text
            and not extra_user_content_parts
            and not image_urls
            and len(content_blocks) == 1
            and content_blocks[0]["type"] == "text"
        ):
            return {"role": "user", "content": content_blocks[0]["text"]}

        # 否则返回多模态格式
        return {"role": "user", "content": content_blocks}

    async def encode_image_bs64(self, image_url: str) -> str:
        """将图片转换为 base64"""
        if image_url.startswith("base64://"):
            return image_url.replace("base64://", "data:image/jpeg;base64,")
        with open(image_url, "rb") as f:
            image_bs64 = base64.b64encode(f.read()).decode("utf-8")
            return "data:image/jpeg;base64," + image_bs64

    async def terminate(self):
        if self.client:
            await self.client.close()
