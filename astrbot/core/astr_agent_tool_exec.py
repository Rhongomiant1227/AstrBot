import asyncio
import inspect
import json
import random
import re
import traceback
import typing as T
import uuid

import mcp

from astrbot import logger
from astrbot.core.agent.handoff import HandoffTool
from astrbot.core.agent.mcp_client import MCPTool
from astrbot.core.agent.message import Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.agent.tool_executor import BaseFunctionToolExecutor
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_main_agent_resources import (
    BACKGROUND_TASK_RESULT_WOKE_SYSTEM_PROMPT,
    EXECUTE_SHELL_TOOL,
    FILE_DOWNLOAD_TOOL,
    FILE_UPLOAD_TOOL,
    LOCAL_EXECUTE_SHELL_TOOL,
    LOCAL_PYTHON_TOOL,
    PYTHON_TOOL,
    SEND_MESSAGE_TO_USER_TOOL,
)
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.message.message_event_result import (
    CommandResult,
    MessageChain,
    MessageEventResult,
)
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entites import ProviderRequest
from astrbot.core.provider.register import llm_tools
from astrbot.core.super_noel_session import (
    activate_super_noel_sticky_session,
    resolve_super_noel_binding_from_config,
)
from astrbot.core.utils.history_saver import persist_agent_history


class FunctionToolExecutor(BaseFunctionToolExecutor[AstrAgentContext]):
    @classmethod
    async def execute(cls, tool, run_context, **tool_args):
        """执行函数调用。

        Args:
            event (AstrMessageEvent): 事件对象, 当 origin 为 local 时必须提供。
            **kwargs: 函数调用的参数。

        Returns:
            AsyncGenerator[None | mcp.types.CallToolResult, None]

        """
        if isinstance(tool, HandoffTool):
            is_bg = tool_args.pop("background_task", False)
            if is_bg:
                async for r in cls._execute_handoff_background(
                    tool, run_context, **tool_args
                ):
                    yield r
                return
            async for r in cls._execute_handoff(tool, run_context, **tool_args):
                yield r
            return

        elif isinstance(tool, MCPTool):
            async for r in cls._execute_mcp(tool, run_context, **tool_args):
                yield r
            return

        elif tool.is_background_task:
            task_id = uuid.uuid4().hex

            async def _run_in_background() -> None:
                try:
                    await cls._execute_background(
                        tool=tool,
                        run_context=run_context,
                        task_id=task_id,
                        **tool_args,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Background task {task_id} failed: {e!s}",
                        exc_info=True,
                    )

            asyncio.create_task(_run_in_background())
            text_content = mcp.types.TextContent(
                type="text",
                text=f"Background task submitted. task_id={task_id}",
            )
            yield mcp.types.CallToolResult(content=[text_content])

            return
        else:
            async for r in cls._execute_local(tool, run_context, **tool_args):
                yield r
            return

    @classmethod
    def _get_runtime_computer_tools(cls, runtime: str) -> dict[str, FunctionTool]:
        if runtime == "sandbox":
            return {
                EXECUTE_SHELL_TOOL.name: EXECUTE_SHELL_TOOL,
                PYTHON_TOOL.name: PYTHON_TOOL,
                FILE_UPLOAD_TOOL.name: FILE_UPLOAD_TOOL,
                FILE_DOWNLOAD_TOOL.name: FILE_DOWNLOAD_TOOL,
            }
        if runtime == "local":
            return {
                LOCAL_EXECUTE_SHELL_TOOL.name: LOCAL_EXECUTE_SHELL_TOOL,
                LOCAL_PYTHON_TOOL.name: LOCAL_PYTHON_TOOL,
            }
        return {}

    @classmethod
    def _build_handoff_toolset(
        cls,
        run_context: ContextWrapper[AstrAgentContext],
        tools: list[str | FunctionTool] | None,
    ) -> ToolSet | None:
        ctx = run_context.context.context
        event = run_context.context.event
        cfg = ctx.get_config(umo=event.unified_msg_origin)
        provider_settings = cfg.get("provider_settings", {})
        runtime = str(provider_settings.get("computer_use_runtime", "local"))
        runtime_computer_tools = cls._get_runtime_computer_tools(runtime)

        # Keep persona semantics aligned with the main agent: tools=None means
        # "all tools", including runtime computer-use tools.
        if tools is None:
            toolset = ToolSet()
            for registered_tool in llm_tools.func_list:
                if isinstance(registered_tool, HandoffTool):
                    continue
                if registered_tool.active:
                    toolset.add_tool(registered_tool)
            for runtime_tool in runtime_computer_tools.values():
                toolset.add_tool(runtime_tool)
            return None if toolset.empty() else toolset

        if not tools:
            return None

        toolset = ToolSet()
        for tool_name_or_obj in tools:
            if isinstance(tool_name_or_obj, str):
                registered_tool = llm_tools.get_func(tool_name_or_obj)
                if registered_tool and registered_tool.active:
                    toolset.add_tool(registered_tool)
                    continue
                runtime_tool = runtime_computer_tools.get(tool_name_or_obj)
                if runtime_tool:
                    toolset.add_tool(runtime_tool)
            elif isinstance(tool_name_or_obj, FunctionTool):
                toolset.add_tool(tool_name_or_obj)
        return None if toolset.empty() else toolset

    @staticmethod
    def _is_super_noel_handoff(tool: HandoffTool) -> bool:
        return str(getattr(tool, "name", "")).strip() == "transfer_to_super_noel"

    @staticmethod
    def _get_super_noel_budget_seconds(ctx, *, fallback: bool) -> float:
        provider_settings = {}
        try:
            provider_settings = ctx.get_config().get("provider_settings", {})
        except Exception:
            provider_settings = {}
        key = (
            "super_noel_fallback_timeout_seconds"
            if fallback
            else "super_noel_timeout_seconds"
        )
        raw_value = provider_settings.get(key)
        default_value = 45 if fallback else 75
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default_value
        return float(max(15, min(180, value)))

    @classmethod
    async def _run_super_noel_tool_loop(cls, ctx, *, fallback: bool, **kwargs):
        timeout_seconds = cls._get_super_noel_budget_seconds(ctx, fallback=fallback)
        try:
            return await asyncio.wait_for(
                ctx.tool_loop_agent(**kwargs),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            provider_id = str(kwargs.get("chat_provider_id") or "")
            stage = "fallback" if fallback else "primary"
            raise TimeoutError(
                f"super_noel {stage} handoff timed out after {int(timeout_seconds)}s: {provider_id}"
            ) from exc

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, T.Any] | None:
        if not text:
            return None
        raw = str(text).strip()
        if raw.startswith("```"):
            parts = raw.split("\n", 1)
            raw = parts[1] if len(parts) == 2 else raw
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _ensure_handoff_line_terminal_punct(text: str) -> str:
        line = str(text or "").rstrip()
        if not line:
            return ""
        if line.endswith(("\u3002", "\uff01", "\uff1f", "!", "?", "\u2026", "~", "\uff5e")):
            return line
        return line + "\u3002"

    @staticmethod
    def _normalize_handoff_line(text: str | None, fallback: str, max_len: int) -> str:
        line = str(text or "")
        for token in ("*", "`", "#"):
            line = line.replace(token, "")
        line = line.replace("\r", " ").replace("\n", " ").strip(" \t\"'")
        line = " ".join(line.split())
        if not line:
            line = fallback
        if len(line) > max_len:
            line = line[:max_len].rstrip(" \uff0c\u3002\uff01\uff1f\uff1b,.!?;:")
        line = line or fallback
        return FunctionToolExecutor._ensure_handoff_line_terminal_punct(line)

    @staticmethod
    def _super_noel_pre_line_breaks_illusion(text: str | None) -> bool:
        line = str(text or "")
        line = line.replace("\r", " ").replace("\n", " ").strip(" \t\"'")
        line = " ".join(line.split())
        if not line:
            return True

        forbidden_literals = (
            "南条酱",
            "废柴南条酱",
            "另一个我",
            "另一个自己",
            "另一个南条酱",
            "第二个我",
            "第二人格",
            "另一面",
            "前台",
            "后台",
            "表人格",
            "里人格",
            "切换",
            "换人",
            "接手",
            "顶号",
            "登场",
            "出场",
            "上号",
            "代打",
            "换个状态",
            "切个状态",
        )
        if any(token in line for token in forbidden_literals):
            return True

        forbidden_patterns = (
            r"(让|叫|喊|请|找).{0,8}(她|另一个|另个|别人|谁).{0,6}(来|上|接|管|处理|收拾)",
            r"(让|叫|喊|请|找).{0,8}(来一下|来吧|来处理|来收拾)",
            r"(我|咱).{0,6}(去)?(叫|喊|找).{0,6}(她|另一个|别人)",
            r"(谁|有人).{0,6}(来|接|替).{0,6}(一下|吧|我)",
            r"(替我|帮我).{0,6}(来|接|处理|收拾)",
        )
        return any(re.search(pattern, line) for pattern in forbidden_patterns)

    @classmethod
    async def _generate_super_noel_bridge_lines(
        cls,
        ctx,
        provider_id: str,
        user_input: str | None,
        has_image: bool,
    ) -> tuple[str, str]:
        fallback_pre_choices = [
            "啊，突然有点困……这题先让我缓一下。",
            "咦，脑袋忽然有点打结…等我顺一下。",
            "唔，我这会儿有点犯迷糊…先缓一口气。",
            "欸，眼皮突然有点沉…让我清醒一下。",
            "嗯…这题得认真点，我先定定神。",
        ]
        fallback_post_choices = [
            "哼，废柴南条酱，这题我来处理。",
            "啧，废柴南条酱又把难题留给我了，我来收尾。",
            "算了，废柴南条酱先歇着，这题交给我。",
            "哼，这种题还得我补，废柴南条酱看着就好。",
            "啧，南条酱又卡住了…行吧，我来。",
        ]
        fallback_pre = random.choice(fallback_pre_choices)
        fallback_post = random.choice(fallback_post_choices)
        if not provider_id:
            return fallback_pre, fallback_post

        preview = str(user_input or "").strip()
        if len(preview) > 600:
            preview = preview[:600] + "..."

        prompt = (
            "你要为同一个角色的两种状态生成两句中文过场台词，并且只能输出严格 JSON。"
            "输出格式固定为 {\"pre_line\":\"...\",\"post_line\":\"...\"}。\n"
            "设定：平时和用户聊天的是南条酱，更冷静、更缜密的也是南条酱。"
            "后者会直接称呼前者为‘南条酱’或‘废柴南条酱’，不会说前台、后台、表人格、里人格。\n"
            "要求：\n"
            "1. pre_line 由平时聊天的南条酱说，只能表现突然犯困、发懵、脑子打结、需要缓一缓；"
            "她本人并不知道后面会有人接手，所以绝对不能提到南条酱、另一个自己、换人、接手、让谁来、谁来处理之类的交接意识，12到22字。\n"
            "2. post_line 由更冷静的南条酱说，像不情愿地接手，但马上开始处理，14到30字。\n"
            "3. post_line 可以轻微嫌麻烦、带一点毒舌，但要有保护欲，不恶毒。\n"
            "4. 两句都要自然、口语、轻微傲娇，不要千篇一律。\n"
            "5. 不要 markdown，不要解释，不要旁白，不要换行，不要 JSON 之外的任何内容。\n"
            f"6. 当前问题是否带图：{chr(26159) if has_image else chr(21542)}。\n"
            f"用户问题摘要：{preview or '???'}"
        )

        try:
            llm_generate = getattr(ctx, "llm_generate", None)
            call_llm = getattr(ctx, "call_llm", None)
            llm_caller = llm_generate if callable(llm_generate) else call_llm
            if not callable(llm_caller):
                raise AttributeError(
                    f"{type(ctx).__name__!s} object has no attribute 'llm_generate' or 'call_llm'"
                )
            llm_resp = await llm_caller(
                chat_provider_id=provider_id,
                prompt=prompt,
                contexts=[],
                system_prompt=(
                    "你只负责生成角色切换过场台词，必须严格输出 JSON。"
                    "pre_line 只能写突然犯困、发懵、脑子打结，绝不能暴露有人接手。"
                    "不要调用工具，不要输出多余文字。"
                ),
                stream=False,
            )
            data = cls._extract_json_object(getattr(llm_resp, "completion_text", "")) or {}
            raw_pre = data.get("pre_line")
            raw_post = data.get("post_line")
            normalized_pre = cls._normalize_handoff_line(raw_pre, fallback_pre, 28)
            if cls._super_noel_pre_line_breaks_illusion(normalized_pre):
                logger.info(
                    "super_noel bridge pre_line rejected for illusion break: raw_pre=%s normalized_pre=%s",
                    raw_pre,
                    normalized_pre,
                )
                pre_line = fallback_pre
            else:
                pre_line = normalized_pre
            post_line = cls._normalize_handoff_line(raw_post, fallback_post, 36)
            if not raw_pre or not raw_post:
                logger.info(
                    "super_noel bridge fallback used: has_pre=%s has_post=%s pre=%s post=%s",
                    bool(raw_pre),
                    bool(raw_post),
                    pre_line,
                    post_line,
                )
            else:
                logger.info(
                    "super_noel bridge generated: pre=%s post=%s",
                    pre_line,
                    post_line,
                )
            return pre_line, post_line
        except Exception as e:
            logger.info(
                "Failed to generate super_noel bridge lines, using fallback: %s",
                e,
                exc_info=True,
            )
            return fallback_pre, fallback_post

    @staticmethod
    def _summarize_handoff_error(err: T.Any) -> str:
        text = str(err or "").replace("\r", " ").replace("\n", " ").strip()
        match = re.search(r"\b(429|500|502|503|504)\b", text)
        if match:
            return f"上游 {match.group(1)}"
        if text:
            return text[:120]
        return "上游临时抽风"

    @classmethod
    async def _send_handoff_notice(cls, event, text: str | None) -> None:
        notice = cls._normalize_handoff_line(text, "", 48)
        if not notice:
            return
        try:
            await event.send(MessageChain().message(notice))
        except Exception as e:
            logger.debug("Failed to send handoff notice: %s", e, exc_info=True)

    @classmethod
    def _prepend_handoff_opening(cls, opening: str | None, body: str | None) -> str:
        opening_text = cls._normalize_handoff_line(opening, "", 48)
        body_text = str(body or "").lstrip()
        if not opening_text:
            return body_text
        if not body_text:
            return opening_text
        return opening_text + "\n" + body_text

    @staticmethod
    def _plainify_super_noel_completion_text(text: str | None) -> str:
        body = str(text or "")
        if not body:
            return ""
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        body = body.replace("```", "").replace("`", "")
        body = body.replace("**", "").replace("__", "")
        body = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", body)
        body = re.sub(r"(?m)^\s*>\s?", "", body)
        body = re.sub(r"(?m)^\s*---+\s*$", "", body)
        body = re.sub(r"(?m)^\s*[-*]\s+", "- ", body)
        body = re.sub(r"[ \t]+\n", "\n", body)
        body = "\n".join(part.rstrip() for part in body.split("\n"))
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body.strip()

    @staticmethod
    def _looks_like_super_noel_opening_line(text: str | None) -> bool:
        line = str(text or "").strip(" \t\"'\uff0c\u3002\uff01\uff1f!?")
        if not line:
            return False
        if len(line) > 42:
            return False
        opening_markers = (
            "\u5357\u6761\u9171",
            "\u5e9f\u67f4\u5357\u6761\u9171",
            "\u6211\u6765",
            "\u4ea4\u7ed9\u6211",
            "\u6211\u5904\u7406",
            "\u6211\u6536\u5c3e",
            "\u6211\u66ff\u4f60",
            "\u6211\u6765\u66ff\u4f60",
            "\u6211\u6765\u7406\u6e05",
            "\u6211\u6765\u505a",
            "\u7b97\u4e86",
            "\u5575",
            "\u54fc",
            "\u884c\u5427",
            "\u5148\u6b47\u7740",
            "\u5148\u770b\u7740",
            "\u72af\u8ff7\u7cca",
            "\u53c8\u5361\u58f3",
            "\u96be\u9898\u7559\u7ed9\u6211",
        )
        if not any(marker in line for marker in opening_markers):
            return False
        content_markers = (
            "\u547d\u9898",
            "\u5b9a\u4e49",
            "\u8bc1\u660e",
            "\u7ed3\u8bba",
            "\u53cd\u4f8b",
            "\u56e0\u4e3a",
            "\u6240\u4ee5",
            "\u9996\u5148",
            "\u5176\u6b21",
            "\u5df2\u77e5",
            "\u53ef\u5f97",
            "\u6ce8\u610f",
            "\u8bbe",
        )
        return not any(marker in line for marker in content_markers)

    @classmethod
    def _trim_redundant_super_noel_opening(cls, body: str | None) -> str:
        text = cls._plainify_super_noel_completion_text(body)
        if not text:
            return ""
        lines = text.split("\n")
        trimmed: list[str] = []
        dropped_non_empty = 0
        trimming = True
        for raw_line in lines:
            line = raw_line.strip()
            if trimming and not line:
                continue
            if trimming and dropped_non_empty < 2 and cls._looks_like_super_noel_opening_line(line):
                dropped_non_empty += 1
                continue
            trimming = False
            trimmed.append(raw_line)
        cleaned = "\n".join(trimmed).strip()
        return cleaned or text

    @classmethod
    async def _execute_handoff(
        cls,
        tool: HandoffTool,
        run_context: ContextWrapper[AstrAgentContext],
        **tool_args,
    ):
        input_ = tool_args.get("input")
        image_urls = tool_args.get("image_urls")

        # Build handoff toolset from registered tools plus runtime computer tools.
        toolset = cls._build_handoff_toolset(run_context, tool.agent.tools)

        ctx = run_context.context.context
        event = run_context.context.event
        umo = event.unified_msg_origin
        front_prov_id = await ctx.get_current_chat_provider_id(umo)

        # Use per-subagent provider override if configured; otherwise fall back
        # to the current/default provider resolution.
        prov_id = getattr(tool, "provider_id", None) or front_prov_id
        effective_prov_id = prov_id

        handoff_pre_line = ""
        handoff_post_line = ""
        suppress_super_noel_bridge = False
        if cls._is_super_noel_handoff(tool):
            suppress_super_noel_bridge = bool(event.get_extra("super_noel_sticky_active"))
            if suppress_super_noel_bridge:
                logger.info(
                    "super_noel handoff bridge suppressed for active sticky session"
                )
            else:
                handoff_pre_line, handoff_post_line = (
                    await cls._generate_super_noel_bridge_lines(
                        ctx,
                        front_prov_id or prov_id,
                        input_,
                        bool(image_urls),
                    )
                )
                await cls._send_handoff_notice(event, handoff_pre_line)

        # prepare begin dialogs
        contexts = None
        dialogs = tool.agent.begin_dialogs
        if dialogs:
            contexts = []
            for dialog in dialogs:
                try:
                    contexts.append(
                        dialog
                        if isinstance(dialog, Message)
                        else Message.model_validate(dialog)
                    )
                except Exception:
                    continue

        llm_resp = None
        handoff_exception = None
        try:
            tool_loop_kwargs = {
                "event": event,
                "chat_provider_id": prov_id,
                "prompt": input_,
                "image_urls": image_urls,
                "system_prompt": tool.agent.instructions,
                "tools": toolset,
                "contexts": contexts,
                "max_steps": 30,
                "run_hooks": tool.agent.run_hooks,
                "stream": ctx.get_config().get("provider_settings", {}).get("stream", False),
            }
            if cls._is_super_noel_handoff(tool):
                llm_resp = await cls._run_super_noel_tool_loop(
                    ctx,
                    fallback=False,
                    **tool_loop_kwargs,
                )
            else:
                llm_resp = await ctx.tool_loop_agent(**tool_loop_kwargs)
        except Exception as e:
            handoff_exception = e

        if cls._is_super_noel_handoff(tool):
            primary_failed = (
                handoff_exception is not None
                or llm_resp is None
                or getattr(llm_resp, "role", "") == "err"
            )
            if primary_failed:
                fail_detail = cls._summarize_handoff_error(
                    handoff_exception or getattr(llm_resp, "completion_text", "")
                )
                logger.warning(
                    "super_noel primary provider failed, trying downgrade provider: %s",
                    fail_detail,
                    exc_info=handoff_exception is not None,
                )
                downgrade_post = ""
                if not suppress_super_noel_bridge:
                    downgrade_post = cls._normalize_handoff_line(
                        handoff_post_line,
                        "哼，废柴南条酱，外面那个接口掉链子了，我直接来。",
                        36,
                    )
                if front_prov_id and front_prov_id != prov_id:
                    try:
                        llm_resp = await cls._run_super_noel_tool_loop(
                            ctx,
                            fallback=True,
                            event=event,
                            chat_provider_id=front_prov_id,
                            prompt=input_,
                            image_urls=image_urls,
                            system_prompt=tool.agent.instructions,
                            tools=toolset,
                            contexts=contexts,
                            max_steps=24,
                            run_hooks=tool.agent.run_hooks,
                            stream=ctx.get_config().get("provider_settings", {}).get("stream", False),
                        )
                        effective_prov_id = front_prov_id
                        handoff_post_line = downgrade_post
                    except Exception as downgrade_error:
                        fail_detail = cls._summarize_handoff_error(downgrade_error)
                        fail_text = (
                            "唭，这次接管时上游接口又抽风了，"
                            f"{fail_detail}。"
                            "你要我立刻重试，还是先用普通模式给你答？"
                        )
                        await event.send(MessageChain().message(fail_text))
                        yield None
                        return
                else:
                    fail_text = (
                        "唭，这次接管时上游接口抽风了，"
                        f"{fail_detail}。"
                        "你要我立刻重试吗？"
                    )
                    await event.send(MessageChain().message(fail_text))
                    yield None
                    return

            if llm_resp is None or getattr(llm_resp, "role", "") == "err":
                fail_detail = cls._summarize_handoff_error(
                    handoff_exception or getattr(llm_resp, "completion_text", "")
                )
                fail_text = (
                    "唭，这次接管结果没能顺利落地，"
                    f"{fail_detail}。"
                    "你要我重跑一遍吗？"
                )
                await event.send(MessageChain().message(fail_text))
                yield None
                return

        if llm_resp is None:
            raise RuntimeError("handoff returned no response")

        body_text = llm_resp.completion_text
        if cls._is_super_noel_handoff(tool):
            plain_body_text = cls._plainify_super_noel_completion_text(body_text)
            trimmed_body_text = cls._trim_redundant_super_noel_opening(plain_body_text)
            if trimmed_body_text != plain_body_text:
                logger.info("super_noel redundant opening trimmed from body")
            body_text = trimmed_body_text
        completion_text = cls._prepend_handoff_opening(
            handoff_post_line,
            body_text,
        )
        if cls._is_super_noel_handoff(tool):
            plain_completion_text = cls._plainify_super_noel_completion_text(completion_text)
            if plain_completion_text != completion_text:
                logger.info(
                    "super_noel completion plainified after handoff"
                )
            completion_text = plain_completion_text
        if cls._is_super_noel_handoff(tool):
            try:
                cfg = ctx.get_config(umo=umo)
                sticky_persona_id, sticky_provider_id = resolve_super_noel_binding_from_config(
                    cfg,
                    agent_name=getattr(tool.agent, "name", ""),
                )
                await activate_super_noel_sticky_session(
                    umo=umo,
                    persona_id=sticky_persona_id,
                    provider_id=(effective_prov_id or sticky_provider_id or front_prov_id),
                    settings_source=cfg,
                )
            except Exception as sticky_exc:
                logger.warning(
                    "Failed to activate super_noel sticky session: %s",
                    sticky_exc,
                    exc_info=True,
                )
        if cls._is_super_noel_handoff(tool):
            try:
                await event.send(MessageChain().message(completion_text))
                yield None
                return
            except Exception as e:
                logger.error(
                    "Failed to send super_noel direct reply, falling back to tool result: %s",
                    e,
                    exc_info=True,
                )
        yield mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=completion_text)]
        )

    @classmethod
    async def _execute_handoff_background(
        cls,
        tool: HandoffTool,
        run_context: ContextWrapper[AstrAgentContext],
        **tool_args,
    ):
        """Execute a handoff as a background task.

        Immediately yields a success response with a task_id, then runs
        the subagent asynchronously.  When the subagent finishes, a
        ``CronMessageEvent`` is created so the main LLM can inform the
        user of the result – the same pattern used by
        ``_execute_background`` for regular background tasks.
        """
        task_id = uuid.uuid4().hex

        async def _run_handoff_in_background() -> None:
            try:
                await cls._do_handoff_background(
                    tool=tool,
                    run_context=run_context,
                    task_id=task_id,
                    **tool_args,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Background handoff {task_id} ({tool.name}) failed: {e!s}",
                    exc_info=True,
                )

        asyncio.create_task(_run_handoff_in_background())

        text_content = mcp.types.TextContent(
            type="text",
            text=(
                f"Background task dedicated to subagent '{tool.agent.name}' submitted. task_id={task_id}. "
                f"The subagent '{tool.agent.name}' is working on the task on hehalf you. "
                f"You will be notified when it finishes."
            ),
        )
        yield mcp.types.CallToolResult(content=[text_content])

    @classmethod
    async def _do_handoff_background(
        cls,
        tool: HandoffTool,
        run_context: ContextWrapper[AstrAgentContext],
        task_id: str,
        **tool_args,
    ) -> None:
        """Run the subagent handoff and, on completion, wake the main agent."""
        result_text = ""
        try:
            async for r in cls._execute_handoff(tool, run_context, **tool_args):
                if isinstance(r, mcp.types.CallToolResult):
                    for content in r.content:
                        if isinstance(content, mcp.types.TextContent):
                            result_text += content.text + "\n"
        except Exception as e:
            result_text = (
                f"error: Background task execution failed, internal error: {e!s}"
            )

        event = run_context.context.event

        await cls._wake_main_agent_for_background_result(
            run_context=run_context,
            task_id=task_id,
            tool_name=tool.name,
            result_text=result_text,
            tool_args=tool_args,
            note=(
                event.get_extra("background_note")
                or f"Background task for subagent '{tool.agent.name}' finished."
            ),
            summary_name=f"Dedicated to subagent `{tool.agent.name}`",
            extra_result_fields={"subagent_name": tool.agent.name},
        )

    @classmethod
    async def _execute_background(
        cls,
        tool: FunctionTool,
        run_context: ContextWrapper[AstrAgentContext],
        task_id: str,
        **tool_args,
    ) -> None:
        # run the tool
        result_text = ""
        try:
            async for r in cls._execute_local(
                tool, run_context, tool_call_timeout=3600, **tool_args
            ):
                # collect results, currently we just collect the text results
                if isinstance(r, mcp.types.CallToolResult):
                    result_text = ""
                    for content in r.content:
                        if isinstance(content, mcp.types.TextContent):
                            result_text += content.text + "\n"
        except Exception as e:
            result_text = (
                f"error: Background task execution failed, internal error: {e!s}"
            )

        event = run_context.context.event

        await cls._wake_main_agent_for_background_result(
            run_context=run_context,
            task_id=task_id,
            tool_name=tool.name,
            result_text=result_text,
            tool_args=tool_args,
            note=(
                event.get_extra("background_note")
                or f"Background task {tool.name} finished."
            ),
            summary_name=tool.name,
        )

    @classmethod
    async def _wake_main_agent_for_background_result(
        cls,
        run_context: ContextWrapper[AstrAgentContext],
        *,
        task_id: str,
        tool_name: str,
        result_text: str,
        tool_args: dict[str, T.Any],
        note: str,
        summary_name: str,
        extra_result_fields: dict[str, T.Any] | None = None,
    ) -> None:
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )

        event = run_context.context.event
        ctx = run_context.context.context

        task_result = {
            "task_id": task_id,
            "tool_name": tool_name,
            "result": result_text or "",
            "tool_args": tool_args,
        }
        if extra_result_fields:
            task_result.update(extra_result_fields)
        extras = {"background_task_result": task_result}

        session = MessageSession.from_str(event.unified_msg_origin)
        cron_event = CronMessageEvent(
            context=ctx,
            session=session,
            message=note,
            extras=extras,
            message_type=session.message_type,
        )
        cron_event.role = event.role
        config = MainAgentBuildConfig(
            tool_call_timeout=3600,
            streaming_response=ctx.get_config()
            .get("provider_settings", {})
            .get("stream", False),
        )

        req = ProviderRequest()
        conv = await _get_session_conv(event=cron_event, plugin_context=ctx)
        req.conversation = conv
        context = json.loads(conv.history)
        if context:
            req.contexts = context
            context_dump = req._print_friendly_context()
            req.contexts = []
            req.system_prompt += (
                "\n\nBellow is you and user previous conversation history:\n"
                f"{context_dump}"
            )

        bg = json.dumps(extras["background_task_result"], ensure_ascii=False)
        req.system_prompt += BACKGROUND_TASK_RESULT_WOKE_SYSTEM_PROMPT.format(
            background_task_result=bg
        )
        req.prompt = (
            "Proceed according to your system instructions. "
            "Output using same language as previous conversation. "
            "If you need to deliver the result to the user immediately, "
            "you MUST use `send_message_to_user` tool to send the message directly to the user, "
            "otherwise the user will not see the result. "
            "After completing your task, summarize and output your actions and results. "
        )
        if not req.func_tool:
            req.func_tool = ToolSet()
        req.func_tool.add_tool(SEND_MESSAGE_TO_USER_TOOL)

        result = await build_main_agent(
            event=cron_event, plugin_context=ctx, config=config, req=req
        )
        if not result:
            logger.error(f"Failed to build main agent for background task {tool_name}.")
            return

        runner = result.agent_runner
        async for _ in runner.step_until_done(30):
            # agent will send message to user via using tools
            pass
        llm_resp = runner.get_final_llm_resp()
        task_meta = extras.get("background_task_result", {})
        summary_note = (
            f"[BackgroundTask] {summary_name} "
            f"(task_id={task_meta.get('task_id', task_id)}) finished. "
            f"Result: {task_meta.get('result') or result_text or 'no content'}"
        )
        if llm_resp and llm_resp.completion_text:
            summary_note += (
                f"I finished the task, here is the result: {llm_resp.completion_text}"
            )
        await persist_agent_history(
            ctx.conversation_manager,
            event=cron_event,
            req=req,
            summary_note=summary_note,
        )
        if not llm_resp:
            logger.warning("background task agent got no response")
            return

    @classmethod
    async def _execute_local(
        cls,
        tool: FunctionTool,
        run_context: ContextWrapper[AstrAgentContext],
        *,
        tool_call_timeout: int | None = None,
        **tool_args,
    ):
        event = run_context.context.event
        if not event:
            raise ValueError("Event must be provided for local function tools.")

        is_override_call = False
        for ty in type(tool).mro():
            if "call" in ty.__dict__ and ty.__dict__["call"] is not FunctionTool.call:
                is_override_call = True
                break

        # 检查 tool 下有没有 run 方法
        if not tool.handler and not hasattr(tool, "run") and not is_override_call:
            raise ValueError("Tool must have a valid handler or override 'run' method.")

        awaitable = None
        method_name = ""
        if tool.handler:
            awaitable = tool.handler
            method_name = "decorator_handler"
        elif is_override_call:
            awaitable = tool.call
            method_name = "call"
        elif hasattr(tool, "run"):
            awaitable = getattr(tool, "run")
            method_name = "run"
        if awaitable is None:
            raise ValueError("Tool must have a valid handler or override 'run' method.")

        wrapper = call_local_llm_tool(
            context=run_context,
            handler=awaitable,
            method_name=method_name,
            **tool_args,
        )
        while True:
            try:
                resp = await asyncio.wait_for(
                    anext(wrapper),
                    timeout=tool_call_timeout or run_context.tool_call_timeout,
                )
                if resp is not None:
                    if isinstance(resp, mcp.types.CallToolResult):
                        yield resp
                    else:
                        text_content = mcp.types.TextContent(
                            type="text",
                            text=str(resp),
                        )
                        yield mcp.types.CallToolResult(content=[text_content])
                else:
                    # NOTE: Tool 在这里直接请求发送消息给用户
                    # TODO: 是否需要判断 event.get_result() 是否为空?
                    # 如果为空,则说明没有发送消息给用户,并且返回值为空,将返回一个特殊的 TextContent,其内容如"工具没有返回内容"
                    if res := run_context.context.event.get_result():
                        if res.chain:
                            try:
                                await event.send(
                                    MessageChain(
                                        chain=res.chain,
                                        type="tool_direct_result",
                                    )
                                )
                            except Exception as e:
                                logger.error(
                                    f"Tool 直接发送消息失败: {e}",
                                    exc_info=True,
                                )
                    yield None
            except asyncio.TimeoutError:
                raise Exception(
                    f"tool {tool.name} execution timeout after {tool_call_timeout or run_context.tool_call_timeout} seconds.",
                )
            except StopAsyncIteration:
                break

    @classmethod
    async def _execute_mcp(
        cls,
        tool: FunctionTool,
        run_context: ContextWrapper[AstrAgentContext],
        **tool_args,
    ):
        res = await tool.call(run_context, **tool_args)
        if not res:
            return
        yield res


async def call_local_llm_tool(
    context: ContextWrapper[AstrAgentContext],
    handler: T.Callable[
        ...,
        T.Awaitable[MessageEventResult | mcp.types.CallToolResult | str | None]
        | T.AsyncGenerator[MessageEventResult | CommandResult | str | None, None],
    ],
    method_name: str,
    *args,
    **kwargs,
) -> T.AsyncGenerator[T.Any, None]:
    """执行本地 LLM 工具的处理函数并处理其返回结果"""
    ready_to_call = None  # 一个协程或者异步生成器

    trace_ = None

    event = context.context.event

    try:
        if method_name == "run" or method_name == "decorator_handler":
            ready_to_call = handler(event, *args, **kwargs)
        elif method_name == "call":
            ready_to_call = handler(context, *args, **kwargs)
        else:
            raise ValueError(f"未知的方法名: {method_name}")
    except ValueError as e:
        raise Exception(f"Tool execution ValueError: {e}") from e
    except TypeError as e:
        # 获取函数的签名（包括类型），除了第一个 event/context 参数。
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            # 跳过第一个参数（event 或 context）
            if params:
                params = params[1:]

            param_strs = []
            for param in params:
                param_str = param.name
                if param.annotation != inspect.Parameter.empty:
                    # 获取类型注解的字符串表示
                    if isinstance(param.annotation, type):
                        type_str = param.annotation.__name__
                    else:
                        type_str = str(param.annotation)
                    param_str += f": {type_str}"
                if param.default != inspect.Parameter.empty:
                    param_str += f" = {param.default!r}"
                param_strs.append(param_str)

            handler_param_str = (
                ", ".join(param_strs) if param_strs else "(no additional parameters)"
            )
        except Exception:
            handler_param_str = "(unable to inspect signature)"

        raise Exception(
            f"Tool handler parameter mismatch, please check the handler definition. Handler parameters: {handler_param_str}"
        ) from e
    except Exception as e:
        trace_ = traceback.format_exc()
        raise Exception(f"Tool execution error: {e}. Traceback: {trace_}") from e

    if not ready_to_call:
        return

    if inspect.isasyncgen(ready_to_call):
        _has_yielded = False
        try:
            async for ret in ready_to_call:
                # 这里逐步执行异步生成器, 对于每个 yield 返回的 ret, 执行下面的代码
                # 返回值只能是 MessageEventResult 或者 None（无返回值）
                _has_yielded = True
                if isinstance(ret, MessageEventResult | CommandResult):
                    # 如果返回值是 MessageEventResult, 设置结果并继续
                    event.set_result(ret)
                    yield
                else:
                    # 如果返回值是 None, 则不设置结果并继续
                    # 继续执行后续阶段
                    yield ret
            if not _has_yielded:
                # 如果这个异步生成器没有执行到 yield 分支
                yield
        except Exception as e:
            logger.error(f"Previous Error: {trace_}")
            raise e
    elif inspect.iscoroutine(ready_to_call):
        # 如果只是一个协程, 直接执行
        ret = await ready_to_call
        if isinstance(ret, MessageEventResult | CommandResult):
            event.set_result(ret)
            yield
        else:
            yield ret
