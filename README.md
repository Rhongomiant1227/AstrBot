# AstrBot Fork: Noel / sub2api / MCP Edition

这个仓库是基于 [AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot) 的个人定制 fork，内容来自当前 NAS 中实际运行的一版 AstrBot 核心代码快照。

目标不是替代上游文档，而是保留这一版已经跑起来的定制能力，方便之后继续维护、回滚和二次开发。

## 当前这版包含什么

1. `sub2api` 后端适配层
- 增加了面向 `sub2api` 的工具调用兼容层。
- 支持把工具调用包装成适配格式，并把工具结果重新桥接回模型。
- 对一些上游返回异常格式做了兼容处理。

2. OpenAI Responses / SSE 兼容
- 修过 `response.created` / `response.failed` / `response.output_text.delta` 这类 Responses SSE 文本回包解析。
- 用于兼容一些代理或上游在非标准流式/半流式场景下返回整段事件流文本的情况。

3. MCP 与常用工具链兼容
- 这版核心围绕 MCP 做过适配，重点处理过实时工具、研究类工具和天气能力。
- `super_noel` 侧保留了有限工具白名单，避免把所有工具无差别暴露给高思考子代理。

4. 双重人格 / 双层代理
- 主人格：`南条酱`
- 子代理人格：`超级南条酱`
- 核心内实现了 `transfer_to_super_noel`、可见接管、短时 sticky session、重复切换抑制、里人格正文纯文本净化等逻辑。
- 普通闲聊、普通看图吹水默认由表人格处理；高精度、多步推理、严格规则判断、复杂图题则更倾向交给 `超级南条酱`。

5. 人格配置保留
- 当前仓库保留了这版的 Noel 双人格 prompt 文档快照。
- 见 [docs/personas/noel_personas_zh.md](docs/personas/noel_personas_zh.md)
- 同时保留了子代理路由快照：
  - [docs/personas/subagent_router_snapshot.json](docs/personas/subagent_router_snapshot.json)

## 这个 fork 不包含什么

为了避免泄露运行环境敏感信息，这个仓库不包含：

- API Key
- Provider 实际密钥
- NAS / Docker 私有配置
- 聊天数据库
- 聊天日志
- 其他敏感运行数据

也就是说，这个仓库更偏向“代码与人格快照”，而不是一份可直接启动的完整私有运行时镜像。

## 主要改动位置

- [astrbot/core/provider/sources/openai_source.py](astrbot/core/provider/sources/openai_source.py)
- [astrbot/core/astr_agent_tool_exec.py](astrbot/core/astr_agent_tool_exec.py)
- [astrbot/core/astr_main_agent.py](astrbot/core/astr_main_agent.py)
- [astrbot/core/super_noel_session.py](astrbot/core/super_noel_session.py)

## 适合拿来做什么

- 作为当前魔改版 AstrBot 的代码备份
- 作为后续继续补丁开发的基线
- 作为对比上游更新时的冲突参考
- 作为 Noel 双人格和 sub2api 适配思路的实现存档

## 说明

如果你只是想用官方稳定版，请优先看上游项目：
- 上游仓库：https://github.com/AstrBotDevs/AstrBot
- 官方文档：https://astrbot.app/

这个 fork 更适合已经知道自己在做什么、并且就是想复用这套 Noel / sub2api / MCP / 双人格定制逻辑的人。
