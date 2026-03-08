# AstrBot 定制版交接文档

更新时间：2026-03-08

## 范围
- 部署主机：`192.168.114.183`
- 运行容器：`astrbot`
- 主要目标：把 `sub2api` 做成稳定可用的后端模式，并在核心层完成工具适配、人格切换、输出清洗与兼容性修复。

## 当前状态
- `sub2api` 已作为独立模式接入核心层，不再只是零散补丁。
- `super_noel` 已作为高难问题接管路径存在，并带有粘性会话机制。
- `Responses API / sub2api` 路径已补过重试、超时和解析容错，整体稳定性比最初版本高很多。
- 最近一次 live 部署已直接打到 NAS 容器内，并已完成回归验证。

## 已完成的核心改动

### 1. sub2api 核心模式与工具适配
涉及文件：
- `astrbot/core/provider/sources/openai_source.py`
- `astrbot/core/astr_main_agent.py`
- `astrbot/core/config/default.py`

主要内容：
- 为 `sub2api` 增加独立识别与配置开关。
- 在 provider 层加入 tool adapter 逻辑，使模型即使不走原生标准工具调用，也能通过约定格式完成工具规划、调用和结果回填。
- 工具调用解析支持：
  - `<astrbot_tool_call>`
  - `<tool_call>`
  - `<function_call>`
  - fenced JSON
  - 嵌套 function payload
  - 噪声转义容错
- 只删除已成功解析成合法工具调用的标签块，避免误删正文。
- 在主 agent 的 modality/能力判断中，为 `sub2api + adapter` 保留虚拟工具使用能力。
- 增加前端可见配置项与默认值，使该模式能从配置层调用，而不是只能靠手改。

### 2. super_noel 接管链路
涉及文件：
- `astrbot/core/astr_agent_tool_exec.py`
- `astrbot/core/super_noel_session.py`
- `astrbot/core/provider/sources/openai_source.py`

主要内容：
- 引入 `transfer_to_super_noel` 的接管链路，用于复杂问题转交高思考强度模型处理。
- 普通南条酱与超级南条酱分离：
  - 普通南条酱：`GPT-5.4 medium`
  - 超级南条酱：`GPT-5.4 xhigh`
- 加入 sticky session 机制：里人格接管后可在短时间内持续停留，避免刚出现就立刻退回表人格。
- 前台人格知道里人格存在，但不会用“前台/后台/表人格/里人格”这种系统术语去自我解释。
- provider 层对复杂推理类问题增加更强的 handoff 偏置，使复杂题更积极转交 `super_noel`。

### 3. 上游稳定性与超时修复
涉及文件：
- `astrbot/core/astr_agent_tool_exec.py`
- `astrbot/core/provider/sources/openai_source.py`

主要内容：
- 给 `super_noel` 的主线路和降级线路都补了 wall-clock timeout 包裹，防止长时间卡死。
- 关闭 `responses/sub2api` 路径里重复叠加的 SDK 重试，减少一次失败被放大成多次阻塞。
- 修复 `max_retries = 1` 时可能把实际成功响应误判为失败的问题。
- 优化失败文案，让失败时会给出更清晰的重试/降级说明。

### 4. 里人格输出清洗
涉及文件：
- `astrbot/core/astr_agent_tool_exec.py`

主要内容：
- 为 super_noel 增加冗余开场裁剪，解决这类问题：
  - `啧，废柴南条酱先歇着，这题我来给你理清。`
  - `算了，还是我来。这个命题不成立……`
- 新增 `_trim_leading_super_noel_opening_sentences()`，进一步清理“正文第一句其实还是开场白”的情况。
- super_noel 的接管开场句改为单独发送，而不是再和正文强行拼成一条。
- 当前 QQ 侧的预期表现是三段式：
  1. 前台犯困/顶号前提示
  2. 里人格接管开场
  3. 正式正文

### 5. 数学/公式文本清洗
涉及文件：
- `astrbot/core/provider/sources/openai_source.py`

主要内容：
- 新增 `_plainify_common_latex_markup()`，把常见 LaTeX/公式残留压成普通聊天文本。
- 已接入 Responses API 和 Chat Completions 两条解析链。
- 当前重点覆盖：
  - `\( ... \)` / `\[ ... \]`
  - `\text{...}` / `\mathrm{...}` / `\operatorname{...}` / `\mathbf{...}` 等
  - `\Rightarrow` / `\Longrightarrow` / `\implies`
  - `\to` / `\rightarrow` / `\mapsto`
  - `\le` / `\ge` / `\neq` / `\approx` / `\infty` / `\pm` / `\sqrt`
- 目标不是完整 TeX 渲染，而是避免 QQ 里直接看到一堆反斜杠和 `\text{}`。

### 6. 已做的收尾输出约束
涉及文件：
- `astrbot/core/provider/sources/openai_source.py`
- `astrbot/core/astr_agent_tool_exec.py`

主要内容：
- 裁掉已经回答完成后多余的尾巴，例如：
  - `如果你愿意，我可以继续……`
  - `你要哪个？`
  - `要不要我再帮你……`
- 同时清理重复开场、Markdown 代码围栏、部分格式残留，减少 QQ 侧观感问题。

## 最近一次修复对应的问题
本次同步进仓库的修复，主要对应以下用户可见问题：
- 里人格回复中出现重复的两层开场，语义重复。
- 接管开场和正文挤在同一条消息里，观感别扭。
- 数学解释会吐出 `\(H\)`、`\text{清一色}`、`\Rightarrow` 这种原始 LaTeX。
- 回答已经结束后还继续追问“要不要我继续写”“你要哪个”。

## 验证结果
### 本地验证
- `py_compile` 通过。
- 字符串级回归通过，覆盖：
  - 里人格冗余开场裁剪
  - LaTeX -> 普通文本
  - 尾部邀约裁剪

### 远端验证
- 改动已直接部署到 NAS 容器 `astrbot`。
- 部署后核对了容器内文件 hash 与本地一致。
- 容器已重启并恢复在线。
- 容器内回归结果：`REMOTE_REGRESSION_OK`

## 部署方式
本机到 NAS 的可靠流程：
1. 先把文件传到宿主机临时路径。
2. 再使用 `docker cp` 覆盖进 `astrbot` 容器。
3. 重启容器。
4. 在容器内执行最小回归脚本确认行为。

说明：
- 这台 NAS 上 `sudo + docker exec -i + stdin` 的组合不稳定，容易卡住。
- `docker cp` 方案更稳定，后续继续沿用即可。

## Git 记录
最近两次与输出清洗相关的提交：
- `db911b11` `Tighten Noel handoff output cleanup`
- `dcb9ca34` `Refine Noel handoff cleanup and plainify math`

## 后续维护建议
- 如果继续调整人格话术，优先改核心层的清洗与接管桥接，不要回退到 prompt 堆规则。
- 如果未来还会出现新的公式残留，优先扩展 `_plainify_common_latex_markup()`，不要在插件层做分散修补。
- 如果 QQ 侧仍然出现格式异常，先查：
  - provider 最终文本
  - handoff bridge 文案
  - 分段发送插件/核心分段是否叠加
- 如果以后要同步更多定制说明，可以继续放在 `docs/handover/` 下，不要再依赖仓库外那份单独文档。


## 2026-03-08 ??????????????
?????
- `astrbot/core/provider/sources/openai_source.py`
- `astrbot/core/super_noel_session.py`

?????
- ??????????????? `/save` ?????? `/` ??? `?` ??????
- ??????? handoff ?????????????? `/save`?`/xxx` ??????? super_noel ??????
- ? sticky session ?????????????????? sticky ???????????????????????????

???
- ???????`/save`?`/save xxx`?`?save xxx`
- ?????????`REMOTE_COMMAND_GUARD_OK`


## 2026-03-08 ????? command handler ??????? LLM
?????
- `astrbot/core/pipeline/waking_check/stage.py`
- `astrbot/core/pipeline/process_stage/stage.py`

?????
- ??????? `matched_command_handler` ?????????? `CommandFilter` ? `CommandGroupFilter`????????
- ????????????
  - ?? `matched_command_handler = true`
  - ????? `command_allow_default_llm`
  - ?????????? LLM ??
- ?????
  - ???????????????
  - ??????? super_noel ??????
- ??????????? super_noel sticky????????????????????

???
- ????????? `/save` ?????????????? command handler??????????????
- ??????? wake ??????????????? command filter??? LLM ???????

???
- ?????`PROCESS_COMMAND_FALLBACK_GUARD_OK`
- ???????`REMOTE_COMMAND_PIPELINE_GUARD_OK`
