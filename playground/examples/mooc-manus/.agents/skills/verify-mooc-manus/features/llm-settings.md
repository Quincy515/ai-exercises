# 模型提供商配置

用户读取、编辑并保存基础地址、模型名、温度和最大输出 token 数。密钥输入保持 password 与空值，只展示服务端是否已经配置。真实验证保留现有密钥，所有网络操作限于本地配置接口。

## Sub-features

- `llm`：四个共享布局入口的真实 GET、刷新、四字段编辑能力、未改动保存禁用，以及空密钥输入和配置状态 placeholder。
- `llm-save-validation`：温度输入 `3` 后点击保存，等待 Rust 范围校验错误，并确认零 POST。
- `llm-save-persist`：仅将 `max_tokens` 加 1；原值为 JS 最大安全整数时减 1，原值为 null 时取 1。保存后关闭重开，用真实 GET 核对四普通字段及 `api_key_configured`。
- `llm-save-restore`：通过 UI 恢复原始值，包含显式 null，关闭重开再次核对。异常补偿复用终态日志与并发检查。

## How to get to it (user POV)

首页 `/`、会话 `/sessions/1`、定时任务 `/schedules`、资料库 `/library` → “打开设置” → “模型提供商”。面板表单为 `#llm-config-form`；四个字段为 `#base_url`、`#model_name`、`#temperature`、`#max_tokens`，密钥字段为 `#api_key`。

底部“保存”提交当前模型配置，“取消”关闭设置。当前保存自动化覆盖首页入口，其余三个保存入口逐项报告为未验证；`llm` 单独覆盖四入口读取刷新。

## Driving it with verify-mooc-manus

前提：已经运行的开发后端 `GET http://localhost:5150/api/app_configs/llm` 返回四个 nullable 普通字段及布尔值 `api_key_configured`。doctor 只在选择 LLM 流程时检查此接口。

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features llm --port 4317
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features llm-save --allow-config-write true --port 4317
```

1. 选择“模型提供商”后等待“已读取服务器模型配置”，比对公开响应与四个可编辑字段；null 对应空字符串；温度按非空有限的 f32 数值比较，允许 `1e-7` 与 `0.0000001` 的等值表示，其他字段保持精确字符串与整数。密钥始终为空，类型为 password。
2. `api_key_configured=true` 时 placeholder 为“已配置，留空保留原密钥”；false 时为“填写新的 API 密钥”。未修改时“保存”禁用，点击“刷新”产生新 GET。
3. 写流程记录 `original-config.json` 的白名单公开配置与目标；温度 `3` 被 Rust 拒绝后恢复原温度，仅修改 `max_tokens`。
4. 保存前 GET 比对当前值仍等于原值。POST 仅允许同源精确 `/api/app_configs/llm`、恰好四个普通字段、值与本次目标一致；包含任何 `api_key` 字段（含空串/null）均阻止。
5. 等待真实 200 和“模型配置保存成功”；关闭重开后的 GET 必须与目标及原密钥配置状态一致。
6. 通过 UI 恢复原值，重开 GET 确认。`finally` 先核对所有 POST 已观察到完整响应：结果未知时标记失败并暂停恢复；结果已知时，原值保持、本次目标才补偿、第三方值或密钥状态变化时停止覆盖。

每条 `launch/run/drive` 命令选择 `llm-save` 都需要显式 `--allow-config-write true`。全部 GET/HEAD（含页面与静态资产）及获准 POST 均获取真实后端响应，关闭重定向/重试；3xx 直接拒绝，读取异常只记录固定安全原因。GET/HEAD 的 blocked 记录去除 URL 查询部分。其他写请求及跨源请求被阻止。验证环境精确中止同源 `GET /__tsd/console-pipe/sse` 开发辅助流，记录 `ignoredDevRequests` 与固定原因 `devtools stream excluded`，保持零上游请求。脚本完整保留原密钥，模型服务 API 调用保持未验证。

## Gotchas

- `max_tokens` 超出 JS 安全整数范围时记录 blocked，响应体和经过舍入的值均不进入证据。当前验证脚本在安全整数范围内执行，产品自身的大整数协议由 Rust/FFI 测试证明。
- LLM 响应只保存 `base_url/model_name/temperature/max_tokens/api_key_configured`；白名单丢弃旧响应可能包含的 `api_key` 及额外字段。异常 JSON 使用固定报错，原始响应保持内存读取。
- LLM 流程关闭 Playwright 原始 trace/ARIA 全量快照，避免响应体或 DOM 秘密落盘。证据由动作日志、公开响应 JSON、四字段 UI JSON、遮罩密钥输入的截图、POST 终态日志和清理结果组成；默认 Agent/会话/文件流程仍保留 trace。
- `post-outcomes.json` 与 `config-cleanup.json` 分别证明请求终态和恢复结果；超时、断网、3xx 或未知请求结果保持 outcome-unknown，原值证据供后续对账使用。
- 后端缺少版本号/CAS，写验证使用独占开发后端；`api_key_configured` 只能观测配置状态，无法识别“已配置密钥被另一把密钥替换”这一并发变化。脚本通过始终省略 `api_key` 保持服务端当时的密钥。
- 密钥替换、真实模型推理、保存失败 UI、保存中关闭保护、移动端和 Electron 属于独立验证范围。本技能保持借用后端的生命周期。

浏览器 context 关闭前护栏持续生效；关闭后等待已开始的 route handler 完成，再检查最终 blocked 请求、脚本错误和 POST 终态。清理期间迟到的业务错误会记录 failed；开发辅助流独立记录为 ignored。
