# MCP 服务器配置

用户读取服务器与工具名称，使用 JSON 添加或完整更新同名配置，启用、停用和删除服务器。GET 只返回公开列表；三个 POST 都返回完整配置，其中可能包含密钥，因此验证证据只保留安全元信息。

## Sub-features

- `mcp`：首页、会话、定时任务、资料库四入口的真实列表读取刷新，核对名字、transport、enabled 和 tools。
- `mcp-write`：首页无效 JSON 拒绝 → 新增唯一测试名 → 同名更新 → 停用 → 启用 → 删除 → 独立 GET 确认。
- 写流程只操作本次 UUID nonce 派生的 `verification-mcp-<nonce>`，保持基线 names 原样。批量多项写入、stdio 命令执行和真实工具调用分别保持未验证。

## How to get to it

`/`、`/sessions/1`、`/schedules`、`/library` → “打开设置” → “MCP 服务器”。列表为 `aria-label="MCP服务器列表"`，条目具有 `data-mcp-name` 和 `data-mcp-transport`。

“添加配置”打开“添加或更新 MCP 服务器”；字段 label 为“MCP服务器配置”，按钮为“保存配置”。开关为“启用 <name>”；“删除 <name>”打开“删除 MCP 服务器”，点击“确认删除”；父设置页使用“关闭”结束。

## Driving it

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features mcp --port 4317
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features mcp-write --allow-config-write true --port 4317
```

每条包含 `mcp-write` 的 launch/run/drive 都需要本次显式写授权。真实验证借用现有 5150 后端；本技能只管理前端和自己创建的本地协议夹具。

1. doctor 核对公开列表契约；写流程另核对唯一 5150 监听进程为本机本 checkout 的绝对路径 `server-cli`，创建前再次比对 PID 和起始时间。
2. 记录安全列表基线，在 loopback 随机端口启动无状态 Streamable HTTP 夹具。`fixture.json` 仅含 schema、nonce、port、serverName、toolName；使用它在内存重建本次 URL。
3. 输入无效 JSON，等待 Rust 错误并确认零 POST。正常新增只授权单项 `mcpServers`：本次 name、`streamable_http`、固定 loopback URL、enabled=true 和本次 description。env/headers/args/command 只接受缺省或 null。
4. 创建只尝试一次。读取完整 POST 响应时，仅在内存核对本次配置 URL 指纹、transport、enabled、description 与敏感字段为空；原有项仅提取 name/transport/enabled，立即丢弃完整对象。首次确认后写 `owned-record.json` 的本次 name、transport 和指纹。
5. 通过“添加配置”再次提交同名测试项，只改变受控 description，证明同名完整更新成功。后续启停仅允许精确 owned name 和预期 enabled；删除要求精确地址与空 body。每次 POST 后核对自动 GET，原有项公开元信息保持一致。
6. 删除成功后独立 GET 与基线比较，报告 `visibleListRestored`；关闭夹具并记录 `fixtureStopped`。完整配置的秘密字段一致性保持未验证。

浏览器全部读取与获准写入采用真实响应转发，关闭重定向与重试，跨源请求被阻止。开发工具 SSE 继续使用已有精确排除与最终路由收尾检查。`mocks: true` 只表示外部 MCP 协议服务为本地受控夹具，配置 API 和 UI 保持真实。

## Evidence and recovery

- MCP 流程关闭原始 trace 和全量 ARIA。截图遮罩 JSON 输入框；UI JSON 只含 name/transport/enabled。写请求日志也只记录安全元信息，完整配置、env、headers、args、command、url 均不落盘。
- `api-responses.json`：GET 四个公开字段；POST 仅 name/transport/enabled。`post-outcomes.json`：授权动作和传输终态。`owned-record.json`：首次确认的 name 与不可逆 URL 指纹。
- `fixture.json` 与 `fixture-lifecycle.jsonl`：可恢复的本地身份和端口；`fixture-requests.jsonl`：固定操作名及接受/拒绝结果，省略参数和请求体。
- 超时、断网、跳转或确认失败时保持失败，创建最多一次，禁止自动重试或补偿。异常路径最多做一次只读列表定位，保存 `config-cleanup.json`，关闭夹具；结果未知始终要求人工核对。

恢复夹具：

```sh
node .agents/skills/verify-mooc-manus/scripts/mcp-http-fixture.mjs serve --fixture /absolute/run/mcp-write/fixture.json
```

`result.json` 提供本次完整命令。原端口占用时明确失败。确认后端写请求结束后，由负责恢复的代理核对 `owned-record.json`、基线和本次配置指纹，再仅删除已证明的测试 name；仅从公开列表无法证明配置 URL，未确认所有权时应保留证据进一步核对，保持创建一次。结束后 Ctrl+C 关闭恢复夹具。

## Boundaries

- 后端 GET 会连接当前已启用的 MCP 服务并查询工具；既有 stdio 项可能启动对应命令，既有 HTTP 项可能访问远程地址。验证前使用受控开发后端并确认这些既有项已获准探测。脚本自身只创建 loopback HTTP 项，拒绝新增 stdio、凭据和外部地址。
- 夹具只支持 initialize、notifications/initialized 和 tools/list，提供一个声明为只读的工具。tools/call 始终拒绝，测试结果仅证明配置接入和工具发现。
- 后端没有版本号/CAS。使用独占开发后端；同名配置被并发替换时公开 GET 无法证明完整配置身份。写响应的本次 URL 校验、唯一 nonce 和基线 names 护栏限定此次验证范围。
- `driveMcp`、协议夹具及 policy 函数均导出供独立 Electron 验证复用；本技能默认报告 Web，桌面结果单独记录。

纯策略与临时协议夹具自测：

```sh
node --test .agents/skills/verify-mooc-manus/scripts/mcp.test.mjs
```
