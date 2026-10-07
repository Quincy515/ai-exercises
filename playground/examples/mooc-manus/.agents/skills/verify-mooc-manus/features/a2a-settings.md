# A2A 远程 Agent 配置

用户读取远程 Agent 卡片，新增地址，启用、停用及删除配置。后端列表只返回卡片加载成功的服务，响应包含公开卡片属性与启用状态，地址保存在后端。

## Sub-features

- `a2a`：首页、会话、定时任务、资料库四入口的列表读取与刷新，核对 ID、名字、描述、输入/输出模态、能力与启用状态；关闭设置。
- `a2a-write`：首页入口的无效 URL 拒绝 → 新增本次 Card → 停用 → 启用 → 删除 → 独立 GET 核对。
- 本轮所有写操作只作用于本次随机 nonce 对应的测试项，原有服务的开关和删除按钮保持原样。

## How to get to it (user POV)

`/`、`/sessions/1`、`/schedules`、`/library` → “打开设置” → “A2A Agent配置”。列表名称为“远程Agent列表”，每项具有 `data-a2a-id`。

“新增远程Agent”打开“添加远程Agent”，填写 `#a2a_base_url` / “远程Agent地址” 后点“添加”。开关名称为“启用 <name>”。“删除 <name>”打开“删除远程Agent”，点“确认删除”。A2A 页底部“关闭”结束设置。

## Driving it with verify-mooc-manus

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features a2a --port 4317
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features a2a-write --allow-config-write true --port 4317
```

每条选择 `a2a-write` 的 launch/run/drive 命令必须显式写授权，默认三流程保持原样。业务 HTTP 真实访问现有 5150 后端；脚本保持后端、数据库和 Docker 的生命周期。

1. **检查可达性前提**：`lsof` 必须确认 5150 仅有一个监听 PID；`ps` 确认其为本机绝对路径的 `server-cli`，工作目录为本 checkout 的 `server/`。容器、代理、多监听或身份无法确认时 blocked。创建前再核对 PID 与起始时间，避免把本机 loopback 地址注册到另一网络命名空间。
2. **保存基线与可恢复夹具**：读取 `GET /api/app_configs/a2a-servers` 的可见列表，生成随机 UUID nonce，在 127.0.0.1 随机端口提供 `/<nonce>/.well-known/agent-card.json`。name、description 均含 nonce；URL、Card 和端口先写入 `fixture.json`。夹具只提供卡片，其他路径和全部 POST 都拒绝。
3. **校验输入**：新增框输入 `not-an-http-url`，提交后看到 Rust URL 错误，POST 计数保持不变。
4. **新增一次**：只允许 `POST /api/app_configs/a2a-servers` 的单字段 `{base_url: 本次夹具URL}`；真实响应为 JSON null，随后等待 UI 自动 GET。用 name、description 完全相等且 ID 不在基线内的唯一记录确认所有权，保存 `owned-record.json`。
5. **启停与删除**：仅为确认过的 owned UUID 授权 `POST /{id}/enabled` 的精确 `{enabled}`；删除仅允许 `POST /{id}/delete` 空 body。每次操作都要求完整 200/null 响应和后续 GET，确认测试项状态及其他可见项保持原样。
6. **成功清理**：删除成功后独立 GET 确认测试项消失，前后可见列表相同；关闭本次 Card 夹具。结果只声明 `visibleListRestored`，保持数据库完整配置验证为未验证。

`a2a-write` 自动化只覆盖首页写入口，其余三个写入口逐项列为未验证。浏览器跨源请求被阻止；所有读取和写入复用禁重定向的真实响应转发、开发辅助 SSE 隔离与 context 关闭后的最终判定。报告的 `mocks: true` / `mockScope` 只指明外部 Agent Card 是本地受控夹具，配置 API 和 UI 保持真实。

## Gotchas

- 后端 GET 会尝试加载已配置服务的 Agent Card。现有服务卡片超时可能导致列表检查 blocked/failed；卡片不可见也可能表示配置仍存在，不能依据缺项推断数据库已删除。
- 新增响应不返回 ID。新增结果未知或测试项不可见时，流程最多做三次只读 GET 定位，**保持创建仅一次**；ID 始终来自唯一身份匹配，原列表 ID 永远禁止写。
- 异常路径保留 `fixture.json`、`baseline-visible-list.json`、`owned-record.json`（已确认时）、`post-outcomes.json`、`config-cleanup.json` 与 `fixture-requests.jsonl`。脚本关闭本次夹具；实际异常写清理由负责恢复的代理接管。
- 恢复时先在原端口重启同一 Card：

```sh
node .agents/skills/verify-mooc-manus/scripts/a2a-card-fixture.mjs serve --fixture /absolute/run/a2a-write/fixture.json
```

`result.json` 提供本次绝对路径的完整 `recoveryCommand`。原端口被占用时明确失败；保持 nonce、地址、Card 完全一致。确认后端请求已经结束，再用 GET 核对唯一身份；仅删除明确 owned ID，完成后停止这个夹具（Ctrl+C）。结果未知、ID 冲突或无法定位时继续保留证据，禁止重复创建、猜测 ID 或操作原有项。

- `fixture-lifecycle.jsonl` 记录夹具启动/停止；`fixture-requests.jsonl` 仅记录方法、路径和是否提供 Card。测试过程保持远程消息调用为未验证，夹具拒绝 invocation。
- `api-responses.json` 保存公开列表白名单和 200/null 写响应；trace/截图用于证明 UI 动作，`post-outcomes.json` 证明写请求终态。浏览器侧与后端读取外部 Card 是两个网络边界，需要分别报告。

首次确认的新 ID 在整个流程中固定，`owned-record.json` 只写入一次。正常复查和异常恢复发现 ID 变化时，保留原记录并写 `ownership-conflict.json`，停止写入供人工核对。
