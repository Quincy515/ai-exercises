# Agent 配置读取、编辑与保存

用户从通用配置读取三个 Agent 参数，编辑后保存到服务器；输入校验、保存状态和错误由 Rust 管理。配置是开发后端全局设置，写验证期间使用独占开发后端并暂停其他设置编辑。

## Sub-features

- `settings-open/read/refresh/close`：四个共享布局入口读取、刷新、关闭；三个字段可编辑，未修改时“保存”禁用。默认 `settings` 保持纯读取验证。
- `settings-save-validation`：最大迭代次数填 `0` 后点击保存，看到 Rust 校验错误，网络写请求计数保持不变。
- `settings-save-persist`：合法修改最大迭代次数，真实 POST 返回 200，显示“保存成功”，关闭重开后用新 GET 确认目标值。
- `settings-save-restore`：通过 UI 保存原值，再关闭重开 GET 确认恢复；异常分支执行受限补偿恢复并记录结果。
- `settings-save-failure/protection`：保存失败保留草稿、在途阻止编辑/关闭/切换、撤销草稿。属于单独的故障或交互验证；当前自动保存流程将它们记录为未验证。

## How to get to it (user POV)

- 首页 `/`、会话 `/sessions/1`、定时任务 `/schedules`、资料库 `/library` 都有全局“打开设置”按钮。
- 打开“MoocManus 设置”后选“通用配置”；输入“最大迭代次数”“最大重试次数”“最大搜索结果数”，点击底部“保存”。
- noVNC 全屏页使用独立布局。其他 LLM/MCP/A2A 设置的更新另行接入。

## Driving it with verify-mooc-manus

前提：doctor 的 `eligible.settings` 为 true；现有 `GET /api/app_configs/agent` 返回三项整数。保存范围为 `max_iterations: 1..999`、`max_retries: 2..9`、`max_search_results: 2..29`。

默认纯读取流程：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features settings
```

四个页面逐一进入，点击 `button[name="打开设置"]`，Dialog 内点 `button[name="通用配置"]`；等待“已读取服务器配置”，比对 `#max_iterations`、`#max_retries`、`#max_search_results` 与真实 GET。三个 input 可编辑，“保存”禁用。点击“刷新”观察新的成功 GET 并再次比对，点“取消”关闭。证据完整覆盖四入口。

显式写验证（仅借用已启动的独占开发后端）：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features settings-save --allow-config-write true --port 4317
```

`launch/run/drive` 选择 `settings-save` 时，每条命令均需显式传入 `--allow-config-write true`；权限不会从先前启动自动继承。缺少授权直接拒绝并保持后端原样。其他流程的 API 写请求继续被阻止。

1. 从首页打开配置，在写入前保存 `original-config.json`，记录原始三值与目标；目标只将最大迭代次数加 1，原值为 999 时减 1。
2. 输入 `0` 点保存，等待匹配“最大迭代次数”和 `1..999` 的 Rust 错误；确认没有 POST，保留校验截图。
3. 输入合法目标。保存前用 GET 确认当前值仍等于原值；只允许同源精确路径 `POST /api/app_configs/agent`、JSON 中三个合法整数且与本次目标完全一致的一次写请求。浏览器 route 调用真实后端时关闭重定向和重试；3xx 中止并记录未知结果，成功响应原样透传。
4. 等待真实 200 和“保存成功”；关闭重开，从第二次 GET 确认持久化，保存截图与响应。
5. 保存前再 GET 核对当前值仍为本次目标，通过 UI 恢复原始值，关闭重开后 GET 确认。
6. `finally` 先确认全部授权 POST 已观察到完整响应。超时/断网或响应体未结束会持久标记 `outcome-unknown/failed`，保留 `post-outcomes.json` 与原值证据，停止自动补偿及恢复成功声明；即时 GET 可能发生在迟到提交之前。完整响应已知时再读取当前值：等于原值只记录已恢复；等于测试目标才通过同一权限策略补偿 POST 原值并 GET 验证；遇到第三方值停止覆盖，记录 `cleanupError` 和原始值证据。

`settings-save` 自动化完整覆盖首页保存入口；会话、定时任务、资料库三个保存入口逐一列为 `unverifiedEntryPoints`。`settings` 四入口读取的成功可单独报告，保存成功声明限于首页路径。

## Gotchas

- 参数随环境变化，比较本次响应和原始快照。写流程要求三个值都满足真实后端范围；超出范围先报告，保持原配置。
- i64 浏览器展示为字符串；doctor 使用 safe integers，超出范围需要无损解析后再验证。
- 写流程临时修改真实开发配置。后端当前缺少版本号/CAS，GET 与 POST 之间存在并发窗口，使用独占后端保证这一验收前提；出现第三方值时先处理冲突。
- `config-cleanup.json` 记录配置恢复；外层 `cleanup.json` 只说明本次 Vite 实例已清理。两份结果都需检查，配置恢复失败时保留证据并报告。
- 被强制终止可能跳过 `finally`。保留 `original-config.json` 和 `post-outcomes.json`，先核对后端请求执行结果；完整写结果已知后，读取当前值比对，仅在仍等于本次目标时写回原值。结果未知期间的单次 GET 无法排除迟到提交。
- 后端不可用记录 blocked。脚本保持借用后端与数据库原有生命周期；真实请求失败保留截图、日志与 trace。
- 网络故障模拟、LLM/MCP/A2A 保存、移动端和 Electron 安装包属于独立验证范围。
