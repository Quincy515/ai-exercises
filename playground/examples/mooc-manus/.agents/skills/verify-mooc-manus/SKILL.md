---
name: verify-mooc-manus
description: 验证 MoocManus 的 Web 用户流程：Agent 与模型提供商配置读取刷新及显式授权的编辑保存、会话导航与计划展开、任务文件列表。需要复现界面行为、检查 Crux 接入或保存真实操作证据时使用；明确区分真实 API、演示 UI 和未实现功能。
---

# Verify MoocManus

从项目根目录执行。先读 `apps/AGENTS.md`、`apps/ARCHITECTURE.md` 和 [feature map](features/README.md)。主验证面是 Web SPA；Electron 复用 UI，但本脚本的 Web 结果不代表桌面安装包验证。

## Launch

当前脚本在 macOS 验证，依赖 Node、pnpm、just、ps、lsof，以及已有 Playwright 和 Chrome。前端使用现有 `apps/Justfile` 与 pnpm。准备阶段在当前 checkout 的开发进程停止时执行：

```sh
cd apps
just install
just test
pnpm test:shared
pnpm test:crux
cd ..
```

`just install` 更新 WASM/生成类型并清 Vite 缓存。生成目录、pnpm 依赖与 Vite 缓存为同一 checkout 共享；验证期间保持源码稳定，一次只运行一个本技能实例，不与其他代理同时重建/驱动该 checkout。不同 checkout 可使用不同端口。脚本拒绝占用端口和仍存活的技能实例。

一键执行三流程并自动清理：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features settings,sessions,files --port 4317
```

脚本先核对 `apps/tanstack-app/package.json` 的 `scripts.dev` 为 `vite dev --port 3000`；未知启动命令明确失败。从该包解析已有的本地 `vite/bin/vite.js`，以当前 Node (`process.execPath`) 直接执行 `vite.js dev --host 127.0.0.1 --port 4317 --strictPort`，工作目录为 `apps/tanstack-app`，自动复用 `vite.config.ts`。直接启动让 Vite 与子进程归属本次进程组，避免 pnpm 12 脚本子 shell 的额外进程组；`instance.json` 记录实际命令、工作目录与原始脚本。环境仅覆盖 `VITE_API_BASE_URL=` 以固定同源 `/api`。监听进程必须属于本次进程组，首页返回 `Mooc Manus` 且源码/生成包指纹一致才算 ready。端口可用 `--port` 调整；保持后端代理目标 `localhost:5150`。

验证真实保存与恢复时显式选择写流程：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features settings-save --allow-config-write true --port 4317
```

该流程临时改变全局 Agent 配置，并通过 UI 恢复原值；使用独占开发后端，保存期间暂停其他配置编辑。`launch/run/drive` 每次选择 `settings-save` 都必须显式传入 `--allow-config-write true`，权限保持逐命令授权；缺少授权提前拒绝。后端无 CAS，GET 比对与写入之间仍有并发窗口。

模型提供商验证独立选择，默认三流程保持原样：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features llm --port 4317
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features llm-save --allow-config-write true --port 4317
```

LLM 写验证只发送四个普通字段并省略 `api_key`，临时改变 `max_tokens` 后恢复（包含 null）；每条选择 `llm-save` 的 launch/run/drive 都要求本次显式写授权。该流程仅访问本地配置 API，跨源请求被阻止。超出 JS 安全整数的 `max_tokens` 提前 blocked。

需要分步操作时：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs launch --port 4317 --run output/playwright/verify-mooc-manus/manual
node .agents/skills/verify-mooc-manus/scripts/verify.mjs doctor --run output/playwright/verify-mooc-manus/manual
node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features sessions,files
node .agents/skills/verify-mooc-manus/scripts/verify.mjs cleanup --run output/playwright/verify-mooc-manus/manual
```

每次启动选择新的 `--run` 目录，路径必须是 `output/playwright/verify-mooc-manus/` 的直接子目录；启动、doctor、drive 和 cleanup 统一拒绝多级子目录。脚本保留已有证据并拒绝覆盖。省略 `--run` 时生成带时间戳的目录，命令结果给出绝对路径。

后端启动方式来自 `server/README.md`：依赖 PostgreSQL/Redis 就绪后，在 `server/` 执行 `cargo loco start`。就绪条件是 `GET http://localhost:5150/api/app_configs/agent` 返回三个整数；当前查询无需登录和 seed。后端会自动迁移数据库，默认流程只读借用已运行的开发后端，显式 `settings-save` 按下述策略保存并恢复，**不自动启动/重置数据库、Docker 容器或后端，也不结束借用进程**。缺少后端时，配置流程记录 blocked；会话/文件的现有 UI 流程可以独立验证。

Electron 的现有启动命令为 `pnpm --filter electron-app start`（在 `apps/`）；本技能不自动接管 Electron 或用户浏览器。需要桌面证明时独立隔离实例、profile 和调试端口，并按相同 map 驱动，单独报告结果。

## Doctor

`verify.mjs doctor --run <本次目录>` 是只读检查：核对 PID 起始时间、监听端口所属进程组、首页标识、源码/生成物 SHA-256、现有 Playwright 安装与后端契约。它不改运行状态、不操作页面、不调用写 API。

返回 `ok: true` 表示 Web 实例可驱动；`eligible.settings: false` 表示 Agent 后端条件不满足。选择 LLM 流程时额外返回 `llmBackend` 和 `eligible.llm`；响应只保存五个公开字段。每次有异常先执行 doctor；源码变化后启动新的 run，保留旧证明。doctor 输出可重定向到证据目录，但重定向本身属于调用者写证据。

## Drive

重复流程由 `verify.mjs drive --features settings,sessions,files` 执行，具体入口与判断标准见：

- [Agent 配置](features/agent-settings.md)：四个共享布局入口的真实 GET、刷新、可编辑字段和未修改时禁用保存；显式 `settings-save` 另验首页真实保存与恢复。
- [模型提供商](features/llm-settings.md)：`llm` 四入口安全读取刷新；`llm-save` 首页修改普通字段并恢复，始终保留原密钥。
- [会话与计划](features/session-plan.md)：侧栏、深链接刷新、计划展开收起与返回首页。
- [任务文件](features/task-files.md)：Header 文件按钮、六项列表与关闭恢复。

每个流程使用新 browser context，1280×900，临时浏览器资料由 Playwright 管理。仅使用页面导航、ARIA 控件、用户点击与 DOM 读取；不调用 Core 内部方法或设置组件状态。默认流程中的 `/api/` 写请求会被阻止并判失败，GET 请求真实发送。显式 `settings-save` 仅允许同源精确 `POST /api/app_configs/agent`，请求体须为符合范围且等于本次预期值的三个整数；一次点击授权一次写入。获准 POST 由浏览器 route 层调用真实后端，`route.fetch({ maxRedirects: 0, maxRetries: 0, timeout: 10000 })` 获取真实响应；3xx 记录 outcome-unknown 并中止，其他响应通过 `route.fulfill({ response })` 保留原始状态、响应头与内容。该传输路径使用真实后端，结果仍为 `mocks: false`，报告另注明 `configWriteTransport`。默认 Agent GET 流程继续直接发送；异常补偿也使用相同权限与禁止跳转策略。LLM 流程的全部 GET/HEAD（包含页面和静态资产）也通过 `route.fetch` 获取真实响应，关闭跳转和重试，3xx 拒绝后记录脱敏 blocked 信息；传输失败仅记录固定安全原因。各 LLM 结果的 `readTransport` 明确这一验证传输方式。仅在 LLM 验证环境中，同源 `GET /__tsd/console-pipe/sse` 被精确中止并记录到 `ignoredDevRequests`（`devtools stream excluded`）；这个开发工具辅助流保持零上游请求，其他路径和业务请求继续严格检查。

Playwright 优先使用 `MOOC_PLAYWRIGHT_MODULE`、项目已有安装或当前用户的 gstack 安装。当前主机可用 `~/.agents/skills/gstack/node_modules/playwright`；默认 Chrome channel。也可显式指定：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features sessions,files --playwright-module "$HOME/.agents/skills/gstack/node_modules/playwright" --channel chrome
```

不自动下载浏览器或增加项目依赖。缺少运行时应报告准确前提条件。退出码：`0` 所选流程全部通过，`2` 存在 blocked 且无失败，`1` 存在失败或实例/工具错误；读取 JSON 中逐流程结论。

## Evidence

证据位于 `output/playwright/verify-mooc-manus/<run-id>/`：

- `working-tree.json`：Git HEAD、dirty 状态、参与执行的源码、技能与生成包文件哈希，含未提交源码；环境文件仅记录哈希。
- `instance.json`、`server.log`、`launch-doctor.json`、`drive-doctor.json`：启动参数、归属、服务日志、健康与构建检查。
- `<feature>/actions.jsonl`、前后截图、`.aria.txt`、`trace.zip`：动作及结果，Playwright trace 可用已安装 CLI 打开。
- `results.json`、各流程 `result.json`：入口覆盖、断言、API 请求状态、页面错误、未实现项；配置成功时保存三项公开值。
- `settings-save/original-config.json`、`api-responses.json`、`config-cleanup.json`：写入前原值/目标、真实读写响应与配置恢复结果。
- `settings-save/post-outcomes.json`：每个获准 POST 的请求体、时间、HTTP 状态和完整响应终态；超时/断网保留 `outcome-unknown`，迟到响应另外记录。
- LLM 流程保存 `llm/` 或 `llm-save/` 下的白名单响应、四字段 UI JSON 与遮罩截图；关闭原始网络 trace 和完整 ARIA。LLM 原值、POST 终态及恢复结果使用相同文件名，`api_key` 永远省略。
- `cleanup.json`：进程已停、端口释放、证据文件仍存在；配置恢复单独检查 `config-cleanup.json`。

证明必须同时包含操作与结果。截图将有限过渡动画推进到结束状态，保留清晰、稳定的页面证据。配置值与本次浏览器 GET 响应比对；只读流程报告观测到的请求及写请求 guard，不声称检查了数据库全量状态。会话和文件是内置演示数据，点击路由/计划/Dialog 的通过仅证明真实 UI 交互。写流程保存后关闭重开设置，用新的 GET 核对持久化，再经 UI 恢复原值。`settings-save` 的通过限于首页保存路径，其他三个保存入口和 Electron 单独列为未验证。

Mock 只在明确的外部系统边界且单独标记的故障测试中使用；当前默认流程 `mocks: false`。后端不可用的 blocked、未接通的聊天/下载、未测的 Electron 或移动视口，都不能由其他成功入口代替。

## Cleanup

`run` 在成功与失败后自动关闭浏览器并结束本次创建的 Vite 进程组；分步启动后执行 `cleanup --run ...`。如果 drive 失败，先完成该次 cleanup 再重试。脚本按 PID 起始时间和进程组验证归属，归属变化时拒绝终止；禁止按进程名批量结束。

`settings-save` 在 `finally` 先核对全部授权 POST 的终态；超时、断网或仅收到响应头时记录 `outcome-unknown/failed`，保留原值和请求证据，停止自动补偿及 `restored=true` 声明。即时 GET 原值仍可能早于迟到 POST 提交，应在后端请求执行状态明确后人工对账。已观察完整响应时再读取当前配置：等于原值时保持；等于测试目标时才补偿恢复；第三方值出现时停止覆盖并写入 `cleanupError`。被强制终止后依据 `original-config.json` 与当前 GET 对比处理，外层进程 cleanup 保持证据。

所有 route handler Promise 纳入集合；护栏保留到 browser context 完全关闭，再等待已开始的 handler 结束，最后重新检查 blocked 请求、页面/handler 错误及未知 POST 并写结果。`safetyGate.checkedAfterContextClose` 标记最终检查，清理期间迟到错误也会使本次验证失败；`ignoredDevRequests` 单独记录开发辅助流。

清理保留整个证据目录，不删截图、日志、trace、JSON。结束后检查 `cleanup.json` 的 `portReleased`，并确认所声明的截图和结果文件仍存在。若工具被强制中断，使用同一 run 的 doctor/cleanup 恢复；归属冲突先人工核对，保留证据。

## Helpers

- `scripts/verify.mjs`：可执行总入口，上文覆盖 launch、doctor、drive、cleanup、run；`node .../verify.mjs --help` 显示参数。
- `scripts/flows.mjs`：由 `verify.mjs drive/run` 调用的 Playwright 操作模块，选择器和断言集中在此，独立进程入口使用 verify.mjs。
- `scripts/config-write.mjs`：写授权、精确 endpoint/payload 校验、禁跳转的真实响应转发、POST 终态日志、并发冲突恢复决策；由总入口和保存流程调用。
- `scripts/settings-save.mjs`：受控真实保存、关闭重开验证、通过 UI 恢复及异常补偿；通过上述 `run/drive --features settings-save --allow-config-write true` 调用。
- `scripts/verify.test.mjs`：验证配置指纹、证据一致失效、run 路径限制，以及本地 Vite 启动命令契约、缺授权拒绝、写请求护栏、第三方值保护、迟到提交及网络结果未知时的恢复边界、LLM 密钥字段拒绝、响应白名单、安全整数与 null 恢复；使用临时目录，无需应用或后端。执行 `node --test .agents/skills/verify-mooc-manus/scripts/verify.test.mjs`。

- `scripts/llm-policy.mjs`：LLM 响应白名单、四字段无密钥写护栏、安全整数检查、温度 f32 数值比较和恢复比较。
- `scripts/route-lifecycle.mjs`：跟踪 route Promise、等待清理期间回调结束并最终判定结果，由总流程调用。
- `scripts/llm-transport.mjs`：LLM GET/HEAD 的真实响应转发、禁止重定向、精确排除本地开发 SSE 和固定安全错误记录，由 llm/llm-save 流程调用。
- `scripts/llm-flows.mjs`：LLM 读取与保存流程；通过上文 `--features llm` 或 `--features llm-save --allow-config-write true` 调用。
- `scripts/write-guard.browser.test.mjs`：可选真实浏览器回归，仅使用临时本地内存 HTTP 夹具，验证 200 响应保留、307/308 无后续写入及网络异常；使用已有 Playwright/Chrome，独立执行 `node --test .agents/skills/verify-mooc-manus/scripts/write-guard.browser.test.mjs`。默认自测无需浏览器，该夹具也保持开发后端原样。

- `scripts/llm-read-guard.browser.test.mjs`：可选 Chrome 双本地服务器夹具，验证 GET/HEAD 的 200 透传、302/307 跨源目标零命中、开发 SSE 零上游请求、网络失败和清理期间迟到错误。独立执行 `node --test .agents/skills/verify-mooc-manus/scripts/llm-read-guard.browser.test.mjs`，保持真实开发后端原样。

应用变化后使用 `$pstack-maintain-verification-skill` 复核 map、脚本与真实流程；新增入口要同时补 recipe 和证据覆盖。
