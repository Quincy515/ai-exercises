---
name: verify-mooc-manus
description: 验证 MoocManus 的 Web 用户流程：Agent 配置读取刷新、会话导航与计划展开、任务文件列表。需要复现界面行为、检查 Crux 接入或保存真实操作证据时使用；明确区分真实 API、演示 UI 和未实现功能。
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

脚本启动 `pnpm --filter tanstack-app dev --host 127.0.0.1 --port 4317 --strictPort`，环境仅覆盖 `VITE_API_BASE_URL=` 以固定同源 `/api`。监听进程必须属于本次进程组，首页返回 `Mooc Manus` 且源码/生成包指纹一致才算 ready。端口可用 `--port` 调整；保持后端代理目标 `localhost:5150`。

需要分步操作时：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs launch --port 4317 --run output/playwright/verify-mooc-manus/manual
node .agents/skills/verify-mooc-manus/scripts/verify.mjs doctor --run output/playwright/verify-mooc-manus/manual
node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features sessions,files
node .agents/skills/verify-mooc-manus/scripts/verify.mjs cleanup --run output/playwright/verify-mooc-manus/manual
```

每次启动选择新的 `--run` 目录，路径必须是 `output/playwright/verify-mooc-manus/` 的直接子目录；启动、doctor、drive 和 cleanup 统一拒绝多级子目录。脚本保留已有证据并拒绝覆盖。省略 `--run` 时生成带时间戳的目录，命令结果给出绝对路径。

后端启动方式来自 `server/README.md`：依赖 PostgreSQL/Redis 就绪后，在 `server/` 执行 `cargo loco start`。就绪条件是 `GET http://localhost:5150/api/app_configs/agent` 返回三个整数；当前查询无需登录和 seed。后端会自动迁移数据库，本技能只读借用已运行的开发后端，**不自动启动/重置数据库、Docker 容器或后端，也不结束借用进程**。缺少后端时，配置流程记录 blocked；会话/文件的现有 UI 流程可以独立验证。

Electron 的现有启动命令为 `pnpm --filter electron-app start`（在 `apps/`）；本技能不自动接管 Electron 或用户浏览器。需要桌面证明时独立隔离实例、profile 和调试端口，并按相同 map 驱动，单独报告结果。

## Doctor

`verify.mjs doctor --run <本次目录>` 是只读检查：核对 PID 起始时间、监听端口所属进程组、首页标识、源码/生成物 SHA-256、现有 Playwright 安装与后端契约。它不改运行状态、不操作页面、不调用写 API。

返回 `ok: true` 表示 Web 实例可驱动；`eligible.settings: false` 表示后端条件不满足。每次有异常先执行 doctor；源码变化后启动新的 run，保留旧证明。doctor 输出可重定向到证据目录，但重定向本身属于调用者写证据。

## Drive

重复流程由 `verify.mjs drive --features settings,sessions,files` 执行，具体入口与判断标准见：

- [Agent 配置](features/agent-settings.md)：四个共享布局入口的真实 GET、刷新与只读字段。
- [会话与计划](features/session-plan.md)：侧栏、深链接刷新、计划展开收起与返回首页。
- [任务文件](features/task-files.md)：Header 文件按钮、六项列表与关闭恢复。

每个流程使用新 browser context，1280×900，临时浏览器资料由 Playwright 管理。仅使用页面导航、ARIA 控件、用户点击与 DOM 读取；不调用 Core 内部方法或设置组件状态。流程中的 `/api/` 写请求会被阻止并判失败，GET 请求真实发送；不会伪造成功响应。

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
- `cleanup.json`：进程已停、端口释放、证据文件仍存在。

证明必须同时包含操作与结果。截图将有限过渡动画推进到结束状态，保留清晰、稳定的页面证据。配置值与本次浏览器 GET 响应比对；只读流程报告观测到的请求及写请求 guard，不声称检查了数据库全量状态。会话和文件是内置演示数据，点击路由/计划/Dialog 的通过仅证明真实 UI 交互。新增写业务时先扩展 map、隔离数据和清理策略，并从第二个可观察入口核对持久化副作用。

Mock 只在明确的外部系统边界且单独标记的故障测试中使用；当前默认流程 `mocks: false`。后端不可用的 blocked、未接通的聊天/下载、未测的 Electron 或移动视口，都不能由其他成功入口代替。

## Cleanup

`run` 在成功与失败后自动关闭浏览器并结束本次创建的 Vite 进程组；分步启动后执行 `cleanup --run ...`。如果 drive 失败，先完成该次 cleanup 再重试。脚本按 PID 起始时间和进程组验证归属，归属变化时拒绝终止；禁止按进程名批量结束。

清理保留整个证据目录，不删截图、日志、trace、JSON。结束后检查 `cleanup.json` 的 `portReleased`，并确认所声明的截图和结果文件仍存在。若工具被强制中断，使用同一 run 的 doctor/cleanup 恢复；归属冲突先人工核对，保留证据。

## Helpers

- `scripts/verify.mjs`：可执行总入口，上文覆盖 launch、doctor、drive、cleanup、run；`node .../verify.mjs --help` 显示参数。
- `scripts/flows.mjs`：由 `verify.mjs drive/run` 调用的 Playwright 操作模块，选择器和断言集中在此，独立进程入口使用 verify.mjs。
- `scripts/verify.test.mjs`：验证配置指纹、证据一致失效和 run 路径限制；使用临时目录，无需应用或后端。执行 `node --test .agents/skills/verify-mooc-manus/scripts/verify.test.mjs`。

应用变化后使用 `$pstack-maintain-verification-skill` 复核 map、脚本与真实流程；新增入口要同时补 recipe 和证据覆盖。
