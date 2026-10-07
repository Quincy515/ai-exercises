# 前端工作区

每次修改代码先阅读 [AGENTS.md](AGENTS.md) 和 [ARCHITECTURE.md](ARCHITECTURE.md)。
后者是生产目标目录、职责边界、Core 生命周期和迁移顺序的统一依据；下文记录当前已实现功能与操作方式。

`electron-app` 和 `tanstack-app` 通过 `workspace:*` 使用同一个
`@apps/frontend` 源码包。新建会话首页在 `packages/src/new-session.tsx`，
输入区和推荐问题分别在 `packages/src/components/chat-input.tsx`、
`packages/src/components/suggested-questions.tsx`。
本课使用静态附件和问题，上传、移除、发送与推荐问题点击保留为业务入口占位。
会话详情在 `packages/src/chat.tsx`，组合 `SessionHeader`、可折叠的 `PlanPanel` 和共用输入区。
详情页当前显示静态标题、7 条模拟消息和演示计划，计划展开/收起由 React 状态控制。
`components/chat-message.tsx` 展示用户消息气泡、AI 图标与正文，时间在悬停对应消息时显示。
步骤消息展示完成图标、单行描述、虚线与四条工具调用，展开/折叠按钮保留为交互入口。
`components/tool-use.tsx` 展示工具调用提示、执行状态、文件名和悬停时间，由普通工具消息与步骤内部共用。
`components/attachments-message.tsx` 根据 `role` 展示右对齐的用户附件或左对齐的 AI 附件；AI 附件增加查看任务全部文件的入口。
附件复用文件卡片，宽屏两列、窄屏单列；预览与 AI 附件下方的全部文件按钮保留为交互占位。
`components/session-header.tsx` 的右上角文件按钮打开任务文件列表弹窗，列表支持滚动，下载按钮保留为交互占位。
正文、步骤、工具信息、附件与时间仍使用课程示例。
`components/manus-icon.tsx` 保存课程提供的 Manus SVG 标识。
主题、字体和 Tailwind 样式在 `packages/src/index.css`。

当前采用 Web SPA + Electron CSR，业务界面和 Rust WASM 都在客户端运行。
Web 使用 TanStack Start 官方 SPA 模式，构建时生成静态页面壳；
部署 `tanstack-app/dist/client`，并将前端路由回退到 `/_shell.html`。
业务路由设置 `ssr: false`，统一在 `ClientOnly` 内挂载；Query 使用普通客户端 Provider。

共享布局使用三级区域：64px 一级功能栏、可拖动的聊天列表、自动伸展的内容区。
桌面聊天列表默认 280px，可在 220～420px 之间拖动；内容区至少保留 320px。
向左拖过最小宽度 40px 后自动收起，边界向右拖动或点击展开按钮可恢复。
拖动使用 shadcn `Resizable`，实现集中在 `packages/src/app/layout.tsx`，两端共用。
首页和 `/sessions/$id` 复用聊天列表；收起只影响二级列表，一级导航始终保留。
`/schedules` 与 `/library` 已接通两端路由，当前显示占位页；设置弹窗复用在一级栏底部，
账号位置暂作展示。窄屏继续使用聊天列表抽屉，业务核心沿用现有 Crux 接入。

## 首个真实 API：读取与保存 Agent 配置

在两端打开「设置 → 通用配置」，页面自动读取 `GET /api/app_configs/agent`。
三个字段支持编辑，点击「保存」发送 `POST /api/app_configs/agent`，请求和响应均为这三个字段的 JSON 对象。

| 字段 | 整数范围 |
| --- | --- |
| `max_iterations` 最大迭代次数 | 1–999 |
| `max_retries` 最大重试次数 | 2–9 |
| `max_search_results` 最大搜索结果 | 2–29 |

Rust 保存字符串草稿，点击保存时统一校验，清空、非整数或越界输入显示中文提示。
未修改时保存按钮禁用；有草稿时禁用刷新，点击「撤销修改」恢复最近一次服务端确认值。
点击「取消」或关闭弹窗会丢弃未保存草稿，重新打开时读取服务器配置。
保存期间禁用字段、切换面板和关闭，避免提交途中丢失状态；成功后以服务端返回值更新表单并显示「保存成功」。
失败时保留草稿供用户手动重试。HTTP 默认 30 秒超时，保存超时或网络中断会提示结果尚未确认，可重新打开设置核对。
LLM、A2A、MCP 面板沿用课程占位，保存按钮保持禁用。

```text
AgentConfigPanel → useAgentConfig → useCrux → Event::Configs
  → model/configs.rs（草稿、校验与状态）→ api/configs.rs（GET/POST）
  → Http Effect → http.ts → 后端 → AgentConfigReceived / AgentConfigSaved
  → ConfigsModel → Render → view/configs.rs → ViewModel → React
```

- [api/configs.rs](shared/src/api/configs.rs) 保存 DTO 和 HTTP 请求构造，`api/mod.rs` 统一校验和拼接地址。
- [model/configs.rs](shared/src/model/configs.rs) 管理读取、编辑、撤销、保存、去重与错误；[view/configs.rs](shared/src/view/configs.rs) 提供页面数据。
- [features/configs](packages/src/features/configs/index.ts) 集中弹窗、表单、业务 Hook 和事件包装，通过 `@apps/frontend/configs` 导出 `ManusSettings` 与 `useAgentConfig`。
- 两端均通过 `AppLayout → NavigationRail → ManusSettings → AgentConfigPanel` 使用相同实现。
- 弹窗内容挂载时创建一个 Core，表单与底部按钮共享状态；卸载取消请求并释放 Core。应用级 Provider 仍按架构文档迁移。

服务根地址集中在 `packages/src/lib/crux/api-config.ts`。HTTP 页面默认请求页面同源 `/api`：
开发环境由两端共用的 `vite.api.mts` 代理到 `http://localhost:5150`，生产 Web 由 nginx 代理。
当前后端未配置 CORS，开发联调使用此同源代理。

需要指定其他服务时，参照各宿主的 `.env.example`，在对应 `.env.local` 中设置
`VITE_API_BASE_URL=https://manus.example.com`（服务根地址，省略 `/api`），重启开发服务或重新构建。
显式跨域地址需要后端允许页面 Origin；`VITE_` 变量会公开写入客户端产物。
打包 Electron 的 `file://` 页面默认直连本地后端，同样需要服务端 CORS 或桌面专用传输，发布前单独联调。

在 `apps/` 下先执行 `just install`，然后在两个终端分别启动：

```sh
pnpm --filter tanstack-app dev
pnpm --filter electron-app start
```

核对 `curl --fail http://localhost:5150/api/app_configs/agent` 与两端页面数据一致。
新增接口继续按 OpenAPI → Rust DTO/状态/事件 → `just install` → 业务 Hook → 共享组件的顺序推进。
现有 Core、HTTP Shell、地址选择和开发代理可继续复用。

### 按 weather 的职责划分扩展业务

本项目采用官方 `crux/examples/weather` 的职责划分，完整目标结构与强制边界见 [ARCHITECTURE.md](ARCHITECTURE.md)，Rust 细则见 [shared/AGENTS.md](shared/AGENTS.md)。下面是当前已完成的 Rust 设置模块切片。

```text
shared/src/
├── app.rs             # Crux App 入口，委派 update/view
├── effects.rs         # 统一能力协议
├── api/
│   ├── mod.rs         # 公共 URL 工具与模块声明
│   └── configs.rs     # 设置接口 DTO、HTTP 请求构造
├── model/
│   ├── mod.rs         # 根 Event/Model、模块事件分发
│   └── configs.rs     # 设置事件、状态与更新流程
├── view/
│   ├── mod.rs         # 根 ViewModel、视图汇总
│   └── configs.rs     # 设置页面数据、From<&ConfigsModel>
└── capabilities/      # 自定义能力实现，如 SSE
```

`AppCore` 通过 `model.update(event)` 更新业务，通过 `model.into()` 生成视图。
根 `Event::Configs` 携带设置事件，设置模块返回的 Command 使用 `map_event(Event::Configs)` 接回根层。
API 模块独立于业务 Model/Event；状态更新留在 `model/`，页面输出转换留在 `view/`。

后续认证、会话、文件和健康检查按实际业务分别增加 `api/<模块>.rs`、`model/<模块>.rs`、
`view/<模块>.rs`，并在根模块注册。例如会话对应 `api/sessions.rs`、`model/sessions.rs`、`view/sessions.rs`。
一次业务流程可以组合多个接口；根层按业务模块和跨模块协调组织。
单个模块明显变大后再拆子目录，例如 `model/configs/{mod,agent,llm}.rs`。

现有计数器作为教学示例保留在模型层。跨 FFI 的 Event/ViewModel 需要 Facet 与 Serde；内部模型按实际需要派生类型。
`packages/src/app/` 负责共享布局；`packages/src/features/configs/` 负责设置业务 UI、Hook 与事件包装；`packages/src/lib/crux/` 提供通用 Shell 与 Hook。
后续业务继续复用现有 HTTP Shell，并通过 `just install` 同步生成类型和 WASM。

## noVNC 入门查看页

`packages/src/components/vnc-viewer.tsx` 封装课程中的 RFB 连接，
`packages/src/novnc.tsx` 提供两端共用的全屏页面。
本课固定连接 `ws://127.0.0.1:5901`，`viewOnly={false}` 可操作桌面；
URL 中的会话 ID 暂时用于路由，后续再与沙箱 API 关联。

依赖集中在共享包，固定为 `@novnc/novnc` 1.5.0 和
`@types/novnc__novnc` 1.6.0。RFB 在浏览器挂载后加载；
`lib/novnc.ts` 统一不同打包器对 CommonJS 默认导出的包装。
离开页面或修改连接参数时会断开旧连接。

在 `apps` 目录中，先启动 Docker Desktop，再准备沙箱：

```sh
# 首次准备镜像；已有 sandbox-dev 镜像时可跳过
docker build -t sandbox-dev ../sandbox
docker run --rm -d --name sandbox-dev --shm-size=512m \
  -p 127.0.0.1:8080:3000 \
  -p 127.0.0.1:5900:5900 \
  -p 127.0.0.1:5901:5901 \
  -p 127.0.0.1:9222:9222 sandbox-dev
```

本项目沙箱 API 的容器端口是 3000；noVNC 使用 5901 的 WebSocket，
由容器内 websockify 转接到 5900 的 VNC 服务。

- Web：运行 `pnpm dev:web`，访问 `http://localhost:3000/sessions/1/novnc`。
- Electron：运行 `pnpm dev:desktop`，在开发者工具 Console 执行
  `window.location.hash = '/sessions/1/novnc'`；返回会话页可改为 `/sessions/1`。
- 查看页跳过三栏布局，画面按窗口缩放；控制台输出 `Connected` / `Disconnected`。
- 完成后可执行 `docker stop sandbox-dev`，`--rm` 会移除该容器。

参考：[noVNC 1.5.0 API](https://github.com/novnc/noVNC/blob/v1.5.0/docs/API.md)。

## 开发、热更新与发布

以下命令均在 `mooc-manus/apps` 目录执行。从 `mooc-manus` 根目录进入：

```sh
cd apps
```

### 1. 首次准备与日常启动

安装好下文列出的 Rust/Node/pnpm/just 工具后，先生成共享包并安装依赖：

```sh
just install
```

执行顺序为 `wasm → typegen → pnpm install --frozen-lockfile → 清除两端 Vite 预构建缓存`。`wasm` 会自动对齐
BoltFFI CLI 和编译所需的 runtime；安装后若发现本地 `shared` 的旧依赖缓存，
会定向刷新该包。准备完成后，两个终端分别启动：

```sh
# 终端 A：Web，默认 http://localhost:3000
pnpm --filter tanstack-app dev
```

```sh
# 终端 B：Electron 开发应用
pnpm --filter electron-app start
```

这两个命令复用已有 shared 产物，适合同时调试两端。单独启动一端也可以直接用
`pnpm dev:web` / `pnpm dev:desktop`，它们会先执行一次 `just install`。
启动完成后，Vite 持续监听前端文件；Rust 的后续修改按第 3、4 节处理。

### 2. 哪些修改会自动更新

| 修改位置 | 开发时的更新行为 |
| --- | --- |
| `packages/src` 的 React/TS/CSS | 两端 Vite 处理 HMR / React Fast Refresh |
| `tanstack-app/src` | 更新 Web 页面 |
| Electron renderer 的 React/TS/CSS | 更新 Electron 窗口页面 |
| `electron-app/src/preload.ts` | Forge 重编译后发送 full-reload，重新加载窗口 |
| `electron-app/src/main.ts` | Forge 重编译；在交互式 Forge 终端输入 `rs` 回车重启主进程，或 Ctrl+C 后重启开发命令 |
| `shared` 的 Rust / Cargo / BoltFFI 配置 | 重新生成 WASM、类型和本地包，再重启开发服务 / 应用 |

main 的结论以当前 `@electron-forge/plugin-vite@7.11.2` 源码为准：自动重启钩子仍被注释，
看到 `target built main` 表示编译完成，随后需要执行上述重启操作。

React Fast Refresh 会尽量保留可复用的组件状态；完整页面刷新、应用重启和 Core 重新初始化
会创建新的 Rust 内存状态。需要持久化的数据应由业务明确写入 KV 或后端。
开发服务提供热更新，已经打包的 `.app` 和静态发布文件通过重新构建、发布来更新。

### 3. 修改 shared 后的推荐流程

在 `shared/src/model/` 修改业务规则、Event、Model，在 `shared/src/view/` 修改 ViewModel；新增能力协议时修改
`shared/src/capabilities`，并补充 `packages/src/lib/crux` 中对应的 Shell 处理。
生成代码由工具维护，日常编辑源代码。

先停止正在运行的两端开发命令，按顺序执行：

```sh
cargo check -p shared
just test
just install
pnpm typecheck
```

`just install` 生成 `generated/pkg` 和 `generated/types/dist`，刷新 `file:` 本地依赖，并清除两端旧 Vite 预构建缓存。
新增 Event / Effect / ViewModel 字段后，类型检查会指出前端需要同步修改的位置。

然后分别重启两端：

```sh
# Web 终端
pnpm --filter tanstack-app dev
```

```sh
# Electron 终端
pnpm --filter electron-app start
```

`just install` 清除的是可重新生成的 Vite 缓存。本地类型包版本保持不变时，旧缓存可能缺少新增事件或类型；
生成任务自动清理后，重启两端即可使用最新协议。重新加载页面会创建新的 WASM 实例。
仅执行 `cargo build` 会生成 Rust 编译产物；给两端更新 WASM/npm 包使用 `just install`。

### 4. Rust 频繁修改时的可选自动模式（macOS）

当前 Justfile 保持一次性任务。需要持续监听时，可以使用 `cargo-watch`；按需安装：

```sh
cargo install cargo-watch --locked
```

先停止原有开发命令，再从下面两个方案中选择一个。

自动构建 shared 并重启 Web：

```sh
cargo watch -w shared -w Cargo.toml \
  -s 'just install && pnpm --filter tanstack-app dev --force'
```

自动构建 shared 并重启 Electron：

```sh
cargo watch -w shared -w Cargo.toml \
  -s 'just install && pnpm --filter electron-app start'
```

监听器首次运行会构建并启动应用；保存 Rust 文件后，结束旧开发进程，再执行构建和启动。
Web 的 Vite 客户端重连后刷新页面，Electron 会重新启动应用。前端 TS/CSS 仍由 Vite 热更新。
Ctrl+C 结束监听。

这两个监听方案各自会生成同一份 `generated` 产物，因此选择单端自动模式；
同时调试两端时采用第 1、3 节的流程，统一生成一次再启动两端。
这些是可选终端用法，工程保持现有 Justfile 和 pnpm 入口。

### 5. 构建、打包和本地预览

| 命令 | 作用 / 产物 |
| --- | --- |
| `just wasm` | 编译并打包 WASM 与 FFI 绑定到 `generated/pkg` |
| `just typegen` | 生成业务类型并编译到 `generated/types/dist` |
| `pnpm build:shared` | 生成上面两套产物 |
| `pnpm prepare:shared` | 生成两套产物，并完成本地依赖安装 |
| `pnpm build:web` | 更新 shared 后构建 Web，部署目录为 `tanstack-app/dist/client` |
| `pnpm build:desktop` | 更新 shared 后打包 Electron 应用，输出到 `electron-app/out` |
| `pnpm build` | 构建两端，同一次 just 调用内共享前置任务执行一次 |

本地预览已构建的 Web（默认使用 4173 端口）：

```sh
pnpm --filter tanstack-app preview --port 4173
```

preview 用于查看构建产物；源码修改后先重新运行 `pnpm build:web`。
Web 发布时上传整个 `dist/client`，包含 `_shell.html`、JS/CSS、字体及 WASM，并配置
前端路由回退到 `/_shell.html`。TanStack Start 的服务端构建用于生成静态壳，业务在客户端运行。

`pnpm build:desktop` 生成可运行应用。需要 ZIP / 安装包时执行：

```sh
just install
pnpm --filter electron-app make
```

make 输出位于 `electron-app/out/make`；当前 macOS 配置使用 ZIP maker，其他平台按对应
maker 和构建环境处理。已安装的应用通过安装新包获得新版本。

### 6. 自动更新的范围

本地监听可以自动完成构建和重启。当前 Justfile 的工作终点是本地产物，Web 上传/CDN 发布、
在线页面版本检测、Electron 自动升级都尚未接入。Electron 的 `publish` 脚本仍需配置 publisher，
客户端还需更新源与 `autoUpdater` 等发布设施。

因此，开发阶段按前面的 HMR / Rust 重建流程更新；生产阶段发布完整 Web 静态目录或新的
Electron 包。浏览器已打开的旧页面重新加载后使用新版本。

### 7. 提交前验证

```sh
just test
pnpm test:shared
pnpm test:crux
pnpm typecheck
pnpm --filter tanstack-app lint
ESLINT_USE_FLAT_CONFIG=false pnpm --filter electron-app lint
pnpm build
```

Electron 当前使用 `.eslintrc.json`，ESLint 9 通过上述环境变量启用旧配置格式。

### 8. source map 提示的含义

`shared.js.map` 用于将 JavaScript 调试位置映射回生成的 TypeScript。该文件缺失会影响
断点和堆栈定位，WASM 执行依赖的仍是 JavaScript 和 `.wasm`。

BoltFFI 0.31.0 自动收集浏览器与 Node 入口、map 和 WASM 辅助模块，map 中内嵌源码；
`@boltffi/runtime` 则显式交给两端 Vite 预构建，避免逐模块加载时触发上游缺失源码的告警。
上游 runtime 0.31.0 包没有发布其原始 `src/*.ts`，因此 Electron/Vite 6 内部库的原始 TS 调试仍受此限制。
修复后执行 `just install`，再按第 3 节重启开发服务以重新加载依赖缓存。

依据：[Vite 本地依赖与缓存](https://vite.dev/guide/dep-pre-bundling)、
[Electron Forge Vite 插件](https://www.electronforge.io/config/plugins/vite)、
[cargo-watch](https://github.com/watchexec/cargo-watch)，以及本项目 Justfile、pnpm 脚本和已安装插件源码。

## 共享约定

- 页面使用 `import Chat from '@apps/frontend/chat'`。
- 两端样式入口统一导入 `@apps/frontend/styles.css`。
- 两端当前统一使用共享浅色主题。
- 共享包内部使用相对路径；各应用的源码别名归各应用管理。
- 共享包的 React 使用 peer dependency，各宿主提供 React 运行时。
- 各宿主运行自己的 Vite 配置，并编译共享 TSX/CSS 源码。Electron
  保留 Forge 对应的 Vite 6，Web 使用 Vite 8。
- 依赖安装配置和锁文件统一维护在本目录。

当前首页与会话详情属于课程基础 UI，通用配置已通过 `useAgentConfig` / `useCrux` 接入 Rust HTTP 流程。
模型层保留计数器业务示例，现有 WASM 和 Crux 测试继续验证桥接层。

## 在组件中使用 Crux

```tsx
import { useCrux } from '@apps/frontend/crux'

function Counter() {
  const { view, events, dispatch, ready, error } = useCrux()
  return (
    <>
      <p>{view?.text ?? '正在加载…'}</p>
      <button disabled={!ready} onClick={() => dispatch(events.Increment())}>加一</button>
      {error && <p role="alert">{error.message}</p>}
    </>
  )
}
```

`packages/src/lib/crux/core.ts` 统一处理 Bincode 与 effect 分发；
`http.ts`、`sse.ts`、`key-value.ts`、`time.ts` 执行浏览器能力。
当前 Hook 每次挂载创建独立 Core，卸载时取消请求、清除定时器并释放 WASM handle。
应用级共享 Provider、业务取消策略和全局错误隔离是已登记的生产迁移项，接入认证及多模块联动前按架构规范一起完成。
`useCrux()` 的 `view` 初始为 `null`，WASM 就绪后以 Rust 的 `view()` 为准，组件使用可选链处理初始化阶段。

示例 API 使用 Rust 中配置的 `https://crux-counter.fly.dev`。
HTTP/SSE 验证使用模拟响应；上方示例展示组件如何发送 Rust 事件。
SSE 可通过 `dispatch(events.StartWatch())` 发起，并在组件卸载时取消。
KV 适配支持完整字节存储；当前业务通过 `LoadState` 读取状态，自动保存需要 Rust 发出 `Set`。

`pnpm test:crux` 使用现有 TypeScript 编译器和 Node 内置测试，覆盖真实 WASM、
HTTP/SSE/KV/Time、初始化与资源释放。

## Rust / Wasm 与共享类型

当前项目统一使用 BoltFFI CLI / Rust crate / `@boltffi/runtime` `0.31.0`，
Binaryen（`wasm-opt`）使用 `132` 及以上版本。

```sh
brew install just
brew install binaryen
rustup target add wasm32-unknown-unknown
```

**BoltFFI 版本只在 `apps/Cargo.toml` 中维护**：`boltffi = "=0.31.0"`。
`Justfile` 通过 Cargo metadata 读取精确版本，自动安装相同版本的 CLI，并通过官方
`--overlay` 临时配置将生成 npm 包的 runtime 固定到相同版本。
统一使用 `just wasm` / `just install`；直接执行 `boltffi pack wasm` 会绕过这套对齐流程。

BoltFFI 在生成 npm 包之前调用 TypeScript 检查 runtime。为避免循环依赖和旧包缓存，
构建会在被忽略的 `generated/node_modules` 中准备匹配的 runtime、TypeScript 5.9.3
及 Node 类型；已匹配时复用缓存。生成完成后再安装前端工作区依赖。

后续升级时，先将 `Cargo.toml` 的 BoltFFI 精确版本改为三个包均已发布的目标版本，再执行：

```sh
cargo update -p boltffi
just install
pnpm test:shared
pnpm test:crux
pnpm typecheck
pnpm build
```

提交 `Cargo.toml`、`Cargo.lock`、`pnpm-lock.yaml` 及包管理器产生的相关配置改动。
`pnpm test:shared` 同时检查生成包与实际安装的 runtime 是否匹配 Cargo 中的版本。
日常构建遵循已验证的精确版本，升级由上述步骤明确触发。

全部任务集中在 `apps/Justfile`，直接运行 `just` 可查看命令列表。
其中保留 Crux 官方示例的生成与安装顺序；0.31.0 已由上游正确生成包清单和 source map，
此前手工添加 Node 文件与映射源码的补丁已移除。首次运行也可以直接执行 `just install`。

日常开发、Rust 改动后的更新和可选监听命令见上面的开发流程。

两个生成目录集中放在 `apps/generated`，已由 `.gitignore` 忽略：

- `generated/pkg`：BoltFFI 生成的 WASM、JavaScript 和 FFI 类型。
- `generated/types`：Crux 生成的源码；`dist` 提供 Event、Effect、ViewModel 及 Bincode 的 JavaScript 和声明。

`generated/types` 同时列入 pnpm workspace，让 codegen 内部执行的 pnpm
命令复用根锁文件。codegen 将编译产物与源码分开，并通过 package exports
选择 `dist` 中的声明，兼容前端的严格 TypeScript 检查；编译失败会返回错误。

两端 Vite 显式预构建生成的 CommonJS 类型包；WASM 包交给 Vite 处理其资源 URL。
Electron 的 Vite 6 同时配置生产 CommonJS 转换。

`packages/package.json` 使用相对路径引入它们，两端共享同一份依赖：

```json
{
  "shared": "file:../generated/pkg",
  "shared_types": "file:../generated/types"
}
```

需要单独排查时，在 `apps` 下运行：

```sh
just wasm
just typegen
```

在 `shared` 目录执行 `just` 时，会自动使用上级的 `apps/Justfile`。
`just build` 始终构建两端应用；单独编译 Rust 库可执行
`cargo build -p shared`。

BoltFFI 会通过 Cargo metadata 找到实际 target 目录，因此
`boltffi.toml` 省略 `artifact_path`，兼容全局 `target-dir` 配置。
Node 入口用于集成测试，浏览器入口交给 Vite；对应 source map 由生成器维护。
单独打包后执行 `just install`，可以刷新前端使用的本地依赖。

当前 FFI 使用 `CoreFfi.new({ processEffects })`。浏览器使用前需要等待
`shared` 的 `initialized`，Shell 负责处理 `update` / `resolve` 返回的
effects 和 `processEffects` 回调。`pnpm test:shared` 会实际加载生成的
WASM，并验证事件、ViewModel 和 effect 响应的序列化往返。

参考：[Crux React 文档](https://redbadger.github.io/crux/part-1/shell/web/react.html#compile-our-rust-shared-library)、
[Crux lib.just](https://github.com/redbadger/crux/blob/master/lib.just)。

## 可复用用户流程验证

项目技能位于 [verify-mooc-manus](../.agents/skills/verify-mooc-manus/SKILL.md)，默认覆盖通用配置读取刷新、会话导航与计划、任务文件列表。先完成 `just install`，然后在 `mooc-manus` 根目录执行：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features settings,sessions,files --port 4317
```

脚本使用独立端口和浏览器上下文，自动 doctor、驱动并清理自己启动的前端进程，证据保留在 `output/playwright/verify-mooc-manus/`。配置查询需要已有 `localhost:5150` 后端；服务不可用时记录 blocked。会话和文件目前验证真实 UI 交互与内置演示数据，聊天发送和文件下载仍按后续业务接入。具体边界、入口与退出码见技能及 feature map。

验证 Agent 配置编辑保存时，显式启用写流程：

```sh
node .agents/skills/verify-mooc-manus/scripts/verify.mjs run --features settings-save --allow-config-write true --port 4317
```

在独占开发后端执行，期间暂停其他配置编辑。脚本记录原值，检查非法输入、真实保存、关闭重开后的 GET，再通过 UI 恢复原值；证据包含 `original-config.json`、`api-responses.json` 与 `config-cleanup.json`。遇到其他客户端的新值会停止覆盖；写请求结果无法确认时记录失败和待核实状态。结束后同时核对配置恢复结果与进程清理结果。此脚本覆盖 Web 首页保存路径，其他入口与 Electron 按目标平台单独验证。
