# 前端工作区

`electron-app` 和 `tanstack-app` 通过 `workspace:*` 使用同一个
`@apps/frontend` 源码包。新建会话首页在 `packages/src/new-session.tsx`，
输入区和推荐问题分别在 `packages/src/components/chat-input.tsx`、
`packages/src/components/suggested-questions.tsx`。
本课使用静态附件和问题，上传、移除、发送与推荐问题点击保留为业务入口占位。
会话详情在 `packages/src/chat.tsx`，组合 `SessionHeader`、可折叠的 `PlanPanel` 和共用输入区。
详情页当前显示静态标题、7 条模拟消息和演示计划，计划展开/收起由 React 状态控制。
`components/chat-message.tsx` 展示用户消息气泡、AI 图标与正文，时间在悬停对应消息时显示。
正文与时间仍使用课程示例，工具、步骤和附件保留占位；`role` 保留附件来源。
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

执行顺序为 `wasm → typegen → pnpm install --frozen-lockfile`。`wasm` 会自动对齐
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

在 `shared/src/app.rs` 修改业务规则、Event、Model、ViewModel；新增能力协议时修改
`shared/src/capabilities`，并补充 `packages/src/lib/crux` 中对应的 Shell 处理。
生成代码由工具维护，日常编辑源代码。

先停止正在运行的两端开发命令，按顺序执行：

```sh
cargo check -p shared
just test
just install
pnpm typecheck
```

`just install` 生成 `generated/pkg` 和 `generated/types/dist`，并刷新 `file:` 本地依赖。
新增 Event / Effect / ViewModel 字段后，类型检查会指出前端需要同步修改的位置。

然后分别重启两端：

```sh
# Web 终端：重新预构建本地 CommonJS 类型包
pnpm --filter tanstack-app dev --force
```

```sh
# Electron 终端：清除当前 renderer 的 Vite 预构建缓存，再启动
rm -rf electron-app/node_modules/.vite
pnpm --filter electron-app start
```

这里清除的是可重新生成的 Vite 缓存。生成类型包经过依赖预构建，重启时刷新缓存可确保使用
最新协议；重新加载页面后，新的 WASM 实例才会执行新的 Rust 逻辑。
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
  -s 'just install && rm -rf electron-app/node_modules/.vite && pnpm --filter electron-app start'
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

当前首页与会话详情属于课程基础 UI，后续业务通过 `useCrux` 接入同一套 Rust 逻辑。
`shared/src/app.rs` 仍保留计数器业务示例，现有 WASM 和 Crux 测试继续验证桥接层。

## 在组件中使用 Crux

```tsx
import { useCrux } from '@apps/frontend/crux'

function Counter() {
  const { view, events, dispatch, ready, error } = useCrux()
  return (
    <>
      <p>{ready ? view.text : '正在加载…'}</p>
      <button disabled={!ready} onClick={() => dispatch(events.Increment())}>加一</button>
      {error && <p role="alert">{error.message}</p>}
    </>
  )
}
```

`packages/src/lib/crux/core.ts` 统一处理 Bincode 与 effect 分发；
`http.ts`、`sse.ts`、`key-value.ts`、`time.ts` 执行浏览器能力。
Hook 每次挂载创建独立 Core，卸载时取消请求、清除定时器并释放 WASM handle。
`useCrux()` 从空展示状态开始，WASM 就绪后以 Rust 的 `view()` 为准。

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
