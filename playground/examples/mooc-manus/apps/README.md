# 前端工作区

`electron-app` 和 `tanstack-app` 通过 `workspace:*` 使用同一个
`@apps/frontend` 源码包。共享页面在 `packages/src/chat.tsx`，
主题、字体和 Tailwind 样式在 `packages/src/index.css`。

## 开发

首次在本目录执行（先生成本地依赖，再安装前端依赖）：

```sh
just install
just dev-web
```

另开终端，在本目录启动桌面端：

```sh
just dev-desktop
```

Web 默认地址为 `http://localhost:3000`。修改共享 Chat 后，两端的
Vite 开发服务都会更新页面。

## 验证与构建

```sh
just test
pnpm test:shared
pnpm typecheck
just build
```

现有应用的代码检查：

```sh
pnpm --filter tanstack-app lint
ESLINT_USE_FLAT_CONFIG=false pnpm --filter electron-app lint
```

Electron 沿用脚手架的 `.eslintrc.json`，因此 ESLint 9 需要启用旧配置模式。

也可以分别执行 `just build-web` 和 `just build-desktop`。
`pnpm dev:web`、`pnpm dev:desktop`、`pnpm build:web`、
`pnpm build:desktop` 和 `pnpm build` 都调用对应的 just 任务。
Web 产物位于 `tanstack-app/dist`；Electron 应用产物位于
`electron-app/out`，按当前操作系统和架构打包。

## 共享约定

- 页面使用 `import Chat from '@apps/frontend/chat'`。
- 两端样式入口统一导入 `@apps/frontend/styles.css`。
- 两端当前统一使用共享浅色主题。
- 共享包内部使用相对路径；各应用的源码别名归各应用管理。
- 共享包的 React 使用 peer dependency，各宿主提供 React 运行时。
- 各宿主运行自己的 Vite 配置，并编译共享 TSX/CSS 源码。Electron
  保留 Forge 对应的 Vite 6，Web 使用 Vite 8。
- 依赖安装配置和锁文件统一维护在本目录。

当前 Chat 保留现有按钮示例，用于验证共享页面和样式接线。
Rust/Wasm 构建和共享类型已接入；Chat 的业务逻辑与平台能力后续接入。

## Rust / Wasm 与共享类型

当前项目使用 BoltFFI CLI / Rust crate `0.30.1`，
Binaryen（`wasm-opt`）使用 `132` 及以上版本。

```sh
brew install just
cargo install boltffi_cli --version '=0.30.1' --locked
brew install binaryen
rustup target add wasm32-unknown-unknown
```

全部任务集中在 `apps/Justfile`，直接运行 `just` 可查看命令列表。
其中保留 Crux 官方示例的构建顺序和必要的 WASM npm 包修补。

`just install` 可以在安装前端依赖之前执行。
首次构建通过 `npm exec` 临时提供 TypeScript 5.9.3；工作区安装完成后，
构建使用 `node_modules/.bin/tsc`，避免依赖全局 `tsc`。

日常单独更新共享产物：

```sh
just install
```

该命令依次执行 WASM 打包、TypeScript 类型生成和本地依赖安装。
`just dev-web`、`just dev-desktop`、`just build-web` 和
`just build-desktop` 都依赖这一步。`just build` 会依次构建两端，
共享的前置任务执行一次。修改 Rust 后，重新执行对应命令即可更新产物。

单独生成共享包可以执行 `just build-shared`；原有的
`pnpm build:shared` 和 `pnpm prepare:shared` 分别调用
`just build-shared` 和 `just install`。

两个生成目录集中放在 `apps/generated`，已由 `.gitignore` 忽略：

- `generated/pkg`：BoltFFI 生成的 WASM、JavaScript 和 FFI 类型。
- `generated/types`：Crux 生成的 Event、Effect、ViewModel 及 Bincode 编解码器。

`generated/types` 同时列入 pnpm workspace，让 codegen 内部执行的 pnpm
命令复用根锁文件。`typegen` 任务会显式检查类型编译结果。

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

BoltFFI 0.30.1 会通过 Cargo metadata 找到实际 target 目录，因此
`boltffi.toml` 省略 `artifact_path`，兼容全局 `target-dir` 配置。
`wasm` 任务内包含 Crux 官方的修补，将 `shared_node.*` 纳入生成包
的 `files`，供 Node / SSR 入口使用。单独打包后，执行
`just install` 可重新应用该修补并刷新本地依赖。

当前 FFI 使用 `CoreFfi.new({ processEffects })`。浏览器使用前需要等待
`shared` 的 `initialized`，Shell 负责处理 `update` / `resolve` 返回的
effects 和 `processEffects` 回调。`pnpm test:shared` 会实际加载生成的
WASM，并验证事件、ViewModel 和 effect 响应的序列化往返。

参考：[Crux React 文档](https://redbadger.github.io/crux/part-1/shell/web/react.html#compile-our-rust-shared-library)、
[Crux lib.just](https://github.com/redbadger/crux/blob/master/lib.just)。
