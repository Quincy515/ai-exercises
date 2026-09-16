# 前端工作区

`electron-app` 和 `tanstack-app` 通过 `workspace:*` 使用同一个
`@apps/frontend` 源码包。共享页面在 `packages/src/chat.tsx`，
主题、字体和 Tailwind 样式在 `packages/src/index.css`。

## 开发

在本目录执行：

```sh
pnpm install --frozen-lockfile
pnpm dev:web
```

另开终端，在本目录启动桌面端：

```sh
pnpm dev:desktop
```

Web 默认地址为 `http://localhost:3000`。修改共享 Chat 后，两端的
Vite 开发服务都会更新页面。

## 验证与构建

```sh
pnpm typecheck
pnpm build
```

现有应用的代码检查：

```sh
pnpm --filter tanstack-app lint
ESLINT_USE_FLAT_CONFIG=false pnpm --filter electron-app lint
```

Electron 沿用脚手架的 `.eslintrc.json`，因此 ESLint 9 需要启用旧配置模式。

也可以分别执行 `pnpm build:web` 和 `pnpm build:desktop`。
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
Rust/Wasm、Mock 业务流程与平台能力在后续阶段接入。
