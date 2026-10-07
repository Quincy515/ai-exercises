# MoocManus 客户端架构与目录规范

本文件是 `apps/` 的统一架构依据，适用于设置、认证、会话、文件、健康状态及后续业务。每次修改代码都必须遵守职责、依赖方向、生命周期与验证要求。根级 `AGENTS.md` 负责执行入口，子目录规则补充具体约束。

本规范于 2026-10-03 对照本地 Crux 官方 `weather` 示例（提交 `30337489`）确定，2026-10-07 同步 Agent 配置编辑与保存。目录树描述生产目标；Agent 配置已完成 Rust 三层拆分与前端 `features/configs/` 归组。其余业务按实际需求实现，迁移进度见第 6 节。

## 1. 官方示例与项目选择

| 官方依据 | 本项目采用的原则 |
| --- | --- |
| [weather App](crux/examples/weather/shared/src/app.rs) | `app.rs` 只连接关联类型，委派 `Model::update` 和 ViewModel 转换 |
| [weather 子模块](crux/examples/weather/shared/src/model/active/home/mod.rs) | 模块拥有自己的事件与状态，父模块通过 `map_event` 组合 |
| [weather View](crux/examples/weather/shared/src/view/active/home.rs) | ViewModel 按展示需要生成，安全且结构合适的 DTO 可以复用 |
| [weather HTTP](crux/examples/weather/shared/src/effects/http/weather/mod.rs) | 请求构造独立封装；本项目将其放在 `api/` |
| [weather React Provider](crux/examples/weather/web-nextjs/src/lib/core/provider.tsx) | 在应用根管理 Core，子组件消费共享视图与事件分发 |
| [weather 请求版本](crux/examples/weather/shared/src/model/active/favorites/add.rs) | 并发请求需要关联身份，过期响应由模型判断并丢弃 |

项目保留已有 `effects.rs` 和 `capabilities/`，HTTP 客户端集中到 `api/`。API 层独立于业务 Model/Event 是本项目的额外约束。`weather` 的 `Outcome`、生命周期枚举、深层目录和演示密钥存储按业务需要评估；当前配置读写沿用直接返回 `Command` 的更新函数。

## 2. 最终目标目录

下列为目标落位。`configs` 的 Rust 三层和前端 `features/configs/` 已有实现；`provider.tsx` 与其他业务属于后续迁移或接入项。业务文件在有真实代码时创建，职责相同的文件在同层按模块归组。

```text
apps/
├── AGENTS.md                         # 每次修改的强制入口
├── ARCHITECTURE.md                   # 本规范，统一架构依据
├── README.md                         # 启动、构建、使用和验证
├── .cursor/rules/general.mdc          # Cursor 入口及项目技术约定
├── Cargo.toml / Cargo.lock
├── package.json / pnpm-lock.yaml
├── pnpm-workspace.yaml
├── Justfile                          # Rust、生成包、双端任务入口
├── vite.api.mts                      # 双端开发代理
├── shared/
│   ├── AGENTS.md                     # Rust 边界补充
│   ├── Cargo.toml / boltffi.toml
│   └── src/
│       ├── lib.rs                    # 模块声明、必要导出
│       ├── app.rs                    # 薄 Crux App 入口
│       ├── effects.rs                # 统一 Effect 协议
│       ├── ffi.rs                    # 原生/WASM 字节桥接
│       ├── bin/codegen.rs            # 类型生成工具
│       ├── api/
│       │   ├── mod.rs                # URL 与已出现复用需求的协议工具
│       │   ├── configs.rs            # 设置 DTO、HTTP 请求构造
│       │   ├── auth.rs               # 认证接口
│       │   ├── sessions.rs           # 会话接口
│       │   ├── files.rs              # 文件接口
│       │   └── status.rs             # 健康检查接口
│       ├── model/
│       │   ├── mod.rs                # 根 Event/Model、模块分发与协调
│       │   ├── configs.rs            # 设置状态与更新流程
│       │   ├── auth.rs               # 登录、退出、认证失效
│       │   ├── sessions.rs           # 会话与聊天流程
│       │   ├── files.rs              # 上传、下载及文件操作状态
│       │   └── status.rs             # 有展示状态时实现
│       ├── view/
│       │   ├── mod.rs                # 根 ViewModel 与汇总映射
│       │   ├── configs.rs
│       │   ├── auth.rs
│       │   ├── sessions.rs
│       │   ├── files.rs
│       │   └── status.rs
│       └── capabilities/
│           ├── mod.rs
│           └── sse.rs                # 自定义 Operation/响应协议
├── packages/                         # @apps/frontend
│   ├── package.json                  # 稳定的公开 exports
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx            # 共享布局与全局导航装配
│   │   │   └── beautifui/            # 已有样式资源
│   │   ├── features/
│   │   │   ├── configs/
│   │   │   │   ├── settings-dialog.tsx
│   │   │   │   ├── agent-config-panel.tsx
│   │   │   │   ├── use-agent-config.ts
│   │   │   │   ├── events.ts         # 生成事件的设置模块包装
│   │   │   │   ├── other-settings-panels.tsx # LLM/A2A/MCP 课程占位
│   │   │   │   └── index.ts          # @apps/frontend/configs 公开入口
│   │   │   ├── sessions/
│   │   │   │   ├── chat.tsx / new-session.tsx / novnc.tsx
│   │   │   │   ├── use-sessions.ts / events.ts
│   │   │   │   └── components/       # 消息、输入、计划、工具等会话组件
│   │   │   ├── auth/                 # 认证 UI、Hook、事件包装
│   │   │   ├── files/                # 文件 UI、Hook、事件包装
│   │   │   └── status/               # 有对应页面时实现
│   │   ├── components/
│   │   │   ├── ui/                   # 基础组件，保持 shadcn 风格
│   │   │   └── ...                   # 确实跨业务复用的品牌、导航等组件
│   │   ├── hooks/                    # 通用 UI Hook，例如屏幕尺寸
│   │   ├── lib/
│   │   │   ├── crux/
│   │   │   │   ├── core.ts           # FFI、effect 分发、实例资源管理
│   │   │   │   ├── provider.tsx      # 应用级 Core 生命周期与 Context
│   │   │   │   ├── use-crux.ts       # 消费共享 Context
│   │   │   │   ├── http.ts / sse.ts  # 网络能力执行
│   │   │   │   ├── key-value.ts / time.ts
│   │   │   │   ├── api-config.ts     # 地址解析的纯函数
│   │   │   │   └── index.ts          # 通用桥接公开接口
│   │   │   ├── novnc.ts              # 第三方库适配
│   │   │   └── utils.ts
│   │   └── index.css                # 共享主题与语义 token
│   └── tests/                        # WASM、Shell、Provider、业务集成测试
├── tanstack-app/
│   ├── vite.config.ts / .env.example
│   └── src/
│       ├── routes/                   # Web 路由、参数、根 Provider 装配
│       ├── router.tsx
│       └── styles.css
├── electron-app/
│   ├── forge.config.ts / vite.*.config.ts / .env.example
│   └── src/
│       ├── main.ts                   # 窗口、平台生命周期
│       ├── main/                    # 原生能力变多时拆 IPC/传输/安全存储
│       ├── preload.ts               # 按需暴露经过校验的窄接口
│       ├── renderer.ts
│       └── app.tsx                  # Hash 路由、配置、根 Provider 装配
├── generated/                       # 自动生成并忽略，统一用 just install
├── crux/                            # 官方参考代码，保持外部源码独立
├── docs/design/                     # 课程视觉参考
└── deploy/                          # Web 静态部署与 nginx 代理
```

单个业务变大后，把 `<业务>.rs` 改为 `<业务>/mod.rs` 并按真实子流程拆分。例如新增多个设置资源时，可使用 `configs/{mod,agent,llm,mcp_servers,a2a_servers}.rs`；对应前端设置面板随业务增加。接口数量本身不决定文件数量，一个业务流程可以组合多个接口。

测试就近放在 Rust 模块的 `#[cfg(test)]` 中，较长时拆为同模块的 `tests.rs`；根模型测试验证跨模块协调，`ffi.rs` 测试验证协议。Shell 测试沿用现有 `packages/tests/`；新增测试位置或 TSX 编译范围时同步更新测试命令和 `tests/tsconfig.json`，确保测试被实际执行。

## 3. 强制职责与依赖方向

以下箭头表示运行数据流：

```text
宿主 → 共享 UI / Provider → Event → app → model → api / capabilities → Effect
                                    model → view → ViewModel → 共享 UI
Effect → Shell 执行能力 → resolve → 模块内部事件 → model
```

代码依赖方向独立约束：`app → model/view/effects`，`model → api/effects/capabilities`，`view → model/安全的 api DTO`；API 独立于业务 Model/Event/View。共享 UI 的 `features → lib/crux + 基础组件`，`lib/crux` 保持业务无关，共享包独立于两个宿主。

| 位置 | 应当包含 | 边界 |
| --- | --- | --- |
| `app.rs` | 关联类型、委派 update/view | 保持薄入口 |
| `api/` | 路径、DTO、请求构造、必要的协议错误归一化 | 独立于业务 Model/Event/View；泛型调用方决定接收事件 |
| `model/` | 事件、状态、业务规则、重试决策、并发身份、模块协作 | 调用能力产生 Command；平台 I/O 由 Shell 执行 |
| `view/` | 展示数据与 `From<&Model>` 等转换 | 无副作用；安全 DTO 可直接复用，敏感字段必须过滤 |
| `effects.rs` / `capabilities/` | 平台操作协议和能力构造/解码 | 服务于多个模块，独立于具体页面 |
| `ffi.rs` | 字节序列化、Core 桥接 | 保持业务无关，原生与 WASM 协议一致 |
| `lib/crux/` | 通用运行时、Provider、能力执行 | 业务 Hook/命令包装放 `features/` |
| `features/<业务>/` | 页面、业务 UI Hook、生成事件包装 | Rust 管业务状态及可提交的表单草稿；React 管展开、焦点等局部 UI 状态 |
| 宿主 | 路由、根 Provider、公开配置、平台适配 | 宿主通过 `@apps/frontend/*` 入口消费共享代码 |

跨业务操作由根 Model 或明确的父流程协调。业务模块之间通过事件/转换结果交接，避免直接修改兄弟模型。API 与 View 都可复用安全 DTO，避免为目录整齐复制同形结构。

公开 Event、ViewModel 和它们包含的数据构成 FFI 契约；内部响应事件标记 `serde(skip)`、`facet(skip)`。修改嵌套事件、variant、字段、类型后，必须同时生成 WASM 和 TypeScript，并运行跨边界测试。

新增普通 HTTP API 继续复用 Http Effect 与现有 Shell。新增能力才扩展 Rust Operation、Effect、Shell handler 和测试。公共工具在出现明确复用需求时提取，保持当前依赖集合与官方 Justfile 生成流程。

## 4. Core、请求与状态生命周期

1. **每个 Web 应用根、每个 Electron 窗口持有一个 Core。** Provider 在客户端初始化并在应用根卸载时释放；业务 Hook 读取相同 view/dispatch。Provider 放在路由和布局切换外层，切换会话或进入 noVNC 时保持实例。Web 保留 ClientOnly/SPA 边界，避免服务端共享模块级单例。
2. **应用资源与业务资源分别管理。** 应用卸载 `dispose` 取消全部请求、流和定时器；业务页面关闭只处理该业务需要结束的任务。关闭设置弹窗后的读取结果是否保留，由模块策略决定。需要取消时同时明确 Shell 终止 I/O、Model 忽略过期结果的责任。
3. **业务错误归对应模块。** HTTP 失败经 `resolve` 回到 Rust 并写入该业务状态；全局错误用于初始化、FFI、协议等客户端故障。Provider 迁移时同时调整错误上报，避免一个模块请求失败污染其他模块提示。
4. **并发结果有明确身份。** 单配置读取可以用 loading 去重；搜索、会话切换、注销和并发重试使用请求 ID、版本或认证代际识别结果。迟到响应不能恢复已退出的身份或覆盖新会话状态。
5. **请求有终止条件。** 普通 HTTP Shell 默认 30 秒超时，覆盖请求及响应体读取，并与 Core 卸载取消联动；SSE 使用独立生命周期。模型决定重试策略，错误与取消路径也要结束 loading/saving。保存超时或网络中断时结果尚未确认，保留草稿并提示核对；写操作由用户手动重试。
6. **共享业务数据由 Rust 维护。** TanStack Router 负责路由；同一份业务数据保持单一状态来源。已有 TanStack Query 演示代码列入发布整理项。

Provider 的引入必须与取消策略、错误隔离和生命周期测试一起完成；当前每个 `useCrux()` 单独持有 Core 的实现是第 6 节登记的迁移项。当前设置弹窗仅在内容组件调用一次 `useAgentConfig()`，表单与底部保存按钮共享这份状态。普通关闭会卸载内容、丢弃未保存草稿，重开后重新 GET；保存期间统一阻止编辑、切换面板和关闭，待成功、失败或超时后恢复操作。

## 5. 配置、持久化、认证和平台边界

- 宿主启动时提供公开 API 基础地址，统一经过地址校验，Core/业务请求使用已确认的配置。当前事件逐次携带 `base_url` 的单接口方式，在认证及多模块接入时统一迁移。配置只有一份权威值。
- `model/auth` 管认证状态及失效流程；请求构造接收明确的地址和认证参数。API 层保持独立于业务模型，登录与退出导致的跨模块清理由根模型协调。
- 密钥、密码、token 和内部句柄不进入 ViewModel、日志或 `VITE_` 变量。浏览器认证持久化按后端契约设计；Electron 安全存储通过受限平台能力执行。官方演示中的明文 localStorage secret 存储不作为生产认证方案。
- 持久化使用明确字段、版本与恢复逻辑；正在进行的请求、loading、取消句柄和错误提示属于运行态。现有计数器 KV 格式按兼容迁移处理，新业务避免序列化整个根 Model。
- Web 生产使用同源 `/api` 代理，开发使用 `vite.api.mts`。Electron 生产构建必须明确后端地址并验证实际安装包。需要桌面专用 HTTP/安全存储时，通过 preload 的窄接口注入；校验来源和参数，保留浏览器安全机制。
- 聊天 SSE 接入时同步扩展 method、headers、body、事件名、错误和关闭/取消语义。业务解码与重连决策放在 Rust；Shell 执行传输。现有 GET 计数器 SSE 的成功不代表会话 POST SSE 已验收。
- HTTP 响应包装、时间单位、状态枚举和认证要求以当前 OpenAPI/后端源码为准，按接口真实契约建模；Agent 编排、数据库和权限校验继续由 `../server/` 负责。

## 6. 当前状态与必须完成的迁移

已完成的设置切片：

- [api/configs.rs](shared/src/api/configs.rs) 构造 GET/POST 请求；[model/configs.rs](shared/src/model/configs.rs) 管理字符串草稿、整数范围校验、读写去重、保存与失败恢复；[view/configs.rs](shared/src/view/configs.rs) 输出 data/draft/loading/saving/error/saved/dirty/can_save。`app.rs` 继续保持委派入口。
- [features/configs](packages/src/features/configs/index.ts) 集中设置弹窗、Agent 表单、业务 Hook 与事件包装，通过 `@apps/frontend/configs` 导出。通用 `lib/crux` 保留桥接与平台 I/O；旧设置组件和业务 Hook 已迁出。
- [http.ts](packages/src/lib/crux/http.ts) 已加入普通 HTTP 默认 30 秒截止时间与实例取消清理。设置保存根据服务端返回值确认结果，失败保留已确认数据和草稿，超时/网络异常显示结果待核实提示。

以下为剩余迁移项。新业务按目标落位；既有偏差按表分批消除。

| 当前证据 | 调整内容 | 完成时机 |
| --- | --- | --- |
| [model/mod.rs](shared/src/model/mod.rs) 与 [view/mod.rs](shared/src/view/mod.rs) 仍含计数器、外部演示 API、KV/时间/展示细节 | 教学业务迁出根层；保留教学阶段所需兼容，迁移相关测试后从生产 Event/ViewModel 移除。课程参考保留在 `crux/examples` | 生产发布前完成；后续产品逻辑直接进入自己的模块 |
| [use-crux.ts](packages/src/lib/crux/use-crux.ts) 每次挂载 new Core | 引入应用根 Provider，useCrux 消费共享 Context | 认证、会话等多个业务联动前完成 |
| [core.ts](packages/src/lib/crux/core.ts) 只有实例级取消，每个 HTTP Err 同时报告全局错误 | Provider 同批完成请求归属、业务取消/失效响应及全局错误隔离；补真实 Provider 挂载测试 | 与共享 Core 一起验收 |
| 会话页面与业务组件仍分散在 `packages/src/` 与 `components/` | 逐步迁到 `features/sessions/`，保留已有公开入口 | 扩展会话模块时迁移，新增业务直接按目标落位 |
| 当前 ConfigsModel 只管理 Agent 配置读写，事件逐次传 base URL | 多种设置各自管理状态；启动配置统一注入；根据实际展示需要聚合 ConfigsViewModel | 新增 LLM/MCP/A2A 或认证时完成 |
| [capabilities/sse.rs](shared/src/capabilities/sse.rs) / [Shell SSE](packages/src/lib/crux/sse.ts) 当前为 URL + GET，解码错误不可观察 | 支持后端 POST SSE 契约、事件名、可观察错误、取消/结束 | 会话实时业务接入前完成 |
| Electron 使用 file 页面，当前 [main.ts](electron-app/src/main.ts) 无条件打开 DevTools | 明确生产地址与受限传输路径，DevTools 限开发；验收安装包请求与导航 | 桌面发布前完成 |
| 首页/会话仍含演示数据，LLM/MCP/A2A 配置保存仍是占位 | 各模块逐项完成真实数据、交互与失败恢复，演示路由/数据退出生产入口 | 对应业务上线前完成 |

`@apps/frontend/configs` 公开导出 `ManusSettings` 与 `useAgentConfig`。`@apps/frontend/chat`、`/new-session`、`/novnc`、`/layout` 等公开入口可保留名称，通过 `package.json` 的 `exports` 指向迁移后的文件；共享包内部仍使用相对导入。移动 Hook 后同步测试编译范围，避免测试继续覆盖旧文件。

验证证据按当前 checkout 的实际执行结果记录。配置写流程使用 `settings-save --allow-config-write true`，记录原值，保存后重新 GET 核对，再恢复原值；未确认的写请求必须记为失败且结果待核实。剩余生产迁移项各自完成验证后更新本表。

## 7. 每次修改的执行要求

1. 阅读本文件、适用 AGENTS 和 Cursor 项目规则；确认改动属于 api/model/view、Shell、UI 或宿主哪一层。
2. 从当前源码核对相关 API 契约与现有实现；新增代码按目标目录落位。触及登记的旧目录时，按明确范围迁移，保留公开入口和已有行为覆盖。
3. 先说明改动范围与验证方式。结构调整先运行已有回归；无覆盖的关键行为先补测试。保留无关工作区改动和课程注释。
4. 核对依赖方向、Core 所有权、资源释放、错误归属、并发响应身份与敏感数据边界。具体业务流程、请求处理和状态转换放在业务模块；根层负责聚合字段、模块事件分发及跨模块协调，通用 Shell 负责能力执行。
5. 修改 Rust 后更新生成包。变更 Event/Effect/ViewModel 时核对两端编译与真实序列化往返；能力变更同步 Rust/Shell/测试。
6. 执行受影响检查并记录结果与未测范围；结构、生命周期或部署变更要验证真实 Web/Electron 场景。
7. 同步 README/教程相关内容。若新需求明确改变本规范，先修订设计说明与迁移条目，再按修订后的约定实施，避免不同文件持有相互冲突的规则。

现有检查入口，在 `apps/` 执行：

```sh
cargo fmt --all --check
cargo check -p shared
just test
cargo clippy -p shared --lib --tests -- -D warnings
just install
pnpm test:shared
pnpm test:crux
pnpm typecheck
pnpm --filter tanstack-app lint
ESLINT_USE_FLAT_CONFIG=false pnpm --filter electron-app lint
```

涉及构建或发布时继续执行 `pnpm --filter tanstack-app build`、`pnpm --filter electron-app package`。涉及 Provider 时验收多组件共享、路由/noVNC 切换、StrictMode、卸载及迟到回调；涉及跨模块请求时验收错误隔离和注销后的失效响应；涉及生产平台时验收实际静态部署或安装包。

新目录本身不代表生产可用。发布结论以真实业务、错误路径、目标平台和部署配置的验证证据为准。
