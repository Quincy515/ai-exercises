# MoocManus 验证地图

本目录维护当前 Web 用户流程。先阅读索引，再执行对应 feature。Electron 的 Hash 路由是另一表面，Web 证明不自动覆盖桌面窗口或安装包。

## 基线

- 用 `scripts/verify.mjs launch/run` 创建独占实例，默认 `http://127.0.0.1:4317`。
- 先执行 doctor；前端源码和生成包指纹必须与启动时一致。
- 每个流程使用隔离 context、1280×900；当前 Web 页面无需登录。
- 设置查询与保存需 `localhost:5150` 后端；默认三流程仅查询或使用演示 UI。`settings-save` 显式授权后使用独占开发后端临时保存并恢复原值。
- 所选入口全部验证后才标记该 feature passed；条件不足记录 blocked，未接通按钮记录 unimplemented。

## Features

| ID | 用户流程 | 当前边界 | 地图 |
| --- | --- | --- | --- |
| `settings` | 打开设置，查看并刷新 Agent 配置 | 真实 API / WASM / UI；后端不可用则 blocked | [Agent 配置](agent-settings.md) |
| `settings-save` | 编辑校验、真实保存、关闭重开核对、恢复原值 | 显式 `--allow-config-write true`；首页保存路径；其他三个保存入口未验证 | [Agent 配置](agent-settings.md) |
| `llm` | 模型提供商读取、刷新、密码空值及配置状态 | 真实本地 API；四入口读取；公开字段白名单 | [模型提供商](llm-settings.md) |
| `llm-save` | 校验温度、修改 max_tokens、重开核对、恢复原值 | 显式写授权；首页保存；始终省略 api_key；含 null 恢复 | [模型提供商](llm-settings.md) |
| `a2a` | A2A Agent列表读取刷新 | 四入口；只包含卡片加载成功的公开可见项 | [A2A Agent](a2a-settings.md) |
| `a2a-write` | 本次唯一Card的新增、停用、启用、删除 | 显式授权；本机后端前提；仅owned ID；首页写；异常保存恢复证据 | [A2A Agent](a2a-settings.md) |
| `mcp` | MCP 服务器读取刷新与工具发现 | 四入口；公开元信息；后端会探测既有启用项 | [MCP 服务器](mcp-settings.md) |
| `mcp-write` | 本次配置新增、同名更新、启停和删除 | 显式授权；本机HTTP夹具；首次name+指纹证明；秘密响应脱敏 | [MCP 服务器](mcp-settings.md) |
| `sessions` | 进入会话、展开收起计划、返回首页 | 真实路由与交互，演示会话和计划数据 | [会话与计划](session-plan.md) |
| `files` | 查看任务文件列表并关闭 | 真实 Dialog 交互，演示文件数据 | [任务文件](task-files.md) |

## 证明边界

一键 `run --features settings,sessions,files` 顺序运行各流程并分别报告。每个结果绑定 HEAD、working-tree digest 和入口列表。发送聊天、上传/下载、消息区查看全部、认证、noVNC 和生产 Electron 都不在本次通过声明内；新增这些能力后扩展地图及受控数据清理。

`settings-save` 仅在明确选择且本次命令带 `--allow-config-write true` 时运行。它保留原值，合法目标只改最大迭代次数 ±1；默认完成 UI 恢复，失败路径先核对所有 POST 已收到完整响应，再仅在当前值仍等于本次目标时补偿。超时/断网记录 `outcome-unknown` 并保留原值与 `post-outcomes.json`，自动恢复保持未证实。第三方值冲突写入 `cleanupError` 并保留证据。检查 `config-cleanup.json` 证明配置恢复，检查外层 `cleanup.json` 证明测试进程清理。

LLM 流程独立选择：`run --features llm` 或 `run --features llm-save --allow-config-write true`。只访问本地配置 API，GET/POST 证据只含四个普通字段与密钥是否配置。超 JS 安全整数先 blocked；关闭原始 trace/ARIA，保留遮罩截图和白名单 JSON。

A2A写流程仅注册本次nonce的本地受控Card；完整业务HTTP真实访问5150。先验证本机后端归属，未知创建保持仅一次，GET定位唯一owned ID，原有项禁止写。结果范围限于前后可见列表；Card不可见的数据库记录保持未验证。恢复夹具命令详见A2A地图。

MCP 写流程只创建本次 nonce 的 loopback Streamable HTTP 服务，禁止新增 stdio、凭据和外部地址；所有业务 POST 的完整配置响应仅在内存读取，日志只保留安全元信息。创建一次，未知结果保留恢复证据并由负责恢复的代理核对；详见 MCP 地图。
