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
| `sessions` | 进入会话、展开收起计划、返回首页 | 真实路由与交互，演示会话和计划数据 | [会话与计划](session-plan.md) |
| `files` | 查看任务文件列表并关闭 | 真实 Dialog 交互，演示文件数据 | [任务文件](task-files.md) |

## 证明边界

一键 `run --features settings,sessions,files` 顺序运行各流程并分别报告。每个结果绑定 HEAD、working-tree digest 和入口列表。发送聊天、上传/下载、消息区查看全部、认证、noVNC 和生产 Electron 都不在本次通过声明内；新增这些能力后扩展地图及受控数据清理。

`settings-save` 仅在明确选择且本次命令带 `--allow-config-write true` 时运行。它保留原值，合法目标只改最大迭代次数 ±1；默认完成 UI 恢复，失败路径先核对所有 POST 已收到完整响应，再仅在当前值仍等于本次目标时补偿。超时/断网记录 `outcome-unknown` 并保留原值与 `post-outcomes.json`，自动恢复保持未证实。第三方值冲突写入 `cleanupError` 并保留证据。检查 `config-cleanup.json` 证明配置恢复，检查外层 `cleanup.json` 证明测试进程清理。
