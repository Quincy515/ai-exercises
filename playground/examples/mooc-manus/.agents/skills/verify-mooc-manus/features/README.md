# MoocManus 验证地图

本目录维护当前 Web 用户流程。先阅读索引，再执行对应 feature。Electron 的 Hash 路由是另一表面，Web 证明不自动覆盖桌面窗口或安装包。

## 基线

- 用 `scripts/verify.mjs launch/run` 创建独占实例，默认 `http://127.0.0.1:4317`。
- 先执行 doctor；前端源码和生成包指纹必须与启动时一致。
- 每个流程使用隔离 context、1280×900；当前 Web 页面无需登录。
- 设置查询需 `localhost:5150` 后端，其他两项使用现有演示 UI，服务端无写操作。
- 所选入口全部验证后才标记该 feature passed；条件不足记录 blocked，未接通按钮记录 unimplemented。

## Features

| ID | 用户流程 | 当前边界 | 地图 |
| --- | --- | --- | --- |
| `settings` | 打开设置，查看并刷新 Agent 配置 | 真实 API / WASM / UI；后端不可用则 blocked | [Agent 配置](agent-settings.md) |
| `sessions` | 进入会话、展开收起计划、返回首页 | 真实路由与交互，演示会话和计划数据 | [会话与计划](session-plan.md) |
| `files` | 查看任务文件列表并关闭 | 真实 Dialog 交互，演示文件数据 | [任务文件](task-files.md) |

## 证明边界

一键 `run --features settings,sessions,files` 顺序运行各流程并分别报告。每个结果绑定 HEAD、working-tree digest 和入口列表。发送聊天、上传/下载、消息区查看全部、认证、noVNC 和生产 Electron 都不在本次通过声明内；新增这些能力后扩展地图及受控数据清理。
