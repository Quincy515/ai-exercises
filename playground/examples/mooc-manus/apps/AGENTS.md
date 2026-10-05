# 客户端工作区强制架构约定

本目录及所有子目录的代码修改，必须先阅读并遵守 [ARCHITECTURE.md](ARCHITECTURE.md)。它是客户端目录、依赖方向、Core 生命周期、业务边界和验证要求的统一依据；`shared/AGENTS.md` 补充 Rust 细节。

- 新增业务按 `shared/src/api/`、`model/`、`view/` 分工，`app.rs` 保持委派入口，根 Model 负责聚合和跨模块协调。
- 前端业务 UI、Hook、事件包装按 `packages/src/features/<业务>/` 归组；通用桥接留在 `packages/src/lib/crux/`；宿主负责路由、配置和平台能力。
- 生产多模块架构采用每个应用根/窗口一个 Core。Provider 迁移与取消策略、错误隔离、跨组件/路由生命周期测试一起完成。
- 新代码严格按目标边界落位。当前已登记的兼容偏差按 ARCHITECTURE 第 6 节迁移，后续改动不得扩大这些偏差；避免与任务无关的批量搬移或空目录占位。
- React 保持业务状态来源为 Rust；平台 I/O 通过 Shell 执行。生成产物使用 `just install` 更新，官方参考代码保持独立。
- 修改前说明所属层、范围和验证方式；修改后完成受影响测试、类型与 lint 检查，并同步相关文档。新增能力同步 Rust 协议、Shell 和测试。
- 需要根据新需求调整规范时，先修订 ARCHITECTURE 的设计说明和迁移记录；具体用户指令优先，变更后的规则同步到其他入口。

API 契约、UI 设计、工具链及课程约定同时参照 `.cursor/rules/general.mdc`。本规范约束每次实现，生产发布还需完成对应迁移项与目标平台验收。
