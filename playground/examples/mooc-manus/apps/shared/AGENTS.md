# Crux 客户端业务分层

每次修改先阅读 [../ARCHITECTURE.md](../ARCHITECTURE.md) 和上级 `AGENTS.md`。本项目采用 `../crux/examples/weather` 的职责划分。设置、认证、会话、文件和后续业务统一遵循以下约定：

- `src/app.rs`：连接 App 关联类型，`update` 委派给根 Model，`view` 使用 ViewModel 转换。
- `src/api/<业务>.rs`：接口路径、请求/响应 DTO、HTTP 请求构造；保持独立于业务 Model 和 Event。
- `src/model/<业务>.rs`：模块事件、内部状态、业务规则和更新流程；加载、重试、去重和错误状态在这里管理。
- `src/model/mod.rs`：聚合根 Event/Model，分发模块事件并处理跨模块协调；子 Command 通过 `map_event` 接入父事件。
- `src/view/<业务>.rs`：模块 ViewModel 和 `From<&模块Model>` 转换；根 ViewModel 在 `src/view/mod.rs` 汇总。
- `src/effects.rs`：统一 Effect 协议；自定义能力实现继续放 `src/capabilities/`，Shell 执行代码在 `../packages/src/lib/crux/`。

业务模块按实际接入创建，如 `configs`、`auth`、`sessions`、`files`、`status`；单个模块变大后再拆子目录。计数器教学逻辑属于 ARCHITECTURE 第 6 节登记的兼容迁移项，教学阶段保持已有行为和测试；迁移相关测试后退出生产 Event/ViewModel。新增产品逻辑按业务模块落位。

根 Model 聚合状态、路由和跨模块协调。新业务的持久化使用明确字段与版本，运行中请求和临时提示独立管理。并发、切换会话、退出登录要识别并丢弃失效响应；业务错误写入对应模块状态。

跨 FFI 的 Event/ViewModel 使用 Facet 与 Serde；内部响应事件标记 `serde(skip)`、`facet(skip)`。内部 Model 按实际需求派生类型，生成产物由 `../Justfile` 维护。

在 `apps/` 执行 Rust 测试、`just install`、WASM/Shell 测试与两端类型检查。修改 Event/ViewModel 后同步更新 Shell 的生成类型引用，并核对序列化往返、失败重试和卸载清理。
