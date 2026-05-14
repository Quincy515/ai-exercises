use loco_openapi::prelude::{OpenApi, SecurityAddon};

#[derive(OpenApi)]
#[openapi(
    modifiers(&SecurityAddon),
    info(
        title = "MoocManus 通用智能体",
        version = env!("CARGO_PKG_VERSION"),
        description = "MoocManus 是一个通用的 AI Agent 系统，可以完全私有部署，使用 A2A+MCP 连接 Agent/Tool，同时支持沙箱中运行各种内置工具和操作"
    ),
    tags(
        (name = "认证", description = "认证相关接口"),
        (name = "状态模块", description = "包含 **状态监测** 等 API 接口，用于监测系统的运行状态"),
    )
)]
pub struct ApiDoc;
