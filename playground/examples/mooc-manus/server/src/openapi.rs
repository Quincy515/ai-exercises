use std::sync::{LazyLock, Mutex};

use axum::routing::MethodRouter;
use loco_rs::app::AppContext;
use utoipa::{openapi::OpenApi as OpenApiDocument, OpenApi};
use utoipa_axum::router::{OpenApiRouter, UtoipaMethodRouter};

pub use utoipa_axum::routes;

// 只收集文档，不保留路由和应用上下文中的数据库连接。
static ROUTE_DOCUMENT: LazyLock<Mutex<OpenApiDocument>> =
    LazyLock::new(|| Mutex::new(OpenApiDocument::default()));

pub fn clear_routes() {
    *ROUTE_DOCUMENT.lock().expect("OpenAPI document lock") = OpenApiDocument::default();
}

/// 注册 Loco 路由时自动收集 Utoipa 的路径和请求、响应 schema。
pub fn openapi(
    method: MethodRouter<AppContext>,
    metadata: UtoipaMethodRouter<AppContext>,
) -> MethodRouter<AppContext> {
    let (_, document) = OpenApiRouter::new().routes(metadata).split_for_parts();
    ROUTE_DOCUMENT
        .lock()
        .expect("OpenAPI document lock")
        .merge(document);
    method
}

pub fn document() -> OpenApiDocument {
    let mut document = ApiDoc::openapi();
    document.merge(
        ROUTE_DOCUMENT
            .lock()
            .expect("OpenAPI document lock")
            .clone(),
    );
    document
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "MoocManus 通用智能体",
        version = env!("CARGO_PKG_VERSION"),
        description = "MoocManus 是一个通用的 AI Agent 系统，可以完全私有部署，使用 A2A+MCP 连接 Agent/Tool，同时支持沙箱中运行各种内置工具和操作"
    ),
    tags(
        (name = "认证", description = "认证相关接口"),
        (name = "状态模块", description = "包含 **状态监测** 等 API 接口，用于监测系统的运行状态"),
        (name = "文件模块", description = "文件上传、基础信息查询与流式下载"),
        (name = "设置模块", description = "应用配置信息，包含 **Agent** 配置、**LLM** 提供商配置、**A2A** 网络配置、**MCP** 服务配置等"),
    )
)]
pub struct ApiDoc;
