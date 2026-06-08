#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_struct_initialization)]
#![allow(clippy::unused_async)]

use axum::http::StatusCode;
use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;
use validator::ValidationErrors;

use crate::{
    application::{error::AppError, services::app_config_service::McpServerNotFound},
    interfaces::service_dependencies,
    views::{
        app_config::{McpConfigRequest, McpConfigResponse, McpServerEnabledRequest},
        AgentConfigRequest, AgentConfigResponse, LlmConfigRequest, LlmConfigResponse,
    },
};

/// 获取 LLM 配置信息
#[utoipa::path(
    get,
    path = "/api/app_configs/llm",
    tag = "设置模块",
    summary = "获取 LLM 配置信息",
    description = "包含 LLM 提供商的 base_url、temperature、model_name、max_tokens；api_key 只返回是否已配置。",
    responses(
        (status = 200, description = "LLM 配置信息", body = LlmConfigResponse),
        (status = 500, description = "配置读取失败")
    )
)]
#[debug_handler]
pub async fn get_llm_config(State(ctx): State<AppContext>) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let llm_config = service
        .get_llm_config()
        .await
        .map_err(|err| AppError::internal("app_config.get_llm_config_failed", err.to_string()))?;
    format::json(LlmConfigResponse::from(llm_config))
}

/// 更新 LLM 配置信息
#[utoipa::path(
    post,
    path = "/api/app_configs/llm",
    tag = "设置模块",
    summary = "更新 LLM 配置信息",
    description = "更新 LLM 配置信息；api_key 为空时保留旧值。",
    request_body = LlmConfigRequest,
    responses(
        (status = 200, description = "LLM 配置更新成功", body = LlmConfigResponse),
        (status = 422, description = "LLM 配置校验失败"),
        (status = 500, description = "配置写入失败")
    )
)]
#[debug_handler]
pub async fn update_llm_config(
    State(ctx): State<AppContext>,
    Json(config): Json<LlmConfigRequest>,
) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let llm_config = service
        .update_llm_config(config.into())
        .await
        .map_err(|err| map_app_config_error(err, "app_config.update_llm_config_failed"))?;
    format::json(LlmConfigResponse::from(llm_config))
}

/// 获取 Agent 通用配置信息
#[utoipa::path(
    get,
    path = "/api/app_configs/agent",
    tag = "设置模块",
    summary = "获取 Agent 配置信息",
    description = "包含最大迭代次数、最大重试次数、最大搜索结果数",
    responses(
        (status = 200, description = "Agent 配置获取成功", body = AgentConfigResponse),
        (status = 500, description = "配置读取失败")
    )
)]
#[debug_handler]
pub async fn get_agent_config(State(ctx): State<AppContext>) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let agent_config = service
        .get_agent_config()
        .await
        .map_err(|err| AppError::internal("app_config.get_agent_config_failed", err.to_string()))?;
    format::json(AgentConfigResponse::from(agent_config))
}

/// 更新 Agent 配置信息
#[utoipa::path(
     post,
    path = "/api/app_configs/agent",
    tag = "设置模块",
    summary = "更新 Agent 配置信息",
    description="更新Agent通用配置信息",
    request_body = AgentConfigRequest,
    responses(
        (status = 200, description = "Agent 配置更新成功", body = AgentConfigResponse),
        (status = 422, description = "Agent 配置校验失败"),
        (status = 500, description = "配置更新失败")
    )
)]
#[debug_handler]
pub async fn update_agent_config(
    State(ctx): State<AppContext>,
    Json(config): Json<AgentConfigRequest>,
) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let agent_config = service
        .update_agent_config(config.into())
        .await
        .map_err(|err| map_app_config_error(err, "app_config.update_agent_config_failed"))?;
    format::json(AgentConfigResponse::from(agent_config))
}

/// 获取当前系统的 MCP 服务器工具列表
#[utoipa::path(
    get,
    path = "/api/app_configs/mcp-servers",
    tag = "设置模块",
    summary = "获取 MCP 服务器工具列表",
    description = "获取当前系统的 MCP 服务器列表，包含 MCP 服务名字、工具列表、启用状态等",
    responses(
        (status = 200, description = "MCP 服务器工具获取成功", body = McpConfigResponse),
        (status = 500, description = "MCP 服务器工具获取失败")
    )
)]
#[debug_handler]
pub async fn get_mcp_servers(State(ctx): State<AppContext>) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let config = service
        .get_mcp_config()
        .await
        .map_err(|err| AppError::internal("app_config.get_mcp_servers_failed", err.to_string()))?;
    format::json(McpConfigResponse::from(config))
}

/// 根据传递的配置信息创建 MCP 服务
#[utoipa::path(
    post,
    path = "/api/app_configs/mcp-servers",
    tag = "设置模块",
    summary = "新增 MCP 服务配置，支持传递一个或者多个配置",
    description="传递 MCP 配置信息为系统新增 MCP 工具",
    request_body = McpConfigRequest,
    responses(
        (status = 200, description = "MCP 服务配置新增成功", body = McpConfigResponse),
        (status = 422, description = "MCP 服务配置校验失败"),
        (status = 500, description = "MCP 服务配置更新失败")
    )
)]
#[debug_handler]
pub async fn create_mcp_servers(
    State(ctx): State<AppContext>,
    Json(mcp_config): Json<McpConfigRequest>,
) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let config = service
        .update_and_create_mcp_servers(mcp_config.into())
        .await
        .map_err(|err| map_app_config_error(err, "app_config.create_mcp_servers_failed"))?;
    format::json(McpConfigResponse::from(config))
}

/// 根据服务名字删除 MCP 服务器
#[utoipa::path(
    post,
    path = "/api/app_configs/mcp-servers/{server_name}/delete",
    tag = "设置模块",
    summary = "删除 MCP 服务器",
    description="根据传递的 MCP 服务名字删除指定的 MCP 服务器",
    params(
        ("server_name" = String, Path, description = "MCP 服务名称")
    ),
    responses(
        (status = 200, description = "MCP 服务配置删除成功", body = McpConfigResponse),
        (status = 404, description = "MCP 服务不存在"),
        (status = 500, description = "MCP 服务配置更新失败")
    )
)]
#[debug_handler]
pub async fn delete_mcp_server(
    State(ctx): State<AppContext>,
    Path(server_name): Path<String>,
) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let config = service
        .delete_mcp_server(&server_name)
        .await
        .map_err(|err| map_app_config_error(err, "app_config.delete_mcp_server_failed"))?;
    format::json(McpConfigResponse::from(config))
}

/// 根据传递的 server_name+enabled 更新服务的启用状态
#[utoipa::path(
    post,
    path = "/api/app_configs/mcp-servers/{server_name}/enabled",
    tag = "设置模块",
    summary = "更新 MCP 服务器启用状态",
    description="根据传递的 server_name + enabled 更新指定 MCP 服务器的启用状态",
    request_body = McpServerEnabledRequest,
    params(
        ("server_name" = String, Path, description = "MCP 服务名称")
    ),
    responses(
        (status = 200, description = "MCP 服务启用状态更新成功", body = McpConfigResponse),
        (status = 404, description = "MCP 服务不存在"),
        (status = 500, description = "MCP 服务配置更新失败")
    )
)]
#[debug_handler]
pub async fn set_mcp_server_enabled(
    State(ctx): State<AppContext>,
    Path(server_name): Path<String>,
    Json(request): Json<McpServerEnabledRequest>,
) -> Result<Response> {
    let service = service_dependencies::get_app_config_service(&ctx);
    let config = service
        .set_mcp_server_enabled(&server_name, request.enabled)
        .await
        .map_err(|err| map_app_config_error(err, "app_config.set_mcp_server_enabled_failed"))?;
    format::json(McpConfigResponse::from(config))
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/app_configs")
        .add(
            "/llm",
            openapi(
                get(get_llm_config).post(update_llm_config),
                routes!(get_llm_config, update_llm_config),
            ),
        )
        .add(
            "/agent",
            openapi(
                get(get_agent_config).post(update_agent_config),
                routes!(get_agent_config, update_agent_config),
            ),
        )
        .add(
            "/mcp-servers",
            openapi(
                get(get_mcp_servers).post(create_mcp_servers),
                routes!(get_mcp_servers, create_mcp_servers),
            ),
        )
        .add(
            "/mcp-servers/{server_name}/delete",
            openapi(post(delete_mcp_server), routes!(delete_mcp_server)),
        )
        .add(
            "/mcp-servers/{server_name}/enabled",
            openapi(
                post(set_mcp_server_enabled),
                routes!(set_mcp_server_enabled),
            ),
        )
}

fn map_app_config_error(err: anyhow::Error, code: &'static str) -> AppError {
    if err.is::<McpServerNotFound>() {
        AppError::business(StatusCode::NOT_FOUND, code, err.to_string(), None)
    } else if let Some(validation_errors) = err.downcast_ref::<ValidationErrors>() {
        let details = serde_json::to_value(validation_errors).unwrap_or_else(|_| {
            serde_json::json!({
                "message": validation_errors.to_string()
            })
        });

        AppError::validation(code, err.to_string(), details)
    } else {
        AppError::internal(code, err.to_string())
    }
}
