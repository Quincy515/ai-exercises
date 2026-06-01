#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_struct_initialization)]
#![allow(clippy::unused_async)]

use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;

use crate::{
    application::error::AppError,
    interfaces::service_dependencies,
    views::{AgentConfigRequest, AgentConfigResponse, LlmConfigRequest, LlmConfigResponse},
};

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
        .map_err(|err| {
            AppError::internal("app_config.update_llm_config_failed", err.to_string())
        })?;
    format::json(LlmConfigResponse::from(llm_config))
}

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
        .map_err(|err| {
            AppError::internal("app_config.update_agent_config_failed", err.to_string())
        })?;
    format::json(AgentConfigResponse::from(agent_config))
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
}
