#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_struct_initialization)]
#![allow(clippy::unused_async)]

use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;

#[utoipa::path(
    get,
    path = "/api/app_configs/llm",
    tag = "设置模块",
    summary = "获取 LLM 配置信息",
    description = "包含 LLM 提供商的 base_url、temperature、model_name、max_tokens；api_key 只返回是否已配置。",
    responses(
        (status = 200, description = "LLM 配置信息"),
        (status = 500, description = "配置读取失败")
    )
)]
#[debug_handler]
pub async fn get_llm_config(State(_ctx): State<AppContext>) -> Result<Response> {
    format::empty()
}

#[utoipa::path(
    post,
    path = "/api/app_configs/llm",
    tag = "设置模块",
    summary = "更新 LLM 配置信息",
    description = "更新 LLM 配置信息；api_key 为空时保留旧值。",
    responses(
        (status = 200, description = "LLM 配置更新成功"),
        (status = 500, description = "配置写入失败")
    )
)]
#[debug_handler]
pub async fn update_llm_config(State(_ctx): State<AppContext>) -> Result<Response> {
    format::empty()
}

pub fn routes() -> Routes {
    Routes::new().prefix("/api/app_configs").add(
        "/llm",
        openapi(
            get(get_llm_config).post(update_llm_config),
            routes!(get_llm_config, update_llm_config),
        ),
    )
}
