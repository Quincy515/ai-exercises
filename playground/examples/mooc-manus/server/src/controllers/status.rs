#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_struct_initialization)]
#![allow(clippy::unused_async)]
use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;

use crate::{interfaces::service_dependencies, views::HealthStatusResponse};

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "状态模块",
    summary = "系统健康检查",
    description = "监测系统的 Postgres、Redis、storage、后端服务等组件的状态信息。",
    responses(
        (status = 200, description = "系统健康检查结果", body = Vec<HealthStatusResponse>),
        (status = 500, description = "系统状态异常")
    )
)]
#[debug_handler]
pub async fn get_status(State(ctx): State<AppContext>) -> Result<Response> {
    let status_service = service_dependencies::get_status_service(&ctx);
    let data: Vec<HealthStatusResponse> = status_service
        .check_all()
        .await
        .into_iter()
        .map(HealthStatusResponse::from)
        .collect();

    format::json(data)
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/status")
        .add("/", openapi(get(get_status), routes!(get_status)))
}
