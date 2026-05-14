#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_struct_initialization)]
#![allow(clippy::unused_async)]
use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "状态模块",
    summary = "系统健康检查",
    description = "监测系统的 Postgres、Redis、storage、后端服务等组件的状态信息。",
    responses(
        (status = 200, description = "系统状态正常"),
        (status = 500, description = "系统状态异常")
    )
)]
#[debug_handler]
pub async fn index(State(_ctx): State<AppContext>) -> Result<Response> {
    // TODO: 等待 Postgres、Redis、storage 等服务接入后补全代码。
    // TODO: Complete this after Postgres, Redis, storage, and related services are wired.
    format::json(())
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/status")
        .add("/", openapi(get(index), routes!(index)))
}
