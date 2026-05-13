use loco_openapi::prelude::{OpenApi, SecurityAddon};

#[derive(OpenApi)]
#[openapi(
    modifiers(&SecurityAddon),
    info(
        title = "server API",
        version = env!("CARGO_PKG_VERSION"),
        description = "服务端 API 文档"
    ),
    tags(
        (name = "认证", description = "认证相关接口")
    )
)]
pub struct ApiDoc;
