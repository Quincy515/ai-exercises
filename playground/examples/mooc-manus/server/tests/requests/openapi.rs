use loco_rs::testing::prelude::*;
use serial_test::serial;
use server::app::App;

#[tokio::test]
#[serial]
async fn exposes_openapi_json_for_auth_routes() {
    request_with_create_db::<App, _, _>(|request, ctx| async move {
        for endpoint in ["/api-docs/openapi.json", "/redoc/openapi.json"] {
            let response = request.get(endpoint).await;

            assert_eq!(response.status_code(), 200, "{endpoint} should respond");

            let document: serde_json::Value = serde_json::from_str(&response.text())
                .unwrap_or_else(|err| panic!("{endpoint} should return OpenAPI JSON: {err}"));

            assert_eq!(document["openapi"], "3.1.0");
            assert_eq!(
                document["components"]["securitySchemes"]["jwt_token"]["scheme"],
                "bearer"
            );
            assert_eq!(
                document["components"]["securitySchemes"]["api_key"]["name"],
                "apikey"
            );
            assert!(document["paths"].get("/api/auth/login").is_some());
            assert_eq!(
                document["paths"]["/api/auth/login"]["post"]["summary"],
                "用户登录"
            );
            assert_eq!(
                document["paths"]["/api/auth/login"]["post"]["responses"]["401"]["description"],
                "邮箱或密码错误"
            );
            assert!(document["paths"].get("/api/status").is_some());
            assert_eq!(
                document["paths"]["/api/status"]["get"]["summary"],
                "系统健康检查"
            );
        }
        request.get("/swagger/").await.assert_status_ok();
        request.get("/redoc").await.assert_status_ok();
        // 先关闭共享连接池，再让 Loco 删除临时数据库。
        ctx.db.close().await.expect("close test database pool");
    })
    .await;
}

#[test]
#[serial]
fn auth_openapi_uses_automatic_route_collection() {
    use server::{controllers, openapi};

    openapi::clear_routes();
    controllers::auth::routes();
    let auth = openapi::document();
    assert!(auth.paths.paths.contains_key("/api/auth/login"));
    assert!(!auth.paths.paths.contains_key("/api/files"));

    controllers::files::routes();
    let combined = openapi::document();
    assert!(combined.paths.paths.contains_key("/api/auth/login"));
    assert!(combined.paths.paths.contains_key("/api/files"));

    openapi::clear_routes();
    assert!(openapi::document().paths.paths.is_empty());
}
