use loco_rs::testing::prelude::*;
use serial_test::serial;
use server::app::App;

#[tokio::test]
#[serial]
async fn exposes_openapi_json_for_auth_routes() {
    request_with_create_db::<App, _, _>(|request, _ctx| async move {
        for endpoint in ["/api-docs/openapi.json", "/redoc/openapi.json"] {
            let response = request.get(endpoint).await;

            assert_eq!(response.status_code(), 200, "{endpoint} should respond");

            let document: serde_json::Value = serde_json::from_str(&response.text())
                .unwrap_or_else(|err| panic!("{endpoint} should return OpenAPI JSON: {err}"));

            assert_eq!(document["openapi"], "3.1.0");
            assert!(document["paths"].get("/api/auth/login").is_some());
            assert_eq!(
                document["paths"]["/api/auth/login"]["post"]["summary"],
                "用户登录"
            );
            assert_eq!(
                document["paths"]["/api/auth/login"]["post"]["responses"]["401"]["description"],
                "邮箱或密码错误"
            );
        }
    })
    .await;
}

#[test]
fn auth_openapi_uses_automatic_route_collection() {
    let app_source = include_str!("../../src/app.rs");
    let auth_source = include_str!("../../src/controllers/auth.rs");

    assert!(app_source.contains("None,"));
    assert!(!app_source.contains("controllers::auth::openapi_routes"));
    assert!(!auth_source.contains("pub fn openapi_routes"));
    assert!(auth_source.contains("openapi(post(register), routes!(register))"));
    assert!(auth_source.contains("openapi(get(current), routes!(current))"));
}
