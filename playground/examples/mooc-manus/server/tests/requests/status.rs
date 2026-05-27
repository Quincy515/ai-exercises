use loco_rs::testing::prelude::*;
use serial_test::serial;
use server::app::App;

#[tokio::test]
#[serial]
async fn can_get_status() {
    request::<App, _, _>(|request, _ctx| async move {
        let res = request.get("/api/status").await;
        assert_eq!(res.status_code(), 200);
        let body: serde_json::Value = serde_json::from_str(&res.text()).unwrap();
        assert!(body.as_array().is_some_and(|items| !items.is_empty()));
    })
    .await;
}
