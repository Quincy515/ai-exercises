use loco_rs::testing::prelude::*;
use serial_test::serial;
use server::app::App;

#[tokio::test]
#[serial]
async fn can_get_status() {
    request::<App, _, _>(|request, _ctx| async move {
        let res = request.get("/api/status").await;
        assert_eq!(res.status_code(), 200);
        res.assert_json(&serde_json::json!(null));
    })
    .await;
}
