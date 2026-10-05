use chrono::{TimeZone, Utc};

use crux_core::App as _;
use crux_http::{
    HttpError,
    protocol::{HttpRequest, HttpResponse, HttpResult},
    testing::ResponseBuilder,
};
use crux_time::{Instant, TimeRequest, TimeResponse};

use super::{Event, Model};
use crate::{AppCore, Count};

#[test]
fn http_failure_preserves_the_current_view() {
    let app = AppCore::default();
    let mut model = Model::default();
    model.count.value = 7;
    let mut command = app.update(Event::Get, &mut model);
    let mut request = command.expect_one_effect().expect_http();
    request
        .resolve(HttpResult::Err(HttpError::Io("offline".to_string())))
        .unwrap();
    let mut command = app.update(command.expect_one_event(), &mut model);
    command.expect_one_effect().expect_render();
    assert_eq!(app.view(&model).text, "7 (pending)");
}

// ANCHOR: simple_tests
/// Test that a `Get` event causes the app to fetch the current
/// counter-value from the web API
#[test]
fn get_counter() {
    let app = AppCore::default();
    let mut model = Model::default();

    // send a `Get` event to the app
    let mut cmd = app.update(Event::Get, &mut model);

    // the app should emit an HTTP request to fetch the counter
    let (operation, mut request) = cmd.expect_one_effect().expect_http().split();

    // and the request should be a GET to the correct URL
    assert_eq!(
        operation,
        HttpRequest::get("https://crux-counter.fly.dev/").build()
    );

    // resolve the request with a simulated response from the web API
    request
        .resolve(HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{ "value": 1, "updated_at": 1672531200000 }"#)
                .build(),
        ))
        .unwrap();

    // the app should emit a `Set` event with the HTTP response
    let actual = cmd.expect_one_event();
    let expected = Event::Set(Ok(ResponseBuilder::ok()
        .body(Count {
            value: 1,
            updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
        })
        .build()));
    assert_eq!(actual, expected);

    // send the `Set` event back to the app
    let mut cmd = app.update(actual, &mut model);

    // check in flight that the app has not been updated with the server data
    let view = app.view(&model);
    assert_eq!(view.text, "0 (pending)");

    // this should generate an `Update` event
    let event = cmd.expect_one_event();
    assert_eq!(
        event,
        Event::Update(Count {
            value: 1,
            updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
        })
    );

    // send the `Update` event back to the app
    let mut cmd = app.update(event, &mut model);

    // the model should be updated
    assert_eq!(
        model.count,
        Count {
            value: 1,
            updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
        }
    );

    // the app should ask the shell for the current time
    let mut time_request = cmd.expect_effect().expect_time();
    assert_eq!(time_request.operation, TimeRequest::Now);

    // the app should ask the shell to render
    cmd.expect_one_effect().expect_render();

    // resolve the clock request and update the model with a fixed time
    time_request
        .resolve(TimeResponse::Now {
            instant: Instant::new(1_672_531_200, 0),
        })
        .unwrap();
    let event = cmd.expect_one_event();
    assert!(matches!(event, Event::CurrentTime(_)));
    let mut cmd = app.update(event, &mut model);
    cmd.expect_one_effect().expect_render();
    assert_eq!(model.time.as_deref(), Some("2023-01-01T00:00:00Z"));

    // the view should be updated
    let view = app.view(&model);
    assert_eq!(view.text, "1 (2023-01-01 00:00:00 UTC)");
    assert!(view.confirmed);
}
// ANCHOR_END: simple_tests

// Test that an `Increment` event causes the app to increment the counter
#[test]
fn increment_counter() {
    let app = AppCore::default();

    // set up our initial model as though we've previously fetched the counter
    let mut model = Model {
        count: Count {
            value: 1,
            updated_at: Some(Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 0).unwrap()),
        },
        ..Default::default()
    };

    // send an `Increment` event to the app
    let mut cmd = app.update(Event::Increment, &mut model);

    // the app should ask the shell to render the optimistic update
    cmd.expect_effect().expect_render();

    // and send an HTTP post
    let mut request = cmd.expect_one_effect().expect_http();
    assert_eq!(
        &request.operation,
        &HttpRequest::post("https://crux-counter.fly.dev/inc").build()
    );

    // we are expecting our model to be updated "optimistically" before the
    // HTTP request completes, so the value should have been updated
    // but not the timestamp
    assert_eq!(
        model.count,
        Count {
            value: 2,
            updated_at: None
        }
    );

    // resolve the request with a simulated response from the web API
    request
        .resolve(HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{ "value": 2, "updated_at": 1672531200000 }"#)
                .build(),
        ))
        .unwrap();

    // this should generate a `Set` event
    let event = cmd.expect_one_event();
    assert!(matches!(event, Event::Set(_)));

    // send the `Set` event back to the app
    let mut cmd = app.update(event, &mut model);

    // this should generate an `Update` event
    let event = cmd.expect_one_event();
    assert_eq!(
        event,
        Event::Update(Count {
            value: 2,
            updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
        })
    );

    // send the `Update` event back to the app
    let mut cmd = app.update(event, &mut model);

    // the app should ask the shell for the current time
    let time_request = cmd.expect_effect().expect_time();
    assert_eq!(time_request.operation, TimeRequest::Now);

    // the app should ask the shell to render
    cmd.expect_one_effect().expect_render();

    // the model should be updated
    insta::assert_yaml_snapshot!(model, @r#"
    count:
      value: 2
      updated_at: "2023-01-01T00:00:00Z"
    time: ~
    "#);
}

/// Test that a `Decrement` event causes the app to decrement the counter
#[test]
fn decrement_counter() {
    let app = AppCore::default();

    // set up our initial model as though we've previously fetched the counter
    let mut model = Model {
        count: Count {
            value: 0,
            updated_at: Some(Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 0).unwrap()),
        },
        ..Default::default()
    };

    // send a `Decrement` event to the app
    let mut update = app.update(Event::Decrement, &mut model);

    // the app should ask the shell to render the optimistic update
    update.expect_effect().expect_render();

    // and send an HTTP post
    let mut request = update.expect_one_effect().expect_http();
    assert_eq!(
        &request.operation,
        &HttpRequest::post("https://crux-counter.fly.dev/dec").build()
    );

    // we are expecting our model to be updated "optimistically" before the
    // HTTP request completes, so the value should have been updated
    // but not the timestamp
    assert_eq!(
        model.count,
        Count {
            value: -1,
            updated_at: None
        }
    );

    // resolve the request with a simulated response from the web API
    request
        .resolve(HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{ "value": -1, "updated_at": 1672531200000 }"#)
                .build(),
        ))
        .unwrap();

    // this should generate a `Set` event
    let event = update.expect_one_event();
    assert!(matches!(event, Event::Set(_)));

    // send the `Set` event back to the app
    let mut update = app.update(event, &mut model);

    // this should generate an `Update` event
    let event = update.expect_one_event();
    assert_eq!(
        event,
        Event::Update(Count {
            value: -1,
            updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
        })
    );

    // send the `Update` event back to the app
    let mut update = app.update(event, &mut model);

    // the app should ask the shell for the current time
    let time_request = update.expect_effect().expect_time();
    assert_eq!(time_request.operation, TimeRequest::Now);

    // the app should ask the shell to render
    update.expect_one_effect().expect_render();

    // the model should be updated
    insta::assert_yaml_snapshot!(model, @r#"
    count:
      value: -1
      updated_at: "2023-01-01T00:00:00Z"
    time: ~
    "#);
}
