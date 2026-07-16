pub mod controllers;
pub mod core;
pub mod models;
pub mod services;
pub mod views;

use axum::{Router, middleware};
use tower_http::{catch_panic::CatchPanicLayer, cors::CorsLayer};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub use controllers::*;

pub fn create_app() -> Router {
    let app_state = controllers::service_dependencies::AppState::new();
    let api_routes = controllers::create_api_routes().with_state(app_state.clone());

    Router::new()
        .nest("/api", api_routes)
        .merge(
            SwaggerUi::new("/swagger")
                .url("/api-docs/openapi.json", controllers::ApiDoc::openapi()),
        )
        .fallback(controllers::exceptions::not_found)
        .layer(CatchPanicLayer::custom(
            controllers::exceptions::handle_panic,
        ))
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn_with_state(
            app_state,
            core::auto_extend_timeout_middleware,
        ))
}
