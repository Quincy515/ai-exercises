pub mod controllers;
pub mod core;
pub mod models;
pub mod services;

use axum::Router;
use tower_http::cors::CorsLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub use controllers::*;

pub fn create_app() -> Router {
    Router::new()
        .nest("/api", controllers::create_api_routes())
        .merge(
            SwaggerUi::new("/swagger")
                .url("/api-docs/openapi.json", controllers::ApiDoc::openapi()),
        )
        .layer(CorsLayer::permissive())
}
