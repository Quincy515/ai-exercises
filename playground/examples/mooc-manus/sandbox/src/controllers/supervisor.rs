use axum::Router;

use super::service_dependencies::AppState;

pub struct SupervisorController;

impl SupervisorController {
    pub fn routes() -> Router<AppState> {
        Router::new()
    }
}
