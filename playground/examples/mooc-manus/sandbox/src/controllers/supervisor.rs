use axum::Router;

pub struct SupervisorController;

impl SupervisorController {
    pub fn routes() -> Router {
        Router::new()
    }
}
