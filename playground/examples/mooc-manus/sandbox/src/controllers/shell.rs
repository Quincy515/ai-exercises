use axum::Router;

pub struct ShellController;

impl ShellController {
    pub fn routes() -> Router {
        Router::new()
    }
}
