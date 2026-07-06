use axum::Router;

use super::service_dependencies::AppState;

pub struct FileController;

impl FileController {
    pub fn routes() -> Router<AppState> {
        Router::new()
    }
}
