use axum::Router;

pub struct FileController;

impl FileController {
    pub fn routes() -> Router {
        Router::new()
    }
}
