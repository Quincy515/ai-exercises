use std::sync::Arc;

use crate::services::ShellService;

/// 应用共享状态 / shared application state.
#[derive(Clone)]
pub struct AppState {
    pub shell_service: Arc<ShellService>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            shell_service: get_shell_service(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// 获取 Shell 服务依赖 / get shell service dependency.
pub fn get_shell_service() -> Arc<ShellService> {
    Arc::new(ShellService::new())
}
