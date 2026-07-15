use std::sync::Arc;

use crate::services::{FileService, ShellService, SupervisorService};

/// 应用共享状态 / shared application state.
#[derive(Clone)]
pub struct AppState {
    pub shell_service: Arc<ShellService>,
    pub file_service: Arc<FileService>,
    pub supervisor_service: Arc<SupervisorService>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            shell_service: get_shell_service(),
            file_service: get_file_service(),
            supervisor_service: get_supervisor_service(),
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

/// 获取文件服务依赖 / get file service dependency.
pub fn get_file_service() -> Arc<FileService> {
    Arc::new(FileService::new())
}

/// 获取 Supervisor 服务依赖 / get supervisor service dependency.
pub fn get_supervisor_service() -> Arc<SupervisorService> {
    Arc::new(SupervisorService::new())
}
