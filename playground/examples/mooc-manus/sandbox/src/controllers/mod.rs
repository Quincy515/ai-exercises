use axum::Router;
use utoipa::OpenApi;

use service_dependencies::AppState;

pub mod exceptions;
pub mod file;
pub mod service_dependencies;
pub mod shell;
pub mod supervisor;

#[derive(OpenApi)]
#[openapi(
    paths(
        file::read_file,
        file::write_file,
        shell::exec_command,
        shell::view_shell,
        shell::wait_for_process,
        shell::write_to_process,
        shell::kill_process,
    ),
    info(
        title = "MoocManus沙箱系统",
        version = "1.0.0",
        description = "该沙箱系统中预装了Chrome、Python、Node.js，支持运行 Shell 命令、文件管理等功能"
    ),
    tags(
        (name = "文件模块", description = "包含 **文件增删改查** 等 API 接口，用于实现对沙箱文件的操作。"),
        (name = "Shell模块", description = "包含 **执行/查看Shell** 等 API 接口，用于实现操控沙箱内部的 Shell 命令。"),
        (name = "Supervisor模块", description = "使用接口+Supervisor实现管理沙箱系统的程序逻辑"),
    )
)]
pub struct ApiDoc;

/// 创建 API 路由，涵盖整个沙箱项目的所有 API
pub fn create_api_routes() -> Router<AppState> {
    Router::new()
        .nest("/file", file::FileController::routes())
        .nest("/shell", shell::ShellController::routes())
        .nest("/supervisor", supervisor::SupervisorController::routes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_doc_matches_course_metadata() {
        let openapi = ApiDoc::openapi();

        assert_eq!(openapi.info.title, "MoocManus沙箱系统");
        assert_eq!(openapi.info.version, "1.0.0");
        assert_eq!(
            openapi.info.description.as_deref(),
            Some("该沙箱系统中预装了Chrome、Python、Node.js，支持运行 Shell 命令、文件管理等功能")
        );

        let tags = openapi.tags.unwrap_or_default();
        let tag_names: Vec<_> = tags.iter().map(|tag| tag.name.as_str()).collect();
        assert_eq!(tag_names, ["文件模块", "Shell模块", "Supervisor模块"]);

        let path_names: Vec<_> = openapi.paths.paths.keys().map(String::as_str).collect();
        assert_eq!(
            path_names,
            [
                "/api/file/read-file",
                "/api/file/write-file",
                "/api/shell/exec-command",
                "/api/shell/kill-process",
                "/api/shell/view-shell",
                "/api/shell/wait-for-process",
                "/api/shell/write-to-process"
            ]
        );
    }

    #[test]
    fn module_routes_are_composable() {
        let _router = create_api_routes();
    }
}
