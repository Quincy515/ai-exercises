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
    components(schemas(exceptions::ErrorData)),
    paths(
        file::read_file,
        file::write_file,
        file::replace_in_file,
        file::search_in_file,
        file::find_files,
        file::upload_file,
        file::download_file,
        file::check_file_exists,
        file::delete_file,
        shell::exec_command,
        shell::read_shell_output,
        shell::wait_process,
        shell::write_shell_input,
        shell::kill_process,
        supervisor::get_status,
        supervisor::stop_all_processes,
        supervisor::shutdown,
        supervisor::restart,
        supervisor::activate_timeout,
        supervisor::extend_timeout,
        supervisor::cancel_timeout,
        supervisor::get_timeout_status,
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
                "/api/file/check-file-exists",
                "/api/file/delete-file",
                "/api/file/download-file",
                "/api/file/find-files",
                "/api/file/read-file",
                "/api/file/replace-in-file",
                "/api/file/search-in-file",
                "/api/file/upload-file",
                "/api/file/write-file",
                "/api/shell/exec-command",
                "/api/shell/kill-process",
                "/api/shell/read-shell-output",
                "/api/shell/wait-process",
                "/api/shell/write-shell-input",
                "/api/supervisor/activate-timeout",
                "/api/supervisor/cancel-timeout",
                "/api/supervisor/extend-timeout",
                "/api/supervisor/restart",
                "/api/supervisor/shutdown",
                "/api/supervisor/status",
                "/api/supervisor/stop-all-processes",
                "/api/supervisor/timeout-status",
            ]
        );
    }

    #[test]
    fn module_routes_are_composable() {
        let _router = create_api_routes();
    }

    #[test]
    fn upload_schema_exposes_a_binary_file_picker() {
        let json = ApiDoc::openapi()
            .to_json()
            .expect("OpenAPI document should serialize");

        assert!(
            json.contains(r#""file":{"type":"string","format":"binary""#),
            "upload file schema should use string/binary: {json}"
        );
    }

    #[test]
    fn error_response_schema_uses_registered_error_data_component() {
        let openapi = ApiDoc::openapi();
        let components = openapi
            .components
            .as_ref()
            .expect("OpenAPI document should contain components");
        let json = openapi
            .to_json()
            .expect("OpenAPI document should serialize");

        assert!(
            components.schemas.contains_key("ErrorData"),
            "error response data schema should be registered: {json}"
        );
        assert!(
            !json.contains("#/components/schemas/BTreeMap"),
            "OpenAPI document should not expose an unresolved BTreeMap reference: {json}"
        );
    }
}
