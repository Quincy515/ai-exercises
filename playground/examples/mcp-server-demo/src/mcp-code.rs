use std::{
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use rmcp::{
    ServerHandler,
    handler::server::wrapper::Parameters,
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use tokio::{fs, process::Command, time::timeout};

const DEFAULT_BIND_ADDRESS: &str = "127.0.0.1:9888";
const DEFAULT_TIMEOUT_SECONDS: u64 = 30;
const DEFAULT_UV_CMD: &str = "uv";
const DEFAULT_NODE_CMD: &str = "node";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct RunCodeRequest {
    /// 支持的语言：`python` 或 `node`。/ Supported language: `python` or `node`.
    language: String,
    /// 要执行的代码文本。/ Source code text to execute.
    code: String,
    /// 最长运行秒数，默认 30 秒。/ Maximum runtime in seconds, default is 30 seconds.
    timeout: Option<u64>,
}

#[derive(Clone)]
struct CodeServer;

#[tool_router]
impl CodeServer {
    #[tool(description = "根据 language 运行代码并返回 stdout 或错误信息。")]
    async fn run_code(
        &self,
        Parameters(RunCodeRequest {
            language,
            code,
            timeout,
        }): Parameters<RunCodeRequest>,
    ) -> String {
        // 1. 检查传递的编程语言是否符合规则
        let language = language.trim().to_lowercase();
        let Some(suffix) = script_suffix(&language) else {
            return format!("不支持的语言: {language}");
        };

        let base_dir = code_base_dir();
        let timeout_seconds = timeout.unwrap_or(DEFAULT_TIMEOUT_SECONDS);

        // 2. 生成临时脚本路径
        let (script_name, script_path) = temp_script_path(&base_dir, suffix);

        let result = run_temp_script(
            &language,
            &code,
            timeout_seconds,
            &base_dir,
            &script_name,
            &script_path,
        )
        .await;

        let _ = fs::remove_file(&script_path).await;
        result
    }
}

#[tool_handler(
    name = "代码解释器",
    version = "0.1.0",
    instructions = "提供一个 run_code 工具，支持运行 python 或 node 代码并返回执行结果。"
)]
impl ServerHandler for CodeServer {}

#[tokio::main]
async fn main() -> Result<()> {
    let service: StreamableHttpService<CodeServer, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(CodeServer),
            Default::default(),
            StreamableHttpServerConfig::default(),
        );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(DEFAULT_BIND_ADDRESS).await?;

    eprintln!("Code MCP server listening on http://{DEFAULT_BIND_ADDRESS}/mcp");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_temp_script(
    language: &str,
    code: &str,
    timeout_seconds: u64,
    base_dir: &Path,
    script_name: &str,
    script_path: &Path,
) -> String {
    if let Err(err) = fs::create_dir_all(base_dir).await {
        return format!("执行异常: {err}");
    }

    if let Err(err) = fs::write(script_path, code).await {
        return format!("执行异常: {err}");
    }

    let mut command = build_run_command(language, base_dir, script_name, script_path);
    command.current_dir(base_dir).kill_on_drop(true);

    match timeout(Duration::from_secs(timeout_seconds), command.output()).await {
        Ok(Ok(output)) => format_output(output),
        Ok(Err(err)) => format!("命令未找到或路径错误: {err}"),
        Err(_) => format!("执行超时(>{timeout_seconds}s)"),
    }
}

// 根据不同的语言类型执行不同的操作。/ Execute different commands by language.
fn build_run_command(
    language: &str,
    base_dir: &Path,
    script_name: &str,
    script_path: &Path,
) -> Command {
    match language {
        // 使用 uv 来运行对应的 Python 文件。/ Run the Python script with uv.
        "python" => {
            let mut command = Command::new(
                std::env::var("MCP_CODE_UV_CMD").unwrap_or_else(|_| DEFAULT_UV_CMD.to_string()),
            );
            command
                .arg("--directory")
                .arg(base_dir)
                .arg("run")
                .arg(script_name);
            command
        }
        // 使用 node 命令运行脚本。/ Run the script with node.
        "node" => {
            let mut command = Command::new(
                std::env::var("MCP_CODE_NODE_CMD").unwrap_or_else(|_| DEFAULT_NODE_CMD.to_string()),
            );
            command.arg(script_path);
            command
        }
        _ => unreachable!("language is validated before command construction"),
    }
}

// 获取输出与错误结果。/ Extract stdout and stderr from command output.
fn format_output(output: std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if output.status.success() {
        stdout
    } else {
        format!(
            "命令返回非零状态 {}, stderr: \n{}",
            output.status.code().unwrap_or(-1),
            if stderr.is_empty() { stdout } else { stderr }
        )
    }
}

// 获取代码文件名
fn script_suffix(language: &str) -> Option<&'static str> {
    match language {
        "python" => Some("py"),
        "node" => Some("js"),
        _ => None,
    }
}

fn code_base_dir() -> PathBuf {
    match std::env::var("MCP_CODE_BASE_DIR") {
        Ok(path) if !path.trim().is_empty() => {
            let path = PathBuf::from(path);
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir()
                    .map(|cwd| cwd.join(&path))
                    .unwrap_or(path)
            }
        }
        _ => std::env::temp_dir().join("mcp-code-demo"),
    }
}

fn temp_script_path(base_dir: &Path, suffix: &str) -> (String, PathBuf) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let script_name = format!("temp_{}_{}.{}", std::process::id(), timestamp, suffix);
    let script_path = base_dir.join(&script_name);
    (script_name, script_path)
}
