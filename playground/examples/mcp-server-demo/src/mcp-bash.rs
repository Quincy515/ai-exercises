use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::wrapper::Parameters, schemars, tool, tool_handler,
    tool_router, transport::stdio,
};
use serde_json::json;
use tokio::process::Command;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct BashRequest {
    /// 需要在系统 shell 中执行的命令。/ Command to execute in the system shell.
    command: String,
}

#[derive(Clone)]
struct BashServer;

#[tool_router]
impl BashServer {
    #[tool(description = "传递 command 命令，在当前系统 shell 中执行命令。")]
    async fn bash(&self, Parameters(BashRequest { command }): Parameters<BashRequest>) -> String {
        match build_shell_command(&command).output().await {
            Ok(output) => json!({
                "returncode": output.status.code().unwrap_or(-1),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr),
            })
            .to_string(),
            Err(err) => json!({
                "returncode": -1,
                "stdout": "",
                "stderr": format!("命令执行失败: {err}"),
            })
            .to_string(),
        }
    }
}

#[tool_handler(
    name = "Bash工具",
    version = "0.1.0",
    instructions = "提供一个 bash 工具，输入 command 字符串并返回命令的 returncode、stdout 和 stderr。"
)]
impl ServerHandler for BashServer {}

#[tokio::main]
async fn main() -> Result<()> {
    let service = BashServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

fn build_shell_command(command: &str) -> Command {
    if cfg!(windows) {
        let mut shell = Command::new("cmd");
        shell.arg("/C").arg(command);
        shell
    } else {
        let mut shell = Command::new("sh");
        shell.arg("-c").arg(command);
        shell
    }
}
