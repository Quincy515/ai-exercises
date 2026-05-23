use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ClientInfo},
    object,
    transport::TokioChildProcess,
};

const SERVER_BIN_NAME: &str = "mcp-server-stdio";
const DEFAULT_EXPRESSION: &str = "564*34+12.4/455^2";

#[tokio::main]
async fn main() -> Result<()> {
    let expression = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_EXPRESSION.to_string());
    let transport =
        TokioChildProcess::new(server_command()?).context("启动 stdio MCP Server 失败")?;
    let client = ClientInfo::default()
        .serve(transport)
        .await
        .context("初始化 stdio MCP Client 失败")?;

    let tools = client.list_all_tools().await.context("获取工具列表失败")?;
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<Vec<_>>();
    println!("工具列表: {tool_names:?}");

    let result = client
        .call_tool(
            CallToolRequestParams::new("calculator")
                .with_arguments(object!({ "expression": expression })),
        )
        .await
        .context("调用 calculator 工具失败")?;

    let result_text = result
        .content
        .iter()
        .filter_map(|content| content.raw.as_text())
        .map(|text| text.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    println!("工具结果: {result_text}");

    client.cancel().await.context("关闭 MCP Client 失败")?;

    Ok(())
}

fn server_command() -> Result<tokio::process::Command> {
    if let Some(server_path) = sibling_server_binary()? {
        return Ok(tokio::process::Command::new(server_path));
    }

    let mut command = tokio::process::Command::new("cargo");
    command
        .arg("run")
        .arg("--quiet")
        .arg("--bin")
        .arg(SERVER_BIN_NAME)
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    Ok(command)
}

fn sibling_server_binary() -> Result<Option<PathBuf>> {
    let current_exe = std::env::current_exe().context("获取当前 Client 二进制路径失败")?;
    let Some(target_dir) = current_exe.parent() else {
        bail!("当前 Client 二进制没有父目录: {}", current_exe.display());
    };

    let server_name = format!("{SERVER_BIN_NAME}{}", std::env::consts::EXE_SUFFIX);
    let server_path = target_dir.join(server_name);
    Ok(server_path.exists().then_some(server_path))
}
