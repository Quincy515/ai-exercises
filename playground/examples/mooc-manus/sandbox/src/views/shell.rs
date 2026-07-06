use serde::Deserialize;
use utoipa::ToSchema;

/// 执行命令请求结构体 / request body for executing command.
#[derive(Debug, ToSchema, Deserialize)]
pub struct ShellExecuteRequest {
    /// 目标 Shell 会话的唯一标识符
    pub session_id: Option<String>,
    /// 执行命令的工作目录，必须使用绝对路径
    pub exec_dir: Option<String>,
    /// 要执行的 Shell 命令
    pub command: String,
}

/// 查看 Shell 执行内容请求结构体 / request body for viewing shell output.
#[derive(Debug, ToSchema, Deserialize)]
pub struct ViewShellRequest {
    /// 目标 Shell 会话的唯一标识符
    pub session_id: String,
    /// 是否返回控制台记录列表
    pub console: Option<bool>,
}

/// 等待 Shell 命令执行请求结构体 / request body for waiting shell process.
#[derive(Debug, ToSchema, Deserialize)]
pub struct WaitForProcessRequest {
    /// 目标 Shell 会话的唯一标识符
    pub session_id: String,
    /// 等待时间，单位为秒
    pub seconds: Option<i64>,
}

/// 写入数据到子进程请求结构体 / request body for writing to shell process.
#[derive(Debug, ToSchema, Deserialize)]
pub struct ShellWriteRequest {
    /// 目标 Shell 会话的唯一标识符
    pub session_id: String,
    /// 需要写入的内容文本
    pub input_text: String,
    /// 是否按下回车键
    #[serde(default = "default_press_enter")]
    pub press_enter: bool,
}

/// 关闭进程请求结构体 / request body for killing shell process.
#[derive(Debug, ToSchema, Deserialize)]
pub struct ShellKillRequest {
    /// 目标 Shell 会话的唯一标识符
    pub session_id: String,
}

fn default_press_enter() -> bool {
    true
}
