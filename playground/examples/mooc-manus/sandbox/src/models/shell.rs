use serde::{Deserialize, Serialize};
use tokio::{process::Child, task::JoinHandle};
use utoipa::ToSchema;

/// Shell 命令行控制台记录
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ConsoleRecord {
    /// ps1
    pub ps1: String,
    /// 执行命令
    pub command: String,
    /// 输出内容
    pub output: String,
}

impl ConsoleRecord {
    pub fn new(ps1: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            ps1: ps1.into(),
            command: command.into(),
            output: String::new(),
        }
    }

    pub fn with_output(
        ps1: impl Into<String>,
        command: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            ps1: ps1.into(),
            command: command.into(),
            output: output.into(),
        }
    }
}

/// Shell 会话模型
pub struct Shell {
    /// 会话中的子进程
    pub process: Child,
    /// 会话执行目录
    pub exec_dir: String,
    /// 会话输出
    pub output: String,
    /// Shell 会话中控制台记录列表
    pub console_records: Vec<ConsoleRecord>,
    /// 输出读取任务
    pub output_readers: Vec<JoinHandle<()>>,
}

impl Shell {
    pub fn new(process: Child, exec_dir: impl Into<String>) -> Self {
        Self {
            process,
            exec_dir: exec_dir.into(),
            output: String::new(),
            console_records: Vec::new(),
            output_readers: Vec::new(),
        }
    }

    pub fn with_record(mut self, record: ConsoleRecord) -> Self {
        self.console_records.push(record);
        self
    }

    pub fn append_output(&mut self, output: &str) {
        self.output.push_str(output);

        if let Some(record) = self.console_records.last_mut() {
            record.output.push_str(output);
        }
    }
}

/// 会话等待结果模型 / shell wait result model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ShellWaitResult {
    /// 子进程返回代码
    pub returncode: i32,
}

/// Shell 命令结果模型 / shell command view result model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ShellReadResult {
    /// Shell 会话 id
    pub session_id: String,
    /// Shell 会话输出内容
    pub output: String,
    /// 控制台记录
    pub console_records: Vec<ConsoleRecord>,
}

/// Shell 命令执行结果 / shell command execution result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ShellExecuteResult {
    /// Shell 会话 id
    pub session_id: String,
    /// 执行的命令
    pub command: String,
    /// 命令执行状态
    pub status: String,
    /// 进程返回代码，只有进程结束时才有值
    pub returncode: Option<i32>,
    /// 进程执行结果，只有进程结束时才有值
    pub output: Option<String>,
}

/// Shell 命令写入结果模型 / shell command write result model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ShellWriteResult {
    /// 写入状态
    pub status: String,
}

/// Shell 命令关闭结果 / shell command kill result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ShellKillResult {
    /// 进程状态
    pub status: String,
    /// 进程返回状态
    pub returncode: i32,
}
