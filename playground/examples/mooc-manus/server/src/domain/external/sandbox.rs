use anyhow::Result;
use async_trait::async_trait;

use crate::domain::{external::Browser, models::ToolResult};

/// 沙箱服务扩展协议，包含文件工具协议、shell 工具协议以及沙箱本身的扩展
/// Sandbox extension protocol, including file tools, shell tools, and sandbox lifecycle operations.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// 根据传递的会话 id+目录+命令执行对应的命令
    /// Execute a command in the given session and working directory.
    async fn exec_command(
        &self,
        session_id: &str,
        exec_dir: &str,
        command: &str,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的会话 id+是否返回控制台记录获取 shell 结果
    /// Read shell output for a session, optionally including console history.
    async fn view_shell(
        &self,
        session_id: &str,
        console: Option<bool>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的会话 id+秒数等待程序执行
    /// Wait for the session process, optionally bounded by seconds.
    async fn wait_for_process(
        &self,
        session_id: &str,
        seconds: Option<usize>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递会话 id+文本内容+是否回车键写入内容到进程中
    /// Write text to the running process, optionally pressing enter afterward.
    async fn write_to_process(
        &self,
        session_id: &str,
        input_text: &str,
        press_enter: Option<bool>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的会话 id 杀死对应的进程
    /// Kill the process associated with the session id.
    async fn kill_process(&self, session_id: &str) -> Result<ToolResult<String>>;

    /// 根据传递的文件路径+写入内容+追加模式+前后内容新行+超级权限写入对应的文件
    /// Write content to a sandbox file with append/newline/sudo options.
    async fn file_write(
        &self,
        file_path: &str,
        content: &str,
        append: Option<bool>,
        leading_newline: Option<bool>,
        trailing_newline: Option<bool>,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的文件路径+起止行号+超级权限+最大长度读取对应的文件内容
    /// Read a sandbox file with optional line range, sudo access, and maximum output length.
    async fn file_read(
        &self,
        file_path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的文件路径判断文件是否存在
    /// Check whether a sandbox file exists.
    async fn file_exists(&self, file_path: &str) -> Result<ToolResult<bool>>;

    /// 根据传递的文件路径删除指定文件
    /// Delete the specified sandbox file.
    async fn file_delete(&self, file_path: &str) -> Result<ToolResult<String>>;

    /// 根据传递的文件夹路径列出该路径下的所有文件
    /// List files under the specified sandbox directory.
    async fn file_list(&self, dir_path: &str) -> Result<ToolResult<Vec<String>>>;

    /// 根据传递文件路径+新旧内容+超级权限完成文件内容替换
    /// Replace matching content in a sandbox file.
    async fn file_replace(
        &self,
        file_path: &str,
        old_str: &str,
        new_str: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的文件路径+正则+超级权限完成文件内容检索
    /// Search a sandbox file with a regex.
    async fn file_search(
        &self,
        file_path: &str,
        regex: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<Vec<String>>>;

    /// 根据传递的文件夹路径+匹配规则查找文件
    /// Find files under a sandbox directory by glob pattern.
    async fn file_find(
        &self,
        dir_path: &str,
        glob_pattern: &str,
    ) -> Result<ToolResult<Vec<String>>>;

    /// 根据文件源数据+路径+文件名将文件上传到沙箱中
    /// Upload binary data into the sandbox.
    async fn file_upload(
        &self,
        file_data: Vec<u8>,
        file_path: &str,
        file_name: Option<&str>,
    ) -> Result<ToolResult<String>>;

    /// 根据传递的文件路径下载沙箱中的文件
    /// Download a sandbox file as binary data.
    async fn file_download(&self, file_path: &str) -> Result<Vec<u8>>;

    /// 确保当前沙箱存在，如果不存在会创建
    /// Ensure the sandbox exists, creating it when necessary.
    async fn ensure_sandbox(&self) -> Result<bool>;

    /// 销毁当前沙箱实例
    /// Destroy the current sandbox instance.
    async fn destroy(&self) -> Result<bool>;

    /// 获取沙箱中的浏览器实例
    /// Get the browser instance running inside the sandbox.
    async fn get_browser(&self) -> Result<Box<dyn Browser>>;

    /// 只读属性，返回沙箱的 id
    /// Read-only sandbox id.
    fn id(&self) -> &str;

    /// 只读属性，返回沙箱的 cdp 链接(操控浏览器的)
    /// Read-only CDP URL for controlling the sandbox browser.
    fn cdp_url(&self) -> &str;

    /// 只读属性，获取沙箱的 vnc 链接(远程桌面链接)
    /// Read-only VNC URL for the sandbox remote desktop.
    fn vnc_url(&self) -> &str;
}

/// 沙箱工厂协议，负责创建和恢复沙箱实例。
/// Sandbox factory protocol for creating and restoring sandbox instances.
#[async_trait]
pub trait SandboxFactory: Send + Sync {
    /// 类方法，用于快速创建一个沙箱
    /// Create a sandbox instance.
    async fn create(&self) -> Result<Box<dyn Sandbox>>;

    /// 类方法，根据传递的 id 获取沙箱实例
    /// Get a sandbox instance by id.
    async fn get(&self, id: &str) -> Result<Box<dyn Sandbox>>;
}
