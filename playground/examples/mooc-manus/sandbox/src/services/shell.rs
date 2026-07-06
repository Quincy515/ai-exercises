use std::{collections::HashMap, env, path::Path, process::Stdio, sync::Arc};

use axum::http::StatusCode;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStderr, ChildStdout, Command},
    sync::Mutex,
    task::JoinHandle,
    time::{Duration, Instant, sleep, timeout},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    exceptions::{AppException, ErrorData},
    models::{
        ConsoleRecord, Shell, ShellExecuteResult, ShellKillResult, ShellReadResult,
        ShellWaitResult, ShellWriteResult,
    },
};

/// Shell 命令服务
pub struct ShellService {
    active_shells: Arc<Mutex<HashMap<String, Shell>>>,
}

impl ShellService {
    pub fn new() -> Self {
        Self {
            active_shells: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 创建会话 id，使用 uuid4 生成唯一值
    pub fn create_session_id(&self) -> Result<String, AppException> {
        let session_id = Uuid::new_v4().to_string();
        info!("创建一个新的 Shell 会话 ID: {}", session_id);
        Ok(session_id)
    }

    /// 从指定会话中获取控制台记录
    pub async fn get_console_records(
        &self,
        session_id: &str,
    ) -> Result<Vec<ConsoleRecord>, AppException> {
        // 1.判断下传递的会话是否存在
        debug!("正在获取 Shell 会话的控制台记录: {}", session_id);
        let shells = self.active_shells.lock().await;
        let shell = shells
            .get(session_id)
            .ok_or_else(|| self.session_not_found(session_id))?;

        // 2.获取原始的控制台记录列表
        // 3.执行循环处理所有记录输出
        Ok(Self::clean_console_records(shell))
    }

    /// 传递会话 id+时间，等待子进程结束
    pub async fn wait_process(
        &self,
        session_id: String,
        seconds: Option<i64>,
    ) -> Result<ShellWaitResult, AppException> {
        // 1.判断下传递的会话是否存在
        let seconds = seconds.filter(|seconds| *seconds > 0).unwrap_or(60) as u64;
        debug!(
            "正在 Shell 会话中等待进程: {}, 超时: {}s",
            session_id, seconds
        );

        // 2.获取会话和子进程
        // 3.判断是否设置 seconds
        match self.wait_for_process(&session_id, seconds).await? {
            Some(returncode) => {
                // 4.记录日志并返回等待结果
                info!("进程已完成, 返回代码为: {}", returncode);
                Ok(ShellWaitResult { returncode })
            }
            None => {
                // 记录日志并抛出 BadRequest 异常
                warn!("Shell 会话进程等待超时: {}s", seconds);
                Err(AppException::bad_request(format!(
                    "Shell会话进程等待超时: {}s",
                    seconds
                )))
            }
        }
    }

    /// 根据传递的会话 id+是否输出控制台记录获取 Shell 命令结果
    pub async fn read_shell_output(
        &self,
        session_id: String,
        console: bool,
    ) -> Result<ShellReadResult, AppException> {
        // 1.判断下传递的会话是否存在
        debug!("查看 Shell 会话内容: {}", session_id);
        let shells = self.active_shells.lock().await;
        let shell = shells
            .get(&session_id)
            .ok_or_else(|| self.session_not_found(&session_id))?;

        // 2.获取会话
        // 3.获取原生输出并移除额外字符
        let clean_output = Self::remove_ansi_escape_codes(&shell.output);

        // 4.判断是否获取控制台记录
        let console_records = if console {
            Self::clean_console_records(shell)
        } else {
            Vec::new()
        };

        Ok(ShellReadResult {
            session_id,
            output: clean_output,
            console_records,
        })
    }

    /// 传递会话 id+执行目录+命令在沙箱中执行后返回
    pub async fn exec_command(
        &self,
        session_id: String,
        mut exec_dir: String,
        command: String,
    ) -> Result<ShellExecuteResult, AppException> {
        // 1.记录日志并判断执行目录是否存在
        info!("正在会话 {} 中执行命令: {}", session_id, command);
        if exec_dir.is_empty() {
            exec_dir = default_work_dir();
        }
        if !Path::new(&exec_dir).exists() {
            let error_msg = format!("当前目录不存在: {}", exec_dir);
            error!("{}", error_msg);
            return Err(AppException::bad_request(error_msg));
        }

        let setup_result = async {
            // 2.格式化生成 ps1 格式
            let ps1 = self.format_ps1(&exec_dir);

            // 3.判断当前 Shell 会话是否存在
            let old_shell = self.active_shells.lock().await.remove(&session_id);

            let (shell, stdout, stderr) = if let Some(mut shell) = old_shell {
                // 6.该会话已存在则读取数据
                debug!("使用现有的 Shell 会话: {}", session_id);

                // 7.判断旧进程是否还在运行，如果是则先停止旧进程再执行新命令
                if shell
                    .process
                    .try_wait()
                    .map_err(|err| self.command_error(&session_id, &command, err))?
                    .is_none()
                {
                    debug!("正在终止会话中的上一个进程: {}", session_id);
                    self.stop_process(&mut shell.process, &session_id, 1).await;
                }
                wait_output_readers(std::mem::take(&mut shell.output_readers)).await;

                // 10.关闭之后创建一个新的进程
                let (process, stdout, stderr) = self
                    .create_process(&session_id, &exec_dir, &command)
                    .await?;

                // 11.更新会话信息
                shell.process = process;
                shell.exec_dir = exec_dir.clone();
                shell.output.clear();
                shell
                    .console_records
                    .push(ConsoleRecord::new(ps1.clone(), command.clone()));

                (shell, stdout, stderr)
            } else {
                // 4.创建一个新的进程
                debug!("创建一个新的 Shell 会话: {}", session_id);
                let (process, stdout, stderr) = self
                    .create_process(&session_id, &exec_dir, &command)
                    .await?;
                let shell = Shell::new(process, exec_dir.clone())
                    .with_record(ConsoleRecord::new(ps1.clone(), command.clone()));

                (shell, stdout, stderr)
            };

            self.active_shells
                .lock()
                .await
                .insert(session_id.clone(), shell);

            // 5/12.创建后台任务来运行输出读取器
            let output_readers = self.start_output_reader(&session_id, stdout, stderr);
            if !output_readers.is_empty() {
                let mut shells = self.active_shells.lock().await;
                if let Some(shell) = shells.get_mut(&session_id) {
                    shell.output_readers = output_readers;
                }
            }

            Ok::<(), AppException>(())
        }
        .await;

        if let Err(err) = setup_result {
            // 19.执行过程中出现异常并记录日志后返回自定义异常
            error!(error = %err, session_id = %session_id, command = %command, "命令执行失败");
            return Err(err);
        }

        // 13.尝试等待子进程执行(最多等待 5s)
        debug!("正在等待会话中的进程完成: {}", session_id);
        match self.wait_process(session_id.clone(), Some(5)).await {
            Ok(wait_result) => {
                // 14.判断返回代码是否非空(已结束)则同步返回执行结果
                // 15.记录日志并查看结果
                debug!("Shell 会话进程已结束, 代码: {}", wait_result.returncode);
                let view_result = self.read_shell_output(session_id.clone(), false).await?;

                Ok(ShellExecuteResult {
                    session_id,
                    command,
                    status: "completed".to_string(),
                    returncode: Some(wait_result.returncode),
                    output: Some(view_result.output),
                })
            }
            Err(err) if err.status_code == StatusCode::BAD_REQUEST => {
                // 16.等待超时，记录日志不做额外处理让命令在后台继续运行
                warn!("进程在会话超时后仍在运行: {}", session_id);

                // 18.返回正在等待 Shell 执行结果
                Ok(ShellExecuteResult {
                    session_id,
                    command,
                    status: "running".to_string(),
                    returncode: None,
                    output: None,
                })
            }
            Err(err) => {
                // 17.其他异常忽略并让程序继续进行
                warn!("等待进程时出现异常: {}", err);

                // 18.返回正在等待 Shell 执行结果
                Ok(ShellExecuteResult {
                    session_id,
                    command,
                    status: "running".to_string(),
                    returncode: None,
                    output: None,
                })
            }
        }
    }

    /// 根据传递的数据向指定子进程写入数据
    pub async fn write_shell_input(
        &self,
        session_id: String,
        input_text: String,
        press_enter: bool,
    ) -> Result<ShellWriteResult, AppException> {
        // 1.判断下传递的会话是否存在
        debug!(
            "写入 Shell 会话中的子进程: {}, 是否按下回车键: {}",
            session_id, press_enter
        );
        let mut shells = self.active_shells.lock().await;
        let shell = shells
            .get_mut(&session_id)
            .ok_or_else(|| self.session_not_found(&session_id))?;

        // 2.获取会话和子进程
        // 3.检查子进程是否结束
        if shell
            .process
            .try_wait()
            .map_err(|err| self.command_error(&session_id, "", err))?
            .is_some()
        {
            error!("子进程已结束, 无法写入输入: {}", session_id);
            return Err(AppException::bad_request("子进程已结束, 无法写入输入"));
        }

        // 4.根据不同系统选择换行符
        let line_ending = line_ending();

        // 5.准备要发送的内容
        let mut text_to_send = input_text.clone();
        if press_enter {
            text_to_send.push_str(line_ending);
        }

        // 6.将字符串编码为字节流(发送给进程使用)
        let input_data = text_to_send.as_bytes();

        // 7.记录日志/输出
        let log_text = if press_enter {
            format!("{}\n", input_text)
        } else {
            input_text
        };
        shell.append_output(&log_text);

        // 8.向子进程写入数据
        let stdin = shell
            .process
            .stdin
            .as_mut()
            .ok_or_else(|| AppException::bad_request("子进程未打开标准输入, 无法写入输入"))?;
        stdin
            .write_all(input_data)
            .await
            .map_err(|err| AppException::internal(format!("向子进程写入数据出错: {}", err)))?;
        stdin
            .flush()
            .await
            .map_err(|err| AppException::internal(format!("向子进程写入数据出错: {}", err)))?;

        // 9.记录日志并返回写入结果
        info!("成功向子进程写入数据");
        Ok(ShellWriteResult {
            status: "success".to_string(),
        })
    }

    /// 根据传递的 Shell 会话 id 关闭对应进程
    pub async fn kill_process(&self, session_id: String) -> Result<ShellKillResult, AppException> {
        // 1.判断下传递的会话是否存在
        debug!("正在终止会话中的进程: {}", session_id);
        let mut shells = self.active_shells.lock().await;
        let shell = shells
            .get_mut(&session_id)
            .ok_or_else(|| self.session_not_found(&session_id))?;

        // 2.获取会话和子进程
        // 3.检查子进程是否还在运行
        if let Some(status) = shell
            .process
            .try_wait()
            .map_err(|err| self.command_error(&session_id, "", err))?
        {
            // 8.进程已结束无需重复关闭
            let returncode = exit_code(status);
            info!("进程已终止, 返回代码为: {}", returncode);
            return Ok(ShellKillResult {
                status: "already_terminated".to_string(),
                returncode,
            });
        }

        // 4.记录日志并尝试先优雅的关闭
        info!("尝试优雅终止进程: {}", session_id);
        if let Err(err) = shell.process.start_kill() {
            error!("关闭进程失败: {}", err);
            return Err(AppException::internal(format!("关闭进程失败: {}", err)));
        }

        // 5.等待 3 秒时间
        let wait_result = timeout(Duration::from_secs(3), shell.process.wait()).await;
        let returncode = match wait_result {
            Ok(Ok(status)) => exit_code(status),
            Ok(Err(err)) => {
                error!("关闭进程失败: {}", err);
                return Err(AppException::internal(format!("关闭进程失败: {}", err)));
            }
            Err(_) => {
                // 6.优雅关闭失败，则强制关闭
                warn!("尝试强制关闭进程: {}", session_id);
                shell
                    .process
                    .kill()
                    .await
                    .map_err(|err| AppException::internal(format!("关闭进程失败: {}", err)))?;
                shell
                    .process
                    .try_wait()
                    .map_err(|err| AppException::internal(format!("关闭进程失败: {}", err)))?
                    .map(exit_code)
                    .unwrap_or(-1)
            }
        };

        // 7.记录日志并返回关闭结果
        info!("进程已终止, 返回代码为: {}", returncode);
        Ok(ShellKillResult {
            status: "terminated".to_string(),
            returncode,
        })
    }

    /// 获取显示路径，将用户主目录替换成 ~
    fn get_display_path(path: &str) -> String {
        // 1.使用程序获取跨平台下用户的主目录
        let home_dir = default_work_dir();
        debug!("主目录: {}, 路径: {}", home_dir, path);

        // 2.判断传递进来的路径是否是主路径，如果是则替换成 ~
        path.strip_prefix(&home_dir)
            .map(|suffix| format!("~{}", suffix))
            .unwrap_or_else(|| path.to_string())
    }

    /// 格式化命令结构提示，增强交互体验，例如: root@myserver:/var/log $
    fn format_ps1(&self, exec_dir: &str) -> String {
        let username = env_first(&["USER", "USERNAME"], "unknown");
        let hostname = env_first(&["HOSTNAME", "COMPUTERNAME"], "localhost");
        let display_dir = Self::get_display_path(exec_dir);

        format!("{}@{}:{} $", username, hostname, display_dir)
    }

    /// 根据传递的执行目录+命令创建一个 tokio 管理的子进程
    async fn create_process(
        &self,
        session_id: &str,
        exec_dir: &str,
        command: &str,
    ) -> Result<(Child, Option<ChildStdout>, Option<ChildStderr>), AppException> {
        // 1.根据不同的系统选择不同的解释器
        let (shell_exec, shell_kind) = select_shell();
        debug!(
            "在目录 {} 下使用解释器 {} 执行命令 {}",
            exec_dir, shell_exec, command
        );

        let mut command_builder = Command::new(shell_exec);
        match shell_kind {
            ShellKind::Unix => {
                command_builder.arg("-c").arg(command);
            }
            ShellKind::PowerShell => {
                command_builder
                    .arg("-NoProfile")
                    .arg("-Command")
                    .arg(command);
            }
            ShellKind::Cmd => {
                command_builder.arg("/C").arg(command);
            }
        }

        // 3.创建一个系统级的子进程用来执行 shell 命令
        // Python 版本把 stderr 重定向到 stdout；Rust 版本保留两条管道，再把两条管道读到同一个会话输出中。
        command_builder
            .current_dir(exec_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped())
            .spawn()
            .map(|mut child| {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                (child, stdout, stderr)
            })
            .map_err(|err| self.command_error(session_id, command, err))
    }

    fn start_output_reader(
        &self,
        session_id: &str,
        stdout: Option<ChildStdout>,
        stderr: Option<ChildStderr>,
    ) -> Vec<JoinHandle<()>> {
        let mut output_readers = Vec::new();
        if let Some(stdout) = stdout {
            // 标准输出和标准错误都使用同一套读取逻辑，最终都会追加到 Shell::append_output。
            output_readers.push(self.spawn_output_reader(session_id.to_string(), stdout));
        }
        if let Some(stderr) = stderr {
            output_readers.push(self.spawn_output_reader(session_id.to_string(), stderr));
        }
        output_readers
    }

    async fn stop_process(&self, process: &mut Child, session_id: &str, seconds: u64) {
        // 8.结束旧进程并优雅等待
        if let Err(err) = process.start_kill() {
            warn!("终止 Shell 会话中的进程 {} 失败: {}", session_id, err);
            return;
        }

        match timeout(Duration::from_secs(seconds), process.wait()).await {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => {
                // 9.结束旧进程出现错误并记录日志调用 kill 强制关闭进程
                warn!("强制终止 Shell 会话中的进程 {} 失败: {}", session_id, err);
                let _ = process.kill().await;
            }
            Err(_) => {
                // 9.结束旧进程出现错误并记录日志调用 kill 强制关闭进程
                warn!("强制终止 Shell 会话中的进程 {} 超时", session_id);
                let _ = process.kill().await;
            }
        }
    }

    fn spawn_output_reader<R>(&self, session_id: String, mut reader: R) -> JoinHandle<()>
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        let active_shells = self.active_shells.clone();

        tokio::spawn(async move {
            let mut buffer = [0_u8; 4096];
            // 对应 Python 的 codecs 增量 decoder：跨 read 边界的 UTF-8 字符会留到下一块再解码。
            // 当前服务保持 UTF-8 输出契约；Windows GB18030 码页输出需要独立编码转换层。
            let mut decoder = Utf8OutputDecoder::default();

            loop {
                match reader.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(size) => {
                        let output = decoder.decode(&buffer[..size]);
                        if !output.is_empty() {
                            append_shell_output(&active_shells, &session_id, &output).await;
                        }
                    }
                    Err(err) => {
                        warn!("读取进程输出时错误: {}", err);
                        break;
                    }
                }
            }

            let output = decoder.finish();
            if !output.is_empty() {
                append_shell_output(&active_shells, &session_id, &output).await;
            }

            debug!("会话 {} 的输出读取器已完成", session_id);
        })
    }

    async fn wait_for_process(
        &self,
        session_id: &str,
        seconds: u64,
    ) -> Result<Option<i32>, AppException> {
        let deadline = Instant::now() + Duration::from_secs(seconds);

        loop {
            let (status, output_readers) = {
                let mut shells = self.active_shells.lock().await;
                let shell = shells
                    .get_mut(session_id)
                    .ok_or_else(|| self.session_not_found(session_id))?;

                let status = shell
                    .process
                    .try_wait()
                    .map_err(|err| self.command_error(session_id, "", err))?;
                let output_readers = if status.is_some() {
                    std::mem::take(&mut shell.output_readers)
                } else {
                    Vec::new()
                };
                (status, output_readers)
            };

            if let Some(status) = status {
                wait_output_readers(output_readers).await;
                return Ok(Some(exit_code(status)));
            }

            if Instant::now() >= deadline {
                return Ok(None);
            }

            sleep(Duration::from_millis(50)).await;
        }
    }

    fn clean_console_records(shell: &Shell) -> Vec<ConsoleRecord> {
        shell
            .console_records
            .iter()
            .map(|record| {
                ConsoleRecord::with_output(
                    record.ps1.clone(),
                    record.command.clone(),
                    Self::remove_ansi_escape_codes(&record.output),
                )
            })
            .collect()
    }

    /// 从文本中删除 ANSI 转义字符
    fn remove_ansi_escape_codes(text: &str) -> String {
        let mut cleaned = String::with_capacity(text.len());
        let chars = text.chars().collect::<Vec<_>>();
        let mut index = 0;

        while index < chars.len() {
            let ch = chars[index];
            if ch != '\u{1b}' {
                cleaned.push(ch);
                index += 1;
                continue;
            }

            if let Some(sequence_len) = ansi_escape_sequence_len(&chars[index + 1..]) {
                index += sequence_len + 1;
            } else {
                cleaned.push(ch);
                index += 1;
            }
        }

        cleaned
    }

    fn session_not_found(&self, session_id: &str) -> AppException {
        error!("Shell 会话不存在: {}", session_id);
        AppException::not_found(format!("Shell会话不存在: {}", session_id))
    }

    fn command_error(&self, session_id: &str, command: &str, err: std::io::Error) -> AppException {
        // 19.执行过程中出现异常并记录日志后返回自定义异常
        let msg = format!("命令执行失败: {}", err);
        error!(error = %err, session_id = %session_id, command = %command, "命令执行失败");

        AppException::new(
            msg,
            StatusCode::INTERNAL_SERVER_ERROR,
            Some(ErrorData::from([
                ("session_id".to_string(), session_id.to_string()),
                ("command".to_string(), command.to_string()),
            ])),
        )
    }
}

impl Default for ShellService {
    fn default() -> Self {
        Self::new()
    }
}

fn exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

async fn append_shell_output(
    active_shells: &Arc<Mutex<HashMap<String, Shell>>>,
    session_id: &str,
    output: &str,
) {
    let mut shells = active_shells.lock().await;

    if let Some(shell) = shells.get_mut(session_id) {
        shell.append_output(output);
    }
}

async fn wait_output_readers(output_readers: Vec<JoinHandle<()>>) {
    for output_reader in output_readers {
        match timeout(Duration::from_secs(1), output_reader).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => warn!("输出读取任务结束异常: {}", err),
            Err(_) => warn!("等待输出读取任务完成超时"),
        }
    }
}

fn ansi_escape_sequence_len(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if matches!(first, '@'..='Z' | '\\'..='_') {
        return Some(1);
    }

    if first != '[' {
        return None;
    }

    let mut index = 1;
    while chars.get(index).is_some_and(|ch| matches!(ch, '0'..='?')) {
        index += 1;
    }
    while chars.get(index).is_some_and(|ch| matches!(ch, ' '..='/')) {
        index += 1;
    }

    chars
        .get(index)
        .filter(|ch| matches!(ch, '@'..='~'))
        .map(|_| index + 1)
}

#[derive(Default)]
struct Utf8OutputDecoder {
    pending: Vec<u8>,
}

impl Utf8OutputDecoder {
    fn decode(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();

        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid = std::str::from_utf8(&self.pending[..valid_up_to])
                            .expect("valid_up_to always points to a valid UTF-8 prefix");
                        output.push_str(valid);
                        self.pending.drain(..valid_up_to);
                        continue;
                    }

                    if let Some(error_len) = err.error_len() {
                        output.push('\u{fffd}');
                        self.pending.drain(..error_len);
                        continue;
                    }

                    break;
                }
            }
        }

        output
    }

    fn finish(&mut self) -> String {
        let output = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        output
    }
}

#[derive(Debug, Clone, Copy)]
enum ShellKind {
    Unix,
    PowerShell,
    Cmd,
}

fn select_shell() -> (&'static str, ShellKind) {
    if cfg!(windows) {
        if executable_exists("powershell") {
            ("powershell", ShellKind::PowerShell)
        } else {
            ("cmd", ShellKind::Cmd)
        }
    } else if Path::new("/bin/bash").exists() {
        ("/bin/bash", ShellKind::Unix)
    } else if Path::new("/bin/zsh").exists() {
        ("/bin/zsh", ShellKind::Unix)
    } else {
        ("sh", ShellKind::Unix)
    }
}

fn executable_exists(program: &str) -> bool {
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|dir| executable_candidate_exists(&dir, program)))
        .unwrap_or(false)
}

fn executable_candidate_exists(dir: &Path, program: &str) -> bool {
    let candidate = dir.join(program);
    if candidate.is_file() {
        return true;
    }

    if cfg!(windows) && Path::new(program).extension().is_none() {
        return ["exe", "cmd", "bat", "com"]
            .iter()
            .any(|extension| candidate.with_extension(extension).is_file());
    }

    false
}

fn default_work_dir() -> String {
    env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_else(|_| {
            env::current_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| ".".to_string())
        })
}

fn env_first(keys: &[&str], default: &str) -> String {
    keys.iter()
        .find_map(|key| env::var(key).ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| default.to_string())
}

fn line_ending() -> &'static str {
    if cfg!(windows) { "\r\n" } else { "\n" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_shell_matches_current_platform() {
        let (shell_exec, shell_kind) = select_shell();

        assert!(!shell_exec.is_empty());
        if cfg!(windows) {
            assert!(matches!(shell_kind, ShellKind::PowerShell | ShellKind::Cmd));
        } else {
            assert!(matches!(shell_kind, ShellKind::Unix));
        }
    }

    #[test]
    fn default_work_dir_returns_non_empty_path() {
        assert!(!default_work_dir().is_empty());
    }

    #[tokio::test]
    async fn get_console_records_returns_cleaned_copies() {
        let service = ShellService::new();
        let session_id = service.create_session_id().unwrap();
        let command = "printf '\\033[31mconsole\\033[0m'".to_string();

        service
            .exec_command(session_id.clone(), "/tmp".to_string(), command.clone())
            .await
            .expect("exec command should succeed");
        let records = service
            .get_console_records(&session_id)
            .await
            .expect("console records should exist");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].command, command);
        assert_eq!(records[0].output, "console");
    }

    #[test]
    fn utf8_output_decoder_keeps_split_multibyte_chars() {
        let mut decoder = Utf8OutputDecoder::default();

        assert_eq!(decoder.decode(&[0xe4, 0xbd]), "");
        assert_eq!(decoder.decode(&[0xa0]), "你");
        assert_eq!(decoder.decode("好".as_bytes()), "好");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn utf8_output_decoder_replaces_invalid_bytes() {
        let mut decoder = Utf8OutputDecoder::default();

        assert_eq!(decoder.decode(&[0xff, b'a']), "\u{fffd}a");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn remove_ansi_escape_codes_matches_course_regex() {
        assert_eq!(
            ShellService::remove_ansi_escape_codes("a\x1b[31mred\x1b[0m"),
            "ared"
        );
        assert_eq!(
            ShellService::remove_ansi_escape_codes("a\x1b]0;title\x07b"),
            "a0;title\x07b"
        );
        assert_eq!(ShellService::remove_ansi_escape_codes("a\x1bc"), "a\x1bc");
        assert_eq!(
            ShellService::remove_ansi_escape_codes("a\x1b[31"),
            "a\x1b[31"
        );
    }
}
