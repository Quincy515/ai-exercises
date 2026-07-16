use std::{
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::{
    fs::{File, OpenOptions, TryLockError},
    io,
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, Command, Stdio},
    sync::Arc,
};

#[cfg(unix)]
use reqwest::{Client, header::CONTENT_TYPE};
use serde::de::DeserializeOwned;
use tokio::sync::oneshot;
use tracing::error;

use crate::{
    core::Settings,
    exceptions::AppException,
    models::{ProcessInfo, SupervisorActionResult, SupervisorProcessAction},
};

const SUPERVISOR_SOCKET_PATH: &str = "/tmp/supervisor.sock";
const SUPERVISOR_RPC_URL: &str = "http://localhost/RPC2";
const GET_ALL_PROCESS_INFO_METHOD: &str = "supervisor.getAllProcessInfo";
const STOP_ALL_PROCESSES_METHOD: &str = "supervisor.stopAllProcesses";
const START_ALL_PROCESSES_METHOD: &str = "supervisor.startAllProcesses";
const SHUTDOWN_METHOD: &str = "supervisor.shutdown";
const SUPERVISOR_ACTION_SUCCESS_STATUS: i32 = 80;
pub const SUPERVISOR_RESTART_HELPER_ARG: &str = "--supervisor-restart-helper";
#[cfg(unix)]
const RESTART_LOCK_PATH: &str = "/tmp/sandbox-supervisor-restart.lock";
const RESTART_HELPER_DELAY: Duration = Duration::from_millis(500);
const MAX_RPC_RESPONSE_BYTES: usize = 1024 * 1024;
#[cfg(all(unix, test))]
const QUERY_RPC_TIMEOUT: Duration = Duration::from_millis(100);
#[cfg(all(unix, not(test)))]
const QUERY_RPC_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(all(unix, test))]
const MANAGEMENT_RPC_TIMEOUT: Duration = Duration::from_secs(1);
#[cfg(all(unix, not(test)))]
const MANAGEMENT_RPC_TIMEOUT: Duration = Duration::from_secs(120);

#[cfg(unix)]
// 保护同一个 app 内“启动辅助进程 -> 移交文件锁”的短暂窗口；
// RESTART_LOCK_PATH 负责跨越新旧 app 的完整重启周期。
static RESTART_SCHEDULED: AtomicBool = AtomicBool::new(false);

/*
1. Supervisor 启动后，通过 Unix Socket 文件实现 RPC 通信。
2. 连接 /tmp/supervisor.sock，建立 XML-RPC 连接。
3. 使用 reqwest 将 HTTP 请求完整转换到 Unix Socket 传输。
4. 连接成功后根据业务动作调用对应的 Supervisor RPC 方法。
*/

/// Supervisor 服务
#[derive(Debug)]
pub struct SupervisorService {
    /// Supervisor 用于 XML-RPC 通信的 Unix Socket 文件路径。
    rpc_url: String,
    /// 当前自动销毁定时器。
    ///
    /// `None` 表示没有激活超时销毁；`Some` 同时保存计划销毁时间和取消句柄，
    /// 避免“状态显示已激活，实际任务却不存在”这一类状态不一致问题。
    shutdown_timer: Mutex<Option<ShutdownTimer>>,
    /// 是否开启自动保活。
    ///
    /// 本节先保存开关；后续请求中间件可以在每次调用接口时读取它，
    /// 决定是否重新设置定时器并延长沙箱存活时间。服务会通过 `Arc` 被多个
    /// Axum 请求共享，因此使用 `AtomicBool` 完成无锁的并发读写。
    expand_enabled: AtomicBool,
}

/// 一次正在执行的沙箱自动销毁计划。
#[derive(Debug)]
struct ShutdownTimer {
    /// 计划关闭 supervisord 的单调时钟时间点。
    ///
    /// `Instant` 不受系统时间校准影响，比日历时间更适合计算进程内倒计时。
    shutdown_time: Instant,
    /// 定时任务取消句柄，对应课程实现中 `shutdown_task.cancel()` 的职责。
    cancel: oneshot::Sender<()>,
}

impl Default for SupervisorService {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisorService {
    /// 构造函数，保存 Supervisor Unix Socket 路径并启动超时销毁定时器。
    ///
    /// 这里对应客户端初始化阶段；实际连接会在 `call_rpc()` 调用
    /// `send().await` 时异步建立。
    pub fn new() -> Self {
        // 1. 连接 Supervisor 配置：`with_shutdown_timeout()` 会保存 Unix Socket 路径。
        // 2. Supervisor 超时配置：读取 SERVER_TIMEOUT_MINUTES，默认值为 60 分钟。
        let settings = Settings::load();
        let timeout =
            Duration::from_secs(settings.server_timeout_minutes as u64).saturating_mul(60);

        // 3. 检测自动销毁配置：当前 Rust Settings 始终提供分钟数，因此启动定时器。
        // 4. 设置销毁时间并创建定时任务，到期后关闭 supervisord 主进程。
        Self::with_shutdown_timeout(SUPERVISOR_SOCKET_PATH, timeout)
    }

    /// 创建只负责 RPC 调用的服务实例，不启动自动销毁定时器。
    ///
    /// 超时任务到期后会通过这个构造函数复用 `shutdown()`；这样不会在执行关闭时
    /// 再创建一个新的关闭定时器。重启辅助进程和测试 RPC 客户端也使用这条路径。
    pub(crate) fn with_rpc_url(rpc_url: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            shutdown_timer: Mutex::new(None),
            expand_enabled: AtomicBool::new(true),
        }
    }

    /// 创建带有超时销毁能力的 Supervisor 服务。
    ///
    /// 先完成 RPC 客户端状态初始化，再调用 `setup_timer()` 保存销毁时间并启动任务。
    fn with_shutdown_timeout(rpc_url: impl Into<String>, shutdown_timeout: Duration) -> Self {
        let service = Self::with_rpc_url(rpc_url);
        service.setup_timer(shutdown_timeout);
        service
    }

    /// 获取定时器状态锁；即使其他线程持锁时发生 panic，也尽量取回内部状态完成清理。
    fn lock_shutdown_timer(&self) -> MutexGuard<'_, Option<ShutdownTimer>> {
        self.shutdown_timer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// 返回是否配置了沙箱超时销毁，即当前是否存在有效定时器。
    pub fn timeout_active(&self) -> bool {
        self.shutdown_time().is_some()
    }

    /// 返回当前计划关闭 Supervisor 的时间点；未激活时返回 `None`。
    pub fn shutdown_time(&self) -> Option<Instant> {
        self.lock_shutdown_timer()
            .as_ref()
            .map(|timer| timer.shutdown_time)
    }

    /// 只读获取是否开启自动保活。
    pub fn expand_enabled(&self) -> bool {
        self.expand_enabled.load(Ordering::Acquire)
    }

    /// 开启自动保活。
    ///
    /// 这里只修改开关；后续中间件收到下一次 API 请求时才会重新设置销毁时间。
    pub fn enable_expand(&self) {
        self.expand_enabled.store(true, Ordering::Release);
    }

    /// 关闭自动保活。
    ///
    /// 已创建的定时器继续倒计时，后续 API 请求不会再自动延长它。
    pub fn disable_expand(&self) {
        self.expand_enabled.store(false, Ordering::Release);
    }

    /// 传递超时时长并创建定时器，在时间结束后关闭 supervisord 主进程。
    fn setup_timer(&self, timeout: Duration) {
        // 1. 计算销毁时间。极端配置发生时间溢出时不创建无效任务。
        let Some(shutdown_time) = Instant::now().checked_add(timeout) else {
            error!("Supervisor 超时配置超出系统可表示范围");
            return;
        };

        // 2. 创建一次性取消通道：Sender 由服务保存，Receiver 交给异步定时任务。
        let (cancel, cancel_rx) = oneshot::channel();
        let previous_timer = self.lock_shutdown_timer().replace(ShutdownTimer {
            shutdown_time,
            cancel,
        });

        // 3. 检测当前是否存在销毁任务；存在时先取消，再使用新的销毁时间。
        // 取消失败表示旧任务已经结束，不影响新任务继续执行。
        if let Some(previous_timer) = previous_timer {
            let _ = previous_timer.cancel.send(());
        }

        // 4. 获取 Tokio 事件循环并添加任务；运行时不可用时由后台线程兜底。
        // 两种启动方式都失败时清空定时器，确保 `timeout_active()` 返回真实状态。
        if let Err(err) = spawn_shutdown_timer(self.rpc_url.clone(), timeout, cancel_rx) {
            let timer = self.lock_shutdown_timer().take();
            if let Some(timer) = timer {
                let _ = timer.cancel.send(());
            }
            error!(error = %err, "创建 Supervisor 超时销毁定时器失败");
        }
    }

    /// 获取当前 Supervisor 管理的所有进程信息。
    pub async fn get_all_processes(&self) -> Result<Vec<ProcessInfo>, AppException> {
        let response = self
            .call_rpc(GET_ALL_PROCESS_INFO_METHOD)
            .await
            .map_err(Self::process_error)?;
        parse_rpc_response(&response)
            .map_err(|err| Self::process_error(format!("RPC方法调用失败: {err}")))
    }

    /// 停止 Supervisor 管理的全部子进程，保留 supervisord 主进程。
    pub async fn stop_all_processes(&self) -> Result<SupervisorActionResult, AppException> {
        // `wait=false` 让 supervisord 发出停止信号后立即返回每个进程的操作结果。
        // Axum 的 SIGTERM 优雅退出会等待当前 HTTP 请求写完，再结束 app 进程。
        let response = self
            .call_rpc_with_params(
                STOP_ALL_PROCESSES_METHOD,
                std::iter::once(serde_xmlrpc::Value::Bool(false)),
            )
            .await
            .map_err(Self::stop_error)?;
        let result =
            parse_validated_action_response("停止", &response).map_err(Self::stop_error)?;

        Ok(SupervisorActionResult::stopped(result))
    }

    /// 关闭 supervisord 主进程；它作为容器 PID 1 时会同时结束容器。
    pub async fn shutdown(&self) -> Result<SupervisorActionResult, AppException> {
        let response = self
            .call_rpc(SHUTDOWN_METHOD)
            .await
            .map_err(Self::shutdown_error)?;
        let result = parse_shutdown_response(&response)
            .map_err(|err| Self::shutdown_error(format!("RPC方法调用失败: {err}")))?;

        Ok(SupervisorActionResult::shutdown(result))
    }

    /// 启动一个独立辅助进程，让 HTTP 请求可以在 app 停止前返回。
    #[cfg(unix)]
    pub fn schedule_restart(&self) -> Result<SupervisorActionResult, AppException> {
        if !try_acquire_restart_schedule() {
            return Err(AppException::bad_request("Supervisor重启任务正在执行"));
        }

        let result = (|| {
            let restart_lock = match try_acquire_restart_lock(Path::new(RESTART_LOCK_PATH))
                .map_err(|err| Self::restart_error(format!("打开重启任务锁失败: {err}")))?
            {
                Some(restart_lock) => restart_lock,
                None => return Err(AppException::bad_request("Supervisor重启任务正在执行")),
            };
            let executable = std::env::current_exe()
                .map_err(|err| Self::restart_error(format!("获取当前程序路径失败: {err}")))?;
            self.schedule_restart_with(&executable, restart_lock)
        })();
        if result.is_err() {
            release_restart_schedule();
        }
        result
    }

    #[cfg(not(unix))]
    pub fn schedule_restart(&self) -> Result<SupervisorActionResult, AppException> {
        Err(Self::restart_error(
            "Supervisor 重启辅助进程仅支持 Unix 系统".to_string(),
        ))
    }

    #[cfg(unix)]
    fn schedule_restart_with(
        &self,
        executable: &Path,
        restart_lock: File,
    ) -> Result<SupervisorActionResult, AppException> {
        let child = build_restart_helper_command(executable, &self.rpc_url)
            .spawn()
            .map_err(|err| Self::restart_error(format!("启动重启辅助进程失败: {err}")))?;
        watch_restart_helper(child)
            .map_err(|err| Self::restart_error(format!("启动辅助进程回收线程失败: {err}")))?;

        // 辅助进程已经启动并等待同一把文件锁。此处释放后，锁的所有权
        // 从 HTTP app 平滑移交给辅助进程，整个重启周期始终只有一个执行者。
        drop(restart_lock);

        Ok(SupervisorActionResult::restart_scheduled())
    }

    /// 在 Supervisor 管理范围之外执行真正的重启流程。
    #[cfg(unix)]
    pub async fn run_restart_helper(rpc_url: impl Into<String>) -> Result<(), AppException> {
        // 文件锁跨越 app 的新旧进程。新 app 即使已经恢复 HTTP 服务，
        // 也会在本次 stop/start 全部完成前拒绝第二个重启任务。
        let _restart_lock = acquire_restart_lock(Path::new(RESTART_LOCK_PATH))
            .map_err(|err| Self::restart_error(format!("获取重启任务锁失败: {err}")))?;
        tokio::time::sleep(RESTART_HELPER_DELAY).await;
        let result = Self::with_rpc_url(rpc_url).restart_all_processes().await?;
        tracing::info!(
            stopped = result.stop_result.as_ref().map_or(0, Vec::len),
            started = result.start_result.as_ref().map_or(0, Vec::len),
            "Supervisor 所有进程重启成功"
        );
        Ok(())
    }

    #[cfg(not(unix))]
    pub async fn run_restart_helper(_rpc_url: impl Into<String>) -> Result<(), AppException> {
        Err(Self::restart_error(
            "Supervisor 重启辅助进程仅支持 Unix 系统".to_string(),
        ))
    }

    /// 按照先停止、再启动的顺序重启 Supervisor 管理的全部子进程。
    async fn restart_all_processes(&self) -> Result<SupervisorActionResult, AppException> {
        let stop_result = self
            .call_validated_action(STOP_ALL_PROCESSES_METHOD, "停止")
            .await;

        // stop 请求一旦写入 Socket，就可能已经关闭部分进程。无论停止阶段
        // 的响应、解析或状态校验是否成功，都执行启动阶段作为补偿恢复。
        let start_result = self
            .call_validated_action(START_ALL_PROCESSES_METHOD, "启动")
            .await;

        match (stop_result, start_result) {
            (Ok(stop_result), Ok(start_result)) => {
                Ok(SupervisorActionResult::restarted(stop_result, start_result))
            }
            (Err(stop_error), Ok(_)) => Err(Self::restart_error(format!(
                "停止阶段失败，已执行补偿启动: {stop_error}"
            ))),
            (Ok(_), Err(start_error)) => {
                Err(Self::restart_error(format!("启动阶段失败: {start_error}")))
            }
            (Err(stop_error), Err(start_error)) => Err(Self::restart_error(format!(
                "停止阶段失败: {stop_error}; 补偿启动失败: {start_error}"
            ))),
        }
    }

    async fn call_validated_action(
        &self,
        method: &str,
        action: &str,
    ) -> Result<Vec<SupervisorProcessAction>, String> {
        let response = self.call_rpc(method).await?;
        parse_validated_action_response(action, &response)
    }

    /// 根据方法名调用无参数 XML-RPC 服务。
    async fn call_rpc(&self, method: &str) -> Result<String, String> {
        let request = build_rpc_request(method)?;
        self.send_rpc_request(method, request).await
    }

    /// 根据方法名和参数调用 XML-RPC 服务。
    async fn call_rpc_with_params(
        &self,
        method: &str,
        params: impl Iterator<Item = serde_xmlrpc::Value>,
    ) -> Result<String, String> {
        let request = build_rpc_request_with_params(method, params)?;
        self.send_rpc_request(method, request).await
    }

    /// 通过 Supervisor Unix Socket 发送已经序列化的 XML-RPC 请求。
    #[cfg(unix)]
    async fn send_rpc_request(&self, method: &str, request: String) -> Result<String, String> {
        let timeout = rpc_timeout(method);

        // 这段配置承担 `_connect_rpc()` 的职责：
        // 1. `unix_socket()` 把 HTTP 传输固定到 `/tmp/supervisor.sock`；
        // 2. `http1_only()` 使用 Supervisor XML-RPC 所需的 HTTP/1.1；
        // 3. `build()` 创建异步 HTTP 客户端，此时只完成连接配置。
        let client = Client::builder()
            .unix_socket(self.rpc_url.as_str())
            .http1_only()
            .timeout(timeout)
            .build()
            .map_err(|err| map_reqwest_error(err, timeout))?;

        // URL 用于生成 HTTP 请求路径 `/RPC2` 和 Host 头。
        // reqwest 会跳过 DNS，并在 `send().await` 时真正连接上面的 Unix Socket。
        let mut response = client
            .post(SUPERVISOR_RPC_URL)
            .header(CONTENT_TYPE, "text/xml")
            .body(request)
            .send()
            .await
            .map_err(|err| map_reqwest_error(err, timeout))?;

        if !response.status().is_success() {
            return Err(format!(
                "RPC方法调用失败: Supervisor HTTP 请求失败: {}",
                response.status()
            ));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RPC_RESPONSE_BYTES as u64)
        {
            return Err(response_too_large());
        }

        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|err| map_reqwest_error(err, timeout))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_RPC_RESPONSE_BYTES {
                return Err(response_too_large());
            }
            body.extend_from_slice(&chunk);
        }

        String::from_utf8(body).map_err(|err| format!("RPC方法调用失败: 响应不是 UTF-8: {err}"))
    }

    #[cfg(not(unix))]
    async fn send_rpc_request(&self, _method: &str, _request: String) -> Result<String, String> {
        Err("RPC方法调用失败: Supervisor Unix Socket 仅支持 Unix 系统".to_string())
    }

    fn process_error(err: String) -> AppException {
        let msg = format!("获取进程信息失败: {err}");
        error!(error = %msg, "获取 Supervisor 进程信息失败");
        AppException::internal(msg)
    }

    fn stop_error(err: String) -> AppException {
        let msg = format!("停止supervisor所有进程服务失败: {err}");
        error!(error = %msg, "停止 Supervisor 所有进程服务失败");
        AppException::internal(msg)
    }

    fn shutdown_error(err: String) -> AppException {
        let msg = format!("关闭supervisord服务失败: {err}");
        error!(error = %msg, "关闭 supervisord 服务失败");
        AppException::internal(msg)
    }

    fn restart_error(err: String) -> AppException {
        let msg = format!("重启Supervisor进程服务失败: {err}");
        error!(error = %msg, "重启 Supervisor 进程服务失败");
        AppException::internal(msg)
    }
}

impl Drop for SupervisorService {
    fn drop(&mut self) {
        // 服务状态释放时取消尚未到期的任务，避免后台任务继续关闭已结束的沙箱实例。
        let timer = self
            .shutdown_timer
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(timer) = timer {
            let _ = timer.cancel.send(());
        }
    }
}

/// 创建 Supervisor 关闭定时任务。
///
/// 1. 优先获取当前 Tokio 运行时并添加异步任务，不阻塞 Axum 请求线程。
/// 2. 同步上下文没有 Tokio 运行时时，创建单线程运行时作为兜底。
/// 3. 兜底运行时放入命名后台线程，承担课程实现中 `threading.Timer` 的职责。
fn spawn_shutdown_timer(
    rpc_url: String,
    timeout: Duration,
    cancel: oneshot::Receiver<()>,
) -> Result<(), String> {
    // 对应“获取事件循环并添加任务”。Axum 正常启动时会进入这条路径。
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(shutdown_after_timeout(rpc_url, timeout, cancel));
        return Ok(());
    }

    // 对应“事件循环失败则创建新线程执行定时器”。先构建运行时，保证错误可以返回。
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("创建后台定时器运行时失败: {err}"))?;

    // 后台线程只负责驱动一个定时任务；线程启动失败时由调用者清理已保存的状态。
    std::thread::Builder::new()
        .name("supervisor-shutdown-timer".to_string())
        .spawn(move || runtime.block_on(shutdown_after_timeout(rpc_url, timeout, cancel)))
        .map(|_| ())
        .map_err(|err| format!("创建后台定时器线程失败: {err}"))
}

/// 异步等待超时并关闭 Supervisor。
async fn shutdown_after_timeout(rpc_url: String, timeout: Duration, cancel: oneshot::Receiver<()>) {
    // 等待“取消信号”和“倒计时结束”中的任意一个事件。
    // `biased` 让两个事件同时就绪时优先处理取消，避免服务释放瞬间误触发关闭。
    tokio::select! {
        biased;
        _ = cancel => return,
        _ = tokio::time::sleep(timeout) => {}
    }

    // 睡眠指定时间后调用已有 shutdown()，通过 XML-RPC 关闭 supervisord 主进程。
    // supervisord 作为容器 PID 1 退出后，容器运行时会结束该沙箱容器。
    tracing::info!(
        timeout_seconds = timeout.as_secs(),
        "沙箱超过空闲时间，正在关闭 Supervisor"
    );
    if let Err(err) = SupervisorService::with_rpc_url(rpc_url).shutdown().await {
        error!(error = %err, "沙箱超时后关闭 Supervisor 失败");
    }
}

#[cfg(unix)]
fn build_restart_helper_command(executable: &Path, rpc_url: &str) -> Command {
    let mut command = Command::new(executable);
    command
        .arg(SUPERVISOR_RESTART_HELPER_ARG)
        .arg(rpc_url)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    // 独立进程组可避免 app 收到进程组终止信号时连带结束辅助进程。
    command.process_group(0);
    command
}

#[cfg(unix)]
fn try_acquire_restart_schedule() -> bool {
    RESTART_SCHEDULED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(unix)]
fn release_restart_schedule() {
    RESTART_SCHEDULED.store(false, Ordering::Release);
}

#[cfg(unix)]
fn open_restart_lock(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

#[cfg(unix)]
fn try_acquire_restart_lock(path: &Path) -> io::Result<Option<File>> {
    let lock = open_restart_lock(path)?;
    match lock.try_lock() {
        Ok(()) => Ok(Some(lock)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(err)) => Err(err),
    }
}

#[cfg(unix)]
fn acquire_restart_lock(path: &Path) -> io::Result<File> {
    let lock = open_restart_lock(path)?;
    lock.lock()?;
    Ok(lock)
}

#[cfg(unix)]
/// 后台回收辅助进程；若线程创建失败，保留的 `Child` 仍可被终止并等待。
fn watch_restart_helper(child: Child) -> io::Result<()> {
    let child = Arc::new(Mutex::new(Some(child)));
    let watched_child = Arc::clone(&child);
    let spawn_result = std::thread::Builder::new()
        .name("supervisor-restart-reaper".to_string())
        .spawn(move || {
            let mut child = watched_child
                .lock()
                .expect("restart helper child mutex should not be poisoned")
                .take()
                .expect("restart helper child should be available to the reaper");
            match child.wait() {
                Ok(status) if status.success() => {}
                Ok(status) => {
                    error!(?status, "Supervisor 重启辅助进程执行失败");
                }
                Err(err) => {
                    error!(error = %err, "回收 Supervisor 重启辅助进程失败");
                }
            }
            release_restart_schedule();
        });

    if let Err(err) = spawn_result {
        // 辅助进程此时仍在等待调度者持有的文件锁，可以安全终止并回收。
        if let Some(mut child) = child
            .lock()
            .expect("restart helper child mutex should not be poisoned")
            .take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
        return Err(err);
    }

    Ok(())
}

#[cfg(unix)]
fn rpc_timeout(method: &str) -> Duration {
    if method == GET_ALL_PROCESS_INFO_METHOD {
        QUERY_RPC_TIMEOUT
    } else {
        MANAGEMENT_RPC_TIMEOUT
    }
}

fn build_rpc_request(method: &str) -> Result<String, String> {
    build_rpc_request_with_params(method, std::iter::empty::<serde_xmlrpc::Value>())
}

fn build_rpc_request_with_params(
    method: &str,
    params: impl Iterator<Item = serde_xmlrpc::Value>,
) -> Result<String, String> {
    serde_xmlrpc::request_to_string(method, params)
        .map_err(|err| format!("RPC方法调用失败: 构造请求失败: {err}"))
}

fn parse_rpc_response(response: &str) -> Result<Vec<ProcessInfo>, String> {
    let mut processes: Vec<ProcessInfo> = parse_xmlrpc_response(response)?;
    for process in &mut processes {
        decode_process_strings(process)?;
    }
    Ok(processes)
}

fn parse_action_response(response: &str) -> Result<Vec<SupervisorProcessAction>, String> {
    let mut actions: Vec<SupervisorProcessAction> = parse_xmlrpc_response(response)?;
    for action in &mut actions {
        for value in [&mut action.name, &mut action.group, &mut action.description] {
            *value = decode_xml_entities(value)?;
        }
    }
    Ok(actions)
}

fn parse_validated_action_response(
    action: &str,
    response: &str,
) -> Result<Vec<SupervisorProcessAction>, String> {
    let result =
        parse_action_response(response).map_err(|err| format!("RPC方法调用失败: {err}"))?;
    validate_action_results(action, &result).map_err(|err| format!("RPC方法调用失败: {err}"))?;
    Ok(result)
}

fn validate_action_results(
    action: &str,
    results: &[SupervisorProcessAction],
) -> Result<(), String> {
    let failures = results
        .iter()
        .filter(|result| result.status != SUPERVISOR_ACTION_SUCCESS_STATUS)
        .map(|result| {
            format!(
                "{}:{}(status={}, description={})",
                result.group, result.name, result.status, result.description
            )
        })
        .collect::<Vec<_>>();

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Supervisor {action}进程失败: {}",
            failures.join(", ")
        ))
    }
}

fn parse_shutdown_response(response: &str) -> Result<bool, String> {
    let accepted = parse_xmlrpc_response(response)?;
    if accepted {
        Ok(true)
    } else {
        Err("Supervisor未接受关闭请求".to_string())
    }
}

fn parse_xmlrpc_response<T>(response: &str) -> Result<T, String>
where
    T: DeserializeOwned,
{
    if !response.trim_end().ends_with("</methodResponse>") {
        return Err("Supervisor XML-RPC methodResponse 根节点不完整或包含多余内容".to_string());
    }

    match serde_xmlrpc::response_from_str(response) {
        Ok(result) => Ok(result),
        Err(serde_xmlrpc::Error::Fault(mut fault)) => {
            fault.fault_string = decode_xml_entities(&fault.fault_string)?;
            Err(format!("Supervisor XML-RPC 响应解析失败: {fault}"))
        }
        Err(err) => Err(format!("Supervisor XML-RPC 响应解析失败: {err}")),
    }
}

fn decode_process_strings(process: &mut ProcessInfo) -> Result<(), String> {
    for value in [
        &mut process.name,
        &mut process.group,
        &mut process.description,
        &mut process.statename,
        &mut process.spawnerr,
        &mut process.logfile,
        &mut process.stdout_logfile,
        &mut process.stderr_logfile,
    ] {
        *value = decode_xml_entities(value)?;
    }
    Ok(())
}

/// serde_xmlrpc 0.3.0-alpha.2 会保留字符串中的 XML 实体，这里统一解码一次。
fn decode_xml_entities(value: &str) -> Result<String, String> {
    let mut decoded = String::with_capacity(value.len());
    let mut remaining = value;

    while let Some(index) = remaining.find('&') {
        decoded.push_str(&remaining[..index]);
        let entity = &remaining[index + 1..];
        let end = entity
            .find(';')
            .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: {remaining}"))?;
        let entity_name = &entity[..end];
        let character = match entity_name {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            value if value.starts_with("#x") => u32::from_str_radix(&value[2..], 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: &{value};"))?,
            value if value.starts_with('#') => value[1..]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: &{value};"))?,
            value => return Err(format!("Supervisor XML-RPC 包含未知实体: &{value};")),
        };
        decoded.push(character);
        remaining = &entity[end + 1..];
    }

    decoded.push_str(remaining);
    Ok(decoded)
}

#[cfg(unix)]
fn map_reqwest_error(err: reqwest::Error, timeout: Duration) -> String {
    if err.is_timeout() {
        format!("RPC方法调用失败: 超时({}ms)", timeout.as_millis())
    } else {
        format!("RPC方法调用失败: {err}")
    }
}

#[cfg(unix)]
fn response_too_large() -> String {
    format!("RPC方法调用失败: Supervisor 响应过大(上限 {MAX_RPC_RESPONSE_BYTES} 字节)")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[cfg(unix)]
    static RESTART_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const PROCESS_ARRAY: &str = r#"
        <array><data>
          <value><struct>
            <member><name>name</name><value><string>xvfb</string></value></member>
            <member><name>group</name><value><string>services</string></value></member>
            <member><name>description</name><value><string>pid 10, uptime 0:00:03</string></value></member>
            <member><name>start</name><value><int>1750000000</int></value></member>
            <member><name>stop</name><value><int>0</int></value></member>
            <member><name>now</name><value><int>1750000003</int></value></member>
            <member><name>state</name><value><int>20</int></value></member>
            <member><name>statename</name><value><string>RUNNING</string></value></member>
            <member><name>spawnerr</name><value><string></string></value></member>
            <member><name>exitstatus</name><value><int>0</int></value></member>
            <member><name>logfile</name><value><string>/tmp/xvfb.log</string></value></member>
            <member><name>stdout_logfile</name><value><string>/dev/stdout</string></value></member>
            <member><name>stderr_logfile</name><value><string>/dev/stderr</string></value></member>
            <member><name>pid</name><value><int>10</int></value></member>
          </struct></value>
          <value><struct>
            <member><name>name</name><value><string>chrome</string></value></member>
            <member><name>group</name><value><string>services</string></value></member>
            <member><name>description</name><value><string>ready &amp; listening</string></value></member>
            <member><name>start</name><value><int>1750000001</int></value></member>
            <member><name>stop</name><value><int>0</int></value></member>
            <member><name>now</name><value><int>1750000003</int></value></member>
            <member><name>state</name><value><int>20</int></value></member>
            <member><name>statename</name><value><string>RUNNING</string></value></member>
            <member><name>spawnerr</name><value><string></string></value></member>
            <member><name>exitstatus</name><value><int>0</int></value></member>
            <member><name>logfile</name><value><string>/tmp/chrome.log</string></value></member>
            <member><name>stdout_logfile</name><value><string>/dev/stdout</string></value></member>
            <member><name>stderr_logfile</name><value><string>/dev/stderr</string></value></member>
            <member><name>pid</name><value><int>11</int></value></member>
          </struct></value>
        </data></array>
    "#;

    fn success_xml(body: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?><methodResponse><params><param><value>{body}</value></param></params></methodResponse>"
        )
    }

    fn action_result_xml_with_status(status: i32, description: &str) -> String {
        success_xml(&format!(
            r#"<array><data><value><struct>
                <member><name>name</name><value><string>app</string></value></member>
                <member><name>group</name><value><string>services</string></value></member>
                <member><name>status</name><value><int>{status}</int></value></member>
                <member><name>description</name><value><string>{description}</string></value></member>
            </struct></value></data></array>"#
        ))
    }

    fn action_result_xml(description: &str) -> String {
        action_result_xml_with_status(80, description)
    }

    fn http_response(xml: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len()
        )
    }

    #[cfg(unix)]
    async fn read_xmlrpc_request(stream: &mut tokio::net::UnixStream) -> String {
        use tokio::io::AsyncReadExt;

        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        loop {
            let read = stream.read(&mut chunk).await.expect("request should read");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
            if request
                .windows(b"</methodCall>".len())
                .any(|window| window == b"</methodCall>")
            {
                break;
            }
        }

        String::from_utf8(request).expect("XML-RPC request should be UTF-8")
    }

    #[cfg(unix)]
    fn service_without_runtime(rpc_url: String, timeout: Duration) -> SupervisorService {
        std::thread::spawn(move || {
            assert!(tokio::runtime::Handle::try_current().is_err());
            SupervisorService::with_shutdown_timeout(rpc_url, timeout)
        })
        .join()
        .expect("service constructor thread should finish")
    }

    #[cfg(unix)]
    async fn call_mock_supervisor(response: String) -> Result<Vec<ProcessInfo>, AppException> {
        use tokio::{io::AsyncWriteExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let request = read_xmlrpc_request(&mut stream).await;
            assert!(request.starts_with("POST /RPC2 HTTP/1.1"));
            assert!(request.contains("supervisor.getAllProcessInfo"));
            stream
                .write_all(response.as_bytes())
                .await
                .expect("response should write");
        });

        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());
        let result = service.get_all_processes().await;
        server.await.expect("mock server should finish");
        std::fs::remove_file(socket_path).expect("test socket should be removed");
        result
    }

    #[cfg(unix)]
    async fn call_mock_action<F, Fut>(
        exchanges: Vec<(&'static str, Option<&'static str>, String)>,
        call: F,
    ) -> Result<SupervisorActionResult, AppException>
    where
        F: FnOnce(SupervisorService) -> Fut,
        Fut: std::future::Future<Output = Result<SupervisorActionResult, AppException>>,
    {
        use tokio::{io::AsyncWriteExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-action-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            for (expected_method, expected_parameter, response) in exchanges {
                let (mut stream, _) = listener.accept().await.expect("client should connect");
                let request = read_xmlrpc_request(&mut stream).await;
                assert!(
                    request.contains(expected_method),
                    "unexpected request: {request}"
                );
                if let Some(expected_parameter) = expected_parameter {
                    assert!(
                        request.contains(expected_parameter),
                        "request is missing {expected_parameter}: {request}"
                    );
                }
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("response should write");
            }
        });

        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());
        let result = call(service).await;
        server.await.expect("mock server should finish");
        std::fs::remove_file(socket_path).expect("test socket should be removed");
        result
    }

    #[test]
    fn default_service_uses_supervisor_socket() {
        assert_eq!(SupervisorService::default().rpc_url, "/tmp/supervisor.sock");
    }

    #[test]
    fn automatic_keepalive_can_be_enabled_and_disabled() {
        let service = SupervisorService::with_rpc_url("/tmp/test-supervisor.sock");

        assert!(service.expand_enabled());
        service.disable_expand();
        assert!(!service.expand_enabled());
        service.enable_expand();
        assert!(service.expand_enabled());
    }

    #[tokio::test]
    async fn configured_timeout_records_an_automatic_shutdown_deadline() {
        let before = std::time::Instant::now();
        let service = SupervisorService::with_shutdown_timeout(
            "/tmp/test-supervisor.sock",
            Duration::from_secs(60),
        );

        assert!(service.timeout_active());
        let shutdown_time = service
            .shutdown_time()
            .expect("configured timeout should record a deadline");
        assert!(shutdown_time >= before + Duration::from_secs(60));
    }

    #[test]
    fn overflowing_timeout_is_not_reported_as_active() {
        let service =
            SupervisorService::with_shutdown_timeout("/tmp/test-supervisor.sock", Duration::MAX);

        assert!(!service.timeout_active());
        assert!(service.shutdown_time().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn context_without_runtime_uses_background_shutdown_timer() {
        use tokio::{io::AsyncWriteExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-sync-timeout-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let request = read_xmlrpc_request(&mut stream).await;
            stream
                .write_all(http_response(&success_xml("<boolean>1</boolean>")).as_bytes())
                .await
                .expect("response should write");
            request
        });

        let _service =
            service_without_runtime(socket_path.display().to_string(), Duration::from_millis(20));

        let request = tokio::time::timeout(Duration::from_millis(500), server)
            .await
            .expect("background timer should call Supervisor")
            .expect("mock server should finish");
        assert!(
            request.contains(SHUTDOWN_METHOD),
            "unexpected request: {request}"
        );
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_service_cancels_background_shutdown_timer() {
        use tokio::net::UnixListener;

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-sync-timeout-cancel-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let service =
            service_without_runtime(socket_path.display().to_string(), Duration::from_secs(1));

        drop(service);
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err(),
            "a dropped service must cancel its background shutdown"
        );
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn configured_timeout_shuts_down_supervisor_after_the_deadline() {
        use tokio::{io::AsyncWriteExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-timeout-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let request = read_xmlrpc_request(&mut stream).await;
            assert!(
                request.contains(SHUTDOWN_METHOD),
                "unexpected request: {request}"
            );
            stream
                .write_all(http_response(&success_xml("<boolean>1</boolean>")).as_bytes())
                .await
                .expect("response should write");
        });

        let _service = SupervisorService::with_shutdown_timeout(
            socket_path.display().to_string(),
            Duration::from_millis(10),
        );

        tokio::time::timeout(Duration::from_millis(500), server)
            .await
            .expect("shutdown timer should call Supervisor")
            .expect("mock server should finish");
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }

    #[test]
    fn rpc_requests_call_course_methods_without_parameters() {
        for method in [
            GET_ALL_PROCESS_INFO_METHOD,
            START_ALL_PROCESSES_METHOD,
            SHUTDOWN_METHOD,
        ] {
            let request = build_rpc_request(method).expect("request should serialize");
            let (actual_method, parameters) =
                serde_xmlrpc::request_from_str(&request).expect("request should parse");

            assert_eq!(actual_method, method);
            assert!(parameters.is_empty());
        }
    }

    #[test]
    fn stop_all_processes_request_returns_before_the_app_exits() {
        let request = build_rpc_request_with_params(
            STOP_ALL_PROCESSES_METHOD,
            std::iter::once(serde_xmlrpc::Value::Bool(false)),
        )
        .expect("stop request should serialize");
        let (method, parameters) =
            serde_xmlrpc::request_from_str(&request).expect("stop request should parse");

        assert_eq!(method, STOP_ALL_PROCESSES_METHOD);
        assert_eq!(parameters, vec![serde_xmlrpc::Value::Bool(false)]);
    }

    #[test]
    fn action_response_preserves_supervisor_result_data() {
        let result = parse_action_response(&action_result_xml("Stopped &amp; ready"))
            .expect("action response should parse");

        assert_eq!(result[0].name, "app");
        assert_eq!(result[0].status, 80);
        assert_eq!(result[0].description, "Stopped & ready");
    }

    #[test]
    fn action_response_rejects_a_failed_process_result() {
        let result = parse_action_response(&action_result_xml_with_status(50, "SPAWN_ERROR"))
            .expect("action response should parse");

        let error = validate_action_results("启动", &result)
            .expect_err("non-success process status should fail the action");

        assert!(error.contains("services:app"), "unexpected error: {error}");
        assert!(error.contains("50"), "unexpected error: {error}");
        assert!(error.contains("SPAWN_ERROR"), "unexpected error: {error}");
    }

    #[test]
    fn management_rpc_timeout_has_room_for_supervisor_stop_wait() {
        assert!(rpc_timeout(STOP_ALL_PROCESSES_METHOD) > rpc_timeout(GET_ALL_PROCESS_INFO_METHOD));
        assert_eq!(
            rpc_timeout(START_ALL_PROCESSES_METHOD),
            rpc_timeout(STOP_ALL_PROCESSES_METHOD)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_all_processes_calls_course_rpc_method() {
        let result = call_mock_action(
            vec![(
                "supervisor.stopAllProcesses",
                Some("<boolean>0</boolean>"),
                http_response(&action_result_xml("Stopped")),
            )],
            |service| async move { service.stop_all_processes().await },
        )
        .await
        .expect("stop all processes should succeed");

        assert_eq!(result.status, "stopped");
        assert_eq!(
            result.result.expect("stop result should exist")[0].description,
            "Stopped"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_calls_course_rpc_method() {
        let result = call_mock_action(
            vec![(
                "supervisor.shutdown",
                None,
                http_response(&success_xml("<boolean>1</boolean>")),
            )],
            |service| async move { service.shutdown().await },
        )
        .await
        .expect("shutdown should succeed");

        assert_eq!(result.status, "shutdown");
        assert_eq!(result.shutdown_result, Some(true));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_rejects_a_false_rpc_result() {
        let error = call_mock_action(
            vec![(
                "supervisor.shutdown",
                None,
                http_response(&success_xml("<boolean>0</boolean>")),
            )],
            |service| async move { service.shutdown().await },
        )
        .await
        .expect_err("false should mean Supervisor rejected shutdown");

        assert!(
            error.msg.contains("未接受关闭请求"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restart_stops_all_processes_before_starting_them() {
        let result = call_mock_action(
            vec![
                (
                    "supervisor.stopAllProcesses",
                    None,
                    http_response(&action_result_xml("Stopped")),
                ),
                (
                    "supervisor.startAllProcesses",
                    None,
                    http_response(&action_result_xml("Started")),
                ),
            ],
            |service| async move { service.restart_all_processes().await },
        )
        .await
        .expect("restart should succeed");

        assert_eq!(result.status, "restarted");
        assert_eq!(
            result.stop_result.expect("stop result should exist")[0].description,
            "Stopped"
        );
        assert_eq!(
            result.start_result.expect("start result should exist")[0].description,
            "Started"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restart_compensates_when_stop_reports_a_failed_process() {
        let error = call_mock_action(
            vec![
                (
                    "supervisor.stopAllProcesses",
                    None,
                    http_response(&action_result_xml_with_status(50, "FAILED")),
                ),
                (
                    "supervisor.startAllProcesses",
                    None,
                    http_response(&action_result_xml("Started")),
                ),
            ],
            |service| async move { service.restart_all_processes().await },
        )
        .await
        .expect_err("failed stop result should still trigger compensation start");

        assert!(
            error.msg.contains("已执行补偿启动"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restart_compensates_when_stop_response_cannot_be_parsed() {
        let error = call_mock_action(
            vec![
                (
                    "supervisor.stopAllProcesses",
                    None,
                    http_response("<broken>"),
                ),
                (
                    "supervisor.startAllProcesses",
                    None,
                    http_response(&action_result_xml("Started")),
                ),
            ],
            |service| async move { service.restart_all_processes().await },
        )
        .await
        .expect_err("invalid stop response should still trigger compensation start");

        assert!(
            error.msg.contains("已执行补偿启动"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restart_reports_stop_and_compensation_failures_together() {
        let error = call_mock_action(
            vec![
                (
                    "supervisor.stopAllProcesses",
                    None,
                    http_response(&action_result_xml_with_status(50, "STOP_FAILED")),
                ),
                (
                    "supervisor.startAllProcesses",
                    None,
                    http_response(&action_result_xml_with_status(60, "START_FAILED")),
                ),
            ],
            |service| async move { service.restart_all_processes().await },
        )
        .await
        .expect_err("both restart stage failures should be reported");

        assert!(
            error.msg.contains("STOP_FAILED"),
            "unexpected error: {}",
            error.msg
        );
        assert!(
            error.msg.contains("补偿启动失败") && error.msg.contains("START_FAILED"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[test]
    fn restart_helper_command_uses_an_independent_binary_process() {
        use std::{ffi::OsStr, path::Path};

        let command = build_restart_helper_command(
            Path::new("/usr/local/bin/sandbox"),
            "/tmp/test-supervisor.sock",
        );
        let args = command.get_args().collect::<Vec<_>>();

        assert_eq!(command.get_program(), OsStr::new("/usr/local/bin/sandbox"));
        assert_eq!(
            args,
            [
                OsStr::new("--supervisor-restart-helper"),
                OsStr::new("/tmp/test-supervisor.sock")
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn restart_schedule_is_single_flight_and_can_be_released() {
        let _guard = RESTART_TEST_LOCK
            .lock()
            .expect("restart test lock should work");
        release_restart_schedule();
        assert!(try_acquire_restart_schedule());
        assert!(!try_acquire_restart_schedule());
        release_restart_schedule();
        assert!(try_acquire_restart_schedule());
        release_restart_schedule();
    }

    #[cfg(unix)]
    #[test]
    fn restart_lock_blocks_a_new_app_until_the_helper_finishes() {
        let lock_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-restart-test-{}.lock",
            uuid::Uuid::new_v4()
        ));

        let first_lock = try_acquire_restart_lock(&lock_path)
            .expect("first restart lock should open")
            .expect("first restart lock should be acquired");
        assert!(
            try_acquire_restart_lock(&lock_path)
                .expect("second restart lock should open")
                .is_none(),
            "another app must not schedule a restart while the helper owns the lock"
        );

        drop(first_lock);
        let next_lock = try_acquire_restart_lock(&lock_path)
            .expect("released restart lock should open")
            .expect("released restart lock should be acquired");
        drop(next_lock);
        std::fs::remove_file(lock_path).expect("restart test lock should be removed");
    }

    #[cfg(unix)]
    #[test]
    fn failed_restart_helper_is_reaped_and_releases_single_flight() {
        let _guard = RESTART_TEST_LOCK
            .lock()
            .expect("restart test lock should work");
        release_restart_schedule();
        assert!(try_acquire_restart_schedule());

        let child = std::process::Command::new("/usr/bin/false")
            .spawn()
            .expect("false helper should start");
        watch_restart_helper(child).expect("restart helper watcher should start");

        for _ in 0..100 {
            if try_acquire_restart_schedule() {
                release_restart_schedule();
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        panic!("failed helper should be reaped and release the restart schedule");
    }

    #[test]
    fn rpc_response_maps_all_fields_and_preserves_order() {
        let processes = parse_rpc_response(&success_xml(PROCESS_ARRAY))
            .expect("valid Supervisor response should parse");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[0].start, 1_750_000_000);
        assert_eq!(processes[0].pid, 10);
        assert_eq!(processes[1].name, "chrome");
        assert_eq!(processes[1].description, "ready & listening");
        assert_eq!(processes[1].stdout_logfile, "/dev/stdout");
    }

    #[test]
    fn rpc_response_rejects_process_with_missing_required_field() {
        let body = PROCESS_ARRAY.replacen(
            "<member><name>group</name><value><string>services</string></value></member>",
            "",
            1,
        );

        let error = parse_rpc_response(&success_xml(&body))
            .expect_err("missing group should reject the complete response");

        assert!(error.contains("group"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_truncated_array() {
        let xml = success_xml("<array><data><value><struct><member><name>name</name>");

        let error = parse_rpc_response(&xml)
            .expect_err("truncated array should reject the complete response");

        assert!(error.contains("解析失败"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_partial_process_list() {
        let (truncated, _) = PROCESS_ARRAY
            .rsplit_once("</struct></value>")
            .expect("fixture should contain a final process");

        let error = parse_rpc_response(&success_xml(truncated))
            .expect_err("truncated second process should reject the complete response");

        assert!(error.contains("解析失败"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_content_after_method_response_root() {
        let xml = format!("{}<broken", success_xml(PROCESS_ARRAY));

        let error = parse_rpc_response(&xml)
            .expect_err("content after the XML root should fail document validation");

        assert!(error.contains("根节点"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_fault_is_reported_with_fault_message() {
        let body = r#"<fault><value><struct>
            <member><name>faultCode</name><value><int>1</int></value></member>
            <member><name>faultString</name><value><string>UNKNOWN &amp; METHOD</string></value></member>
        </struct></value></fault>"#;
        let xml = format!("<?xml version=\"1.0\"?><methodResponse>{body}</methodResponse>");

        let error = parse_rpc_response(&xml).expect_err("XML-RPC fault should be an error");

        assert!(
            error.contains("UNKNOWN & METHOD"),
            "unexpected error: {error}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_uses_unix_socket_and_returns_typed_models() {
        let xml = success_xml(PROCESS_ARRAY);

        let processes = call_mock_supervisor(http_response(&xml))
            .await
            .expect("mock Supervisor call should succeed");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[1].name, "chrome");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_accepts_chunked_http_response() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{xml}\r\n0\r\n\r\n",
            xml.len()
        );

        let processes = call_mock_supervisor(response)
            .await
            .expect("reqwest should decode chunked HTTP framing");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[1].name, "chrome");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_mismatched_content_length() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len() + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("truncated HTTP response should fail");

        assert!(
            error.msg.contains("RPC方法调用失败"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_conflicting_content_length_headers() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len(),
            xml.len() + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("conflicting Content-Length headers should fail");

        assert!(
            error.msg.contains("RPC方法调用失败"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_oversized_response() {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n",
            MAX_RPC_RESPONSE_BYTES + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("oversized response should fail before reading its body");

        assert!(
            error.msg.contains("响应过大"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_oversized_chunked_response_while_streaming() {
        let first_chunk = "x".repeat(MAX_RPC_RESPONSE_BYTES);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{first_chunk}\r\n1\r\nx\r\n0\r\n\r\n",
            first_chunk.len()
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("streamed response should enforce the same size limit");

        assert!(
            error.msg.contains("响应过大"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_wraps_connection_failure_as_app_error() {
        let socket_path = PathBuf::from(format!(
            "/tmp/missing-supervisor-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());

        let error = service
            .get_all_processes()
            .await
            .expect_err("missing socket should fail");

        assert_eq!(
            error.status_code,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(error.msg.contains("获取进程信息失败"));
        assert!(error.msg.contains("RPC方法调用失败"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_times_out_when_supervisor_stalls() {
        use tokio::{io::AsyncReadExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/stalled-supervisor-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let mut request = vec![0_u8; 4096];
            let _ = stream
                .read(&mut request)
                .await
                .expect("request should read");
            std::future::pending::<()>().await;
        });
        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            service.get_all_processes(),
        )
        .await
        .expect("service should enforce an RPC timeout");
        let error = result.expect_err("stalled Supervisor should time out");

        assert!(
            error.msg.contains("超时"),
            "unexpected error: {}",
            error.msg
        );
        server.abort();
        let _ = server.await;
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }
}
