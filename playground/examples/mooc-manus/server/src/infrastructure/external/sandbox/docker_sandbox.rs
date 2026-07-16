use std::{
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr},
    sync::{LazyLock, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use bollard::{
    errors::Error as BollardError,
    models::{ContainerCreateBody, ContainerInspectResponse, EndpointSettings, HostConfig},
    query_parameters::{CreateContainerOptionsBuilder, RemoveContainerOptionsBuilder},
    Docker,
};
use reqwest::{
    multipart::{Form, Part},
    Client, RequestBuilder,
};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use tokio::net::lookup_host;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        external::{Browser, Sandbox},
        models::ToolResult,
    },
    infrastructure::{external::browser::ChromiumoxideBrowser, settings::SandboxSettings},
};

const DEFAULT_SANDBOX_ID: &str = "lenexus-sandbox";
const SANDBOX_API_PORT: u16 = 3000;
const SANDBOX_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const HOSTNAME_CACHE_CAPACITY: usize = 128;
const SANDBOX_CACHE_CAPACITY: usize = 128;
const SANDBOX_STATUS_MAX_RETRIES: usize = 30;
const SANDBOX_STATUS_RETRY_INTERVAL: Duration = Duration::from_secs(2);
const SANDBOX_STATUS_TOTAL_TIMEOUT: Duration = Duration::from_secs(60);

static HOSTNAME_CACHE: LazyLock<Mutex<HostnameCache>> =
    LazyLock::new(|| Mutex::new(HostnameCache::default()));
static SANDBOX_CACHE: LazyLock<Mutex<SandboxCache>> =
    LazyLock::new(|| Mutex::new(SandboxCache::default()));

/// 基于 Docker 的沙箱服务。
#[derive(Debug, Clone)]
pub struct DockerSandbox {
    client: Client,
    ip: Ipv4Addr,
    id: String,
    container_id: Option<String>,
    base_url: String,
    vnc_url: String,
    cdp_url: String,
}

impl DockerSandbox {
    /// 构造函数，完成 Docker 沙箱扩展创建。
    pub fn new(ip: Ipv4Addr, container_name: Option<String>) -> Result<Self> {
        let client = build_http_client()?;
        let id = container_name
            .clone()
            .unwrap_or_else(|| DEFAULT_SANDBOX_ID.to_string());

        Ok(Self::from_client(ip, id, container_name, client))
    }

    fn from_client(ip: Ipv4Addr, id: String, container_id: Option<String>, client: Client) -> Self {
        Self {
            client,
            ip,
            id,
            container_id,
            base_url: format!("http://{ip}:{SANDBOX_API_PORT}"),
            vnc_url: format!("ws://{ip}:5901"),
            cdp_url: format!("http://{ip}:9222"),
        }
    }

    /// 类方法，创建沙箱容器。
    pub async fn create(settings: &SandboxSettings) -> Result<Self> {
        // 1.获取系统配置信息（由 Loco 的强类型 settings 注入）

        // 2.判断是否使用现成的沙箱
        if let Some(address) = settings.address.as_deref() {
            // 3.将沙箱主机/地址解析成 IP
            let ip = Self::resolve_hostname_to_ip(address)
                .await
                .ok_or_else(|| anyhow!("无法将沙箱主机地址 {address} 解析成 IPv4"))?;
            return Self::new(ip, None);
        }

        // 4.Bollard 提供原生异步 Docker API，直接创建容器后返回
        Self::create_container(settings).await.map_err(|err| {
            error!(error = %err, "创建 Docker 沙箱容器失败");
            err
        })
    }

    /// 根据传递的 id 获取沙箱实例。
    pub async fn get(settings: &SandboxSettings, id: &str) -> Result<Self> {
        let cache_enabled = settings.address.is_some();
        let cache_key = SandboxCacheKey::new(settings, id);
        if cache_enabled {
            if let Some(sandbox) = lock_sandbox_cache().get(&cache_key) {
                return Ok(sandbox);
            }
        }

        // 1.先获取系统配置并判断是否直连沙箱
        let sandbox = if let Some(address) = settings.address.as_deref() {
            let ip = Self::resolve_hostname_to_ip(address)
                .await
                .ok_or_else(|| anyhow!("无法将沙箱主机地址 {address} 解析成 IPv4"))?;
            Self::from_client(ip, id.to_string(), None, build_http_client()?)
        } else {
            // 2.创建 Docker 客户端并根据容器名字获取容器
            let name_prefix =
                required_setting(&settings.name_prefix, "settings.sandbox.name_prefix")?;
            validate_managed_container_id(id, name_prefix)?;
            let docker = Docker::connect_with_defaults().context("创建 Docker 客户端失败")?;
            let container = docker
                .inspect_container(id, None)
                .await
                .with_context(|| format!("获取 Docker 沙箱容器 {id} 失败"))?;

            // 3.获取容器的 IP 地址
            let ip = Self::get_container_ip(&container, settings.network.as_deref())
                .ok_or_else(|| anyhow!("Docker 沙箱容器 {id} 没有可用的 IPv4 地址"))?;
            Self::from_client(
                ip,
                id.to_string(),
                Some(id.to_string()),
                build_http_client()?,
            )
        };

        if cache_enabled {
            lock_sandbox_cache().insert(cache_key, sandbox.clone());
        }
        Ok(sandbox)
    }

    /// 销毁当前的 DockerSandbox 实例。
    pub async fn destroy(&self) -> Result<bool> {
        // reqwest::Client 使用共享所有权，最后一个句柄释放时会自动关闭连接池。
        lock_sandbox_cache().remove_by_id(&self.id);

        // 1.关闭并移除由当前服务管理的容器
        if let Some(container_id) = self.container_id.as_deref() {
            let docker = match Docker::connect_with_defaults() {
                Ok(docker) => docker,
                Err(err) => {
                    error!(sandbox_id = %self.id, error = %err, "创建 Docker 客户端失败");
                    return Ok(false);
                }
            };
            let options = RemoveContainerOptionsBuilder::default().force(true).build();
            match docker.remove_container(container_id, Some(options)).await {
                Ok(()) => {}
                Err(err) if is_container_not_found(&err) => {
                    info!(sandbox_id = %self.id, "Docker 沙箱容器已经不存在");
                }
                Err(err) => {
                    error!(sandbox_id = %self.id, error = %err, "销毁当前 Docker 沙箱失败");
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// 获取沙箱中的浏览器实例。
    pub async fn get_browser(&self) -> Result<Box<dyn Browser>> {
        Ok(Box::new(ChromiumoxideBrowser::from_cdp_url(
            self.cdp_url.clone(),
        )))
    }

    /// 确保沙箱存在，并且 Supervisor 管理的服务全部处于运行状态。
    pub async fn ensure_sandbox(&self) -> Result<bool> {
        self.ensure_sandbox_with_timeout(
            SANDBOX_STATUS_MAX_RETRIES,
            SANDBOX_STATUS_RETRY_INTERVAL,
            SANDBOX_STATUS_TOTAL_TIMEOUT,
        )
        .await
    }

    /// 获取沙箱的唯一 id，使用容器名字作为唯一 id。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub const fn ip(&self) -> Ipv4Addr {
        self.ip
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[must_use]
    pub fn vnc_url(&self) -> &str {
        &self.vnc_url
    }

    #[must_use]
    pub fn cdp_url(&self) -> &str {
        &self.cdp_url
    }

    #[must_use]
    pub const fn client(&self) -> &Client {
        &self.client
    }

    /// 将 Docker 容器主机/地址转换成 IPv4 格式数据。
    async fn resolve_hostname_to_ip(hostname: &str) -> Option<Ipv4Addr> {
        // 1.首先解析传递的 hostname 是不是 IP
        if let Ok(ip) = hostname.parse() {
            return Some(ip);
        }

        if let Some(ip) = lock_hostname_cache().get(hostname) {
            return Some(ip);
        }

        // 2.使用 Tokio 获取地址信息
        let resolved = match lookup_host((hostname, 0)).await {
            Ok(mut addresses) => addresses.find_map(|address| match address.ip() {
                IpAddr::V4(ip) => Some(ip),
                IpAddr::V6(_) => None,
            }),
            Err(err) => {
                error!(hostname, error = %err, "解析 Docker 容器主机地址失败");
                None
            }
        };

        // 3.判断地址信息是否存在，缓存并返回第一个 IPv4 地址
        if let Some(ip) = resolved {
            lock_hostname_cache().insert(hostname.to_string(), ip);
        }

        resolved
    }

    /// 根据传递的容器获取 IP 信息。
    fn get_container_ip(
        container: &ContainerInspectResponse,
        preferred_network: Option<&str>,
    ) -> Option<Ipv4Addr> {
        // 1.获取 inspect 网络设置
        let network_settings = container.network_settings.as_ref()?;
        let networks = network_settings.networks.as_ref()?;

        // 2.判断配置的网络是否存在 IP，存在时优先使用
        if let Some(ip) = preferred_network
            .and_then(|name| networks.get(name))
            .and_then(endpoint_ipv4)
        {
            return Some(ip);
        }

        // 3.循环遍历每一项网络配置，返回第一个有效 IPv4 地址
        networks.values().find_map(endpoint_ipv4)
    }

    /// 创建沙箱容器的异步任务。
    async fn create_container(settings: &SandboxSettings) -> Result<Self> {
        // 1.获取系统配置信息（settings 已由调用方传入）

        // 2.构建容器的名字
        let name_prefix = required_setting(&settings.name_prefix, "settings.sandbox.name_prefix")?;
        let container_name = build_container_name(name_prefix);
        let container_config = build_container_config(settings)?;
        let client = build_http_client()?;

        // 3.创建一个 Docker 客户端，遵循 DOCKER_HOST 和当前平台默认连接方式
        let docker = Docker::connect_with_defaults().context("创建 Docker 客户端失败")?;

        // 4.预配置容器信息在 build_container_config 中完成
        let options = CreateContainerOptionsBuilder::default()
            .name(&container_name)
            .build();

        // 5.可选网络通过 HostConfig.network_mode 传递

        // 6.调用 Docker 客户端创建并启动沙箱
        let created = docker
            .create_container(Some(options), container_config)
            .await
            .with_context(|| format!("创建 Docker 沙箱容器 {container_name} 失败"))?;
        let initialized: Result<Ipv4Addr> = async {
            docker
                .start_container(&created.id, None)
                .await
                .with_context(|| format!("启动 Docker 沙箱容器 {container_name} 失败"))?;

            // 7.重载并刷新容器信息
            let container = docker
                .inspect_container(&created.id, None)
                .await
                .with_context(|| format!("刷新 Docker 沙箱容器 {container_name} 信息失败"))?;
            Self::get_container_ip(&container, settings.network.as_deref())
                .ok_or_else(|| anyhow!("Docker 沙箱容器 {container_name} 没有可用的 IPv4 地址"))
        }
        .await;

        let ip = match initialized {
            Ok(ip) => ip,
            Err(err) => {
                Self::remove_failed_container(&docker, &created.id, &container_name).await;
                return Err(err);
            }
        };

        Ok(Self::from_client(
            ip,
            container_name,
            Some(created.id),
            client,
        ))
    }

    async fn ensure_sandbox_with_policy(
        &self,
        max_retries: usize,
        retry_interval: Duration,
    ) -> Result<bool> {
        // 1.定义最大重试次数和重试间隔
        let mut last_issue = "尚未检查 Supervisor 状态".to_string();

        // 2.循环请求 Supervisor 状态并判断服务是否正常
        for attempt in 1..=max_retries {
            match self.load_supervisor_status().await {
                Ok(tool_result) if !tool_result.success => {
                    last_issue = tool_result
                        .message
                        .unwrap_or_else(|| "Supervisor 状态接口返回失败".to_string());
                    warn!(attempt, max_retries, issue = %last_issue, "Supervisor 进程状态监测失败");
                }
                Ok(tool_result) => {
                    // 3.读取 services 数据并判断
                    let services = tool_result.data.unwrap_or_default();
                    if services.is_empty() {
                        last_issue = "Supervisor 进程中未发现任何服务".to_string();
                        warn!(attempt, max_retries, "{last_issue}");
                    } else {
                        // 4.循环遍历所有服务并判断是否全部正常运行
                        let non_running_services = services
                            .iter()
                            .filter(|service| service.statename != "RUNNING")
                            .map(|service| format!("{}({})", service.name, service.statename))
                            .collect::<Vec<_>>();

                        if non_running_services.is_empty() {
                            info!("Sandbox Supervisor 所有进程服务运行正常");
                            return Ok(true);
                        }

                        last_issue = format!(
                            "正在等待 Sandbox Supervisor 进程服务运行，还未运行的服务列表: {}",
                            non_running_services.join(", ")
                        );
                        info!(attempt, max_retries, "{last_issue}");
                    }
                }
                Err(err) => {
                    last_issue = format!("无法确认 Sandbox Supervisor 进程状态: {err}");
                    warn!(attempt, max_retries, "{last_issue}");
                }
            }

            if attempt < max_retries {
                tokio::time::sleep(retry_interval).await;
            }
        }

        let message = format!(
            "经过 {max_retries} 次尝试后仍无法确认 Sandbox Supervisor 状态信息: {last_issue}"
        );
        error!(sandbox_id = %self.id, "{message}");
        Err(anyhow!(message))
    }

    async fn ensure_sandbox_with_timeout(
        &self,
        max_retries: usize,
        retry_interval: Duration,
        total_timeout: Duration,
    ) -> Result<bool> {
        let result = match tokio::time::timeout(
            total_timeout,
            self.ensure_sandbox_with_policy(max_retries, retry_interval),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                let message = format!(
                    "在 {} 秒内无法确认 Sandbox Supervisor 状态信息",
                    total_timeout.as_secs()
                );
                error!(sandbox_id = %self.id, "{message}");
                Err(anyhow!(message))
            }
        };

        if result.is_err() {
            lock_sandbox_cache().remove_by_id(&self.id);
        }

        result
    }

    async fn load_supervisor_status(&self) -> Result<ToolResult<Vec<SupervisorProcess>>> {
        // 调用 HTTP 客户端向沙箱发起 API 请求获取状态。
        let url = format!("{}/api/supervisor/status", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("请求 Sandbox Supervisor 状态接口 {url} 失败"))?
            .error_for_status()
            .with_context(|| format!("Sandbox Supervisor 状态接口 {url} 返回 HTTP 错误"))?;
        let body = response
            .text()
            .await
            .with_context(|| format!("读取 Sandbox Supervisor 状态响应 {url} 失败"))?;
        let response: SandboxApiResponse<Vec<SupervisorProcess>> = serde_json::from_str(&body)
            .with_context(|| format!("解析 Sandbox Supervisor 状态响应 {url} 失败"))?;

        Ok(ToolResult::from_sandbox(
            response.code,
            response.msg,
            response.data,
        ))
    }

    async fn post_tool_request(&self, endpoint: &str, body: Value) -> Result<ToolResult<Value>> {
        let url = self.api_url(endpoint);
        self.send_tool_request(&url, self.client.post(&url).json(&body))
            .await
    }

    async fn send_tool_request(
        &self,
        url: &str,
        request: RequestBuilder,
    ) -> Result<ToolResult<Value>> {
        let response = request
            .send()
            .await
            .with_context(|| format!("请求 Sandbox API {url} 失败"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .with_context(|| format!("读取 Sandbox API {url} 响应失败"))?;
        let response: SandboxApiResponse<Value> = serde_json::from_str(&body)
            .with_context(|| format!("解析 Sandbox API {url} 响应失败，HTTP 状态码 {status}"))?;

        Ok(ToolResult::from_sandbox(
            response.code,
            response.msg,
            response.data,
        ))
    }

    fn api_url(&self, endpoint: &str) -> String {
        format!("{}/api/{endpoint}", self.base_url)
    }

    async fn remove_failed_container(docker: &Docker, container_id: &str, container_name: &str) {
        let options = RemoveContainerOptionsBuilder::default().force(true).build();
        if let Err(err) = docker.remove_container(container_id, Some(options)).await {
            error!(
                container_name,
                container_id,
                error = %err,
                "清理初始化失败的 Docker 沙箱容器失败"
            );
        }
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    /// 在沙箱中执行命令。
    async fn exec_command(
        &self,
        session_id: &str,
        exec_dir: &str,
        command: &str,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "shell/exec-command",
                json!({
                    "session_id": session_id,
                    "exec_dir": exec_dir,
                    "command": command,
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 读取沙箱中 Shell 的输出。
    async fn read_shell_output(
        &self,
        session_id: &str,
        console: Option<bool>,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "shell/read-shell-output",
                json!({
                    "session_id": session_id,
                    "console": console.unwrap_or(false),
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 等待沙箱中进程的执行。
    async fn wait_process(
        &self,
        session_id: &str,
        seconds: Option<usize>,
    ) -> Result<ToolResult<String>> {
        let seconds = seconds
            .map(i64::try_from)
            .transpose()
            .context("等待秒数超出 Sandbox API 支持范围")?;
        let result = self
            .post_tool_request(
                "shell/wait-process",
                json!({
                    "session_id": session_id,
                    "seconds": seconds,
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 向沙箱的 Shell 进程写入数据。
    async fn write_shell_input(
        &self,
        session_id: &str,
        input_text: &str,
        press_enter: Option<bool>,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "shell/write-shell-input",
                json!({
                    "session_id": session_id,
                    "input_text": input_text,
                    "press_enter": press_enter.unwrap_or(true),
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 杀死沙箱中的指定进程。
    async fn kill_process(&self, session_id: &str) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request("shell/kill-process", json!({ "session_id": session_id }))
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 向沙箱中的指定文件写入内容。
    async fn write_file(
        &self,
        file_path: &str,
        content: &str,
        append: Option<bool>,
        leading_newline: Option<bool>,
        trailing_newline: Option<bool>,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "file/write-file",
                json!({
                    "file_path": file_path,
                    "content": content,
                    "append": append.unwrap_or(false),
                    "leading_newline": leading_newline.unwrap_or(false),
                    "trailing_newline": trailing_newline.unwrap_or(false),
                    "sudo": sudo.unwrap_or(false),
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 读取沙箱中指定路径的文件内容。
    async fn read_file(
        &self,
        file_path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "file/read-file",
                json!({
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "sudo": sudo.unwrap_or(false),
                    "max_length": max_length.unwrap_or(10_000),
                }),
            )
            .await?;
        map_tool_data(result, |value| {
            decode_tool_data::<FileReadData>(value, "读取文件").map(|data| data.content)
        })
    }

    /// 传递指定路径检查沙箱中的文件是否存在。
    async fn check_file_exists(&self, file_path: &str) -> Result<ToolResult<bool>> {
        let result = self
            .post_tool_request("file/check-file-exists", json!({ "file_path": file_path }))
            .await?;
        map_tool_data(result, |value| {
            decode_tool_data::<FileCheckData>(value, "检查文件").map(|data| data.exists)
        })
    }

    /// 删除沙箱中的指定文件。
    async fn delete_file(&self, file_path: &str) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request("file/delete-file", json!({ "file_path": file_path }))
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 替换沙箱中文件的旧内容为指定内容。
    async fn replace_in_file(
        &self,
        file_path: &str,
        old_str: &str,
        new_str: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        let result = self
            .post_tool_request(
                "file/replace-in-file",
                json!({
                    "file_path": file_path,
                    "old_str": old_str,
                    "new_str": new_str,
                    "sudo": sudo.unwrap_or(false),
                }),
            )
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 搜索沙箱中指定文件的内容。
    async fn search_in_file(
        &self,
        file_path: &str,
        regex: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<Vec<String>>> {
        let result = self
            .post_tool_request(
                "file/search-in-file",
                json!({
                    "file_path": file_path,
                    "regex": regex,
                    "sudo": sudo.unwrap_or(false),
                }),
            )
            .await?;
        map_tool_data(result, |value| {
            decode_tool_data::<FileSearchData>(value, "搜索文件").map(|data| data.matches)
        })
    }

    /// 查找沙箱中指定目录的文件列表。
    async fn find_files(
        &self,
        dir_path: &str,
        glob_pattern: &str,
    ) -> Result<ToolResult<Vec<String>>> {
        let result = self
            .post_tool_request(
                "file/find-files",
                json!({
                    "dir_path": dir_path,
                    "glob_pattern": glob_pattern,
                }),
            )
            .await?;
        map_tool_data(result, |value| {
            decode_tool_data::<FileFindData>(value, "查找文件").map(|data| data.files)
        })
    }

    /// 将文件源上传至沙箱指定位置。
    async fn upload_file(
        &self,
        file_data: Vec<u8>,
        file_path: &str,
        file_name: Option<&str>,
    ) -> Result<ToolResult<String>> {
        // 1.预配置上传数据
        let file_name = file_name
            .filter(|name| !name.is_empty())
            .unwrap_or("upload");
        let part = Part::bytes(file_data)
            .file_name(file_name.to_string())
            .mime_str("application/octet-stream")
            .context("构建 Sandbox 文件上传表单失败")?;
        let form = Form::new()
            .part("file", part)
            .text("file_path", file_path.to_string());

        // 2.发起请求上传数据并获取响应
        let url = self.api_url("file/upload-file");
        let result = self
            .send_tool_request(&url, self.client.post(&url).multipart(form))
            .await?;
        map_tool_data(result, json_value_to_string)
    }

    /// 从沙箱中下载文件。
    async fn download_file(&self, file_path: &str) -> Result<Vec<u8>> {
        let url = self.api_url("file/download-file");
        let response = self
            .client
            .get(&url)
            .query(&[("file_path", file_path)])
            .send()
            .await
            .with_context(|| format!("请求 Sandbox 文件下载接口 {url} 失败"))?
            .error_for_status()
            .with_context(|| format!("Sandbox 文件下载接口 {url} 返回 HTTP 错误"))?;

        Ok(response
            .bytes()
            .await
            .with_context(|| format!("读取 Sandbox 下载文件 {url} 失败"))?
            .to_vec())
    }

    async fn ensure_sandbox(&self) -> Result<bool> {
        DockerSandbox::ensure_sandbox(self).await
    }

    async fn destroy(&self) -> Result<bool> {
        DockerSandbox::destroy(self).await
    }

    async fn get_browser(&self) -> Result<Box<dyn Browser>> {
        DockerSandbox::get_browser(self).await
    }

    fn id(&self) -> &str {
        DockerSandbox::id(self)
    }

    fn cdp_url(&self) -> &str {
        DockerSandbox::cdp_url(self)
    }

    fn vnc_url(&self) -> &str {
        DockerSandbox::vnc_url(self)
    }
}

#[derive(Debug, Deserialize)]
struct SandboxApiResponse<T> {
    code: i32,
    msg: String,
    data: Option<T>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SupervisorProcess {
    name: String,
    statename: String,
}

#[derive(Debug, Deserialize)]
struct FileReadData {
    content: String,
}

#[derive(Debug, Deserialize)]
struct FileSearchData {
    matches: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FileFindData {
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FileCheckData {
    exists: bool,
}

#[derive(Debug, Default)]
struct HostnameCache {
    entries: VecDeque<(String, Ipv4Addr)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SandboxCacheKey {
    id: String,
    address: Option<String>,
    network: Option<String>,
}

impl SandboxCacheKey {
    fn new(settings: &SandboxSettings, id: &str) -> Self {
        Self {
            id: id.to_string(),
            address: settings.address.clone(),
            network: settings.network.clone(),
        }
    }
}

#[derive(Debug, Default)]
struct SandboxCache {
    entries: VecDeque<(SandboxCacheKey, DockerSandbox)>,
}

impl SandboxCache {
    fn get(&mut self, key: &SandboxCacheKey) -> Option<DockerSandbox> {
        let index = self.entries.iter().position(|(cached, _)| cached == key)?;
        let entry = self.entries.remove(index)?;
        let sandbox = entry.1.clone();
        self.entries.push_back(entry);
        Some(sandbox)
    }

    fn insert(&mut self, key: SandboxCacheKey, sandbox: DockerSandbox) {
        if let Some(index) = self.entries.iter().position(|(cached, _)| cached == &key) {
            self.entries.remove(index);
        }

        if self.entries.len() == SANDBOX_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((key, sandbox));
    }

    fn remove_by_id(&mut self, id: &str) {
        self.entries.retain(|(key, _)| key.id != id);
    }
}

impl HostnameCache {
    fn get(&mut self, hostname: &str) -> Option<Ipv4Addr> {
        let index = self
            .entries
            .iter()
            .position(|(cached, _)| cached == hostname)?;
        let entry = self.entries.remove(index)?;
        let ip = entry.1;
        self.entries.push_back(entry);
        Some(ip)
    }

    fn insert(&mut self, hostname: String, ip: Ipv4Addr) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(cached, _)| cached == &hostname)
        {
            self.entries.remove(index);
        }

        if self.entries.len() == HOSTNAME_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((hostname, ip));
    }
}

fn lock_hostname_cache() -> MutexGuard<'static, HostnameCache> {
    HOSTNAME_CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn lock_sandbox_cache() -> MutexGuard<'static, SandboxCache> {
    SANDBOX_CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn build_container_name(name_prefix: &str) -> String {
    let uuid = Uuid::new_v4().simple().to_string();
    format!("{name_prefix}-{}", &uuid[..8])
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .timeout(SANDBOX_REQUEST_TIMEOUT)
        .build()
        .context("创建沙箱 HTTP 客户端失败")
}

fn build_container_config(settings: &SandboxSettings) -> Result<ContainerCreateBody> {
    let image = required_setting(&settings.image, "settings.sandbox.image")?;
    let mut environment = Vec::with_capacity(5);

    if let Some(ttl_minutes) = settings.ttl_minutes {
        environment.push(format!("SERVER_TIMEOUT_MINUTES={ttl_minutes}"));
    }
    push_optional_env(&mut environment, "CHROME_ARGS", &settings.chrome_args);
    push_optional_env(&mut environment, "HTTPS_PROXY", &settings.https_proxy);
    push_optional_env(&mut environment, "HTTP_PROXY", &settings.http_proxy);
    push_optional_env(&mut environment, "NO_PROXY", &settings.no_proxy);

    Ok(ContainerCreateBody {
        image: Some(image.to_string()),
        env: Some(environment),
        host_config: Some(HostConfig {
            auto_remove: Some(true),
            network_mode: settings.network.clone(),
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn push_optional_env(environment: &mut Vec<String>, name: &str, value: &Option<String>) {
    if let Some(value) = value {
        environment.push(format!("{name}={value}"));
    }
}

fn required_setting<'a>(value: &'a Option<String>, name: &str) -> Result<&'a str> {
    value
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("{name} 不能为空"))
}

fn validate_managed_container_id(id: &str, name_prefix: &str) -> Result<()> {
    let expected_prefix = format!("{name_prefix}-");
    if id.starts_with(&expected_prefix) {
        return Ok(());
    }

    Err(anyhow!(
        "Docker 沙箱容器 ID {id} 不属于配置的名称前缀 {name_prefix}"
    ))
}

fn endpoint_ipv4(endpoint: &EndpointSettings) -> Option<Ipv4Addr> {
    endpoint.ip_address.as_deref()?.parse().ok()
}

fn map_tool_data<T>(
    result: ToolResult<Value>,
    mapper: impl FnOnce(Value) -> Result<T>,
) -> Result<ToolResult<T>> {
    let ToolResult {
        success,
        message,
        data,
    } = result;
    let data = if success {
        data.map(mapper).transpose()?
    } else {
        None
    };

    Ok(ToolResult {
        success,
        message,
        data,
    })
}

fn decode_tool_data<T: DeserializeOwned>(value: Value, operation: &str) -> Result<T> {
    serde_json::from_value(value).with_context(|| format!("解析 Sandbox {operation}响应数据失败"))
}

fn json_value_to_string(value: Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value),
        value => serde_json::to_string(&value).context("序列化 Sandbox 响应数据失败"),
    }
}

fn is_container_not_found(error: &BollardError) -> bool {
    matches!(
        error,
        BollardError::DockerResponseServerError {
            status_code: 404,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        future::pending,
        net::Ipv4Addr,
        sync::{Arc, Mutex as StdMutex},
        time::Duration,
    };

    use axum::{
        body::{to_bytes, Body},
        extract::State,
        http::{Method, Request, StatusCode},
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use bollard::{
        models::{ContainerInspectResponse, EndpointSettings, NetworkSettings},
        Docker,
    };
    use serde_json::{json, Value};
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::{
        build_container_config, build_container_name, is_container_not_found,
        validate_managed_container_id, BollardError, DockerSandbox, HostnameCache,
        HOSTNAME_CACHE_CAPACITY,
    };
    use crate::{domain::external::Sandbox, infrastructure::settings::SandboxSettings};

    #[derive(Debug, Clone)]
    struct RecordedRequest {
        method: Method,
        path: String,
        query: Option<String>,
        body: Vec<u8>,
    }

    type RecordedRequests = Arc<StdMutex<Vec<RecordedRequest>>>;

    async fn record_sandbox_request(
        State(requests): State<RecordedRequests>,
        request: Request<Body>,
    ) -> Response {
        let method = request.method().clone();
        let path = request.uri().path().to_string();
        let query = request.uri().query().map(str::to_string);
        let body = to_bytes(request.into_body(), 1_000_000)
            .await
            .unwrap()
            .to_vec();
        requests.lock().unwrap().push(RecordedRequest {
            method,
            path: path.clone(),
            query,
            body,
        });

        if path == "/api/file/download-file" {
            return Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("download-data"))
                .unwrap();
        }

        let (status, code, message, data) = match path.as_str() {
            "/api/file/read-file" => (
                StatusCode::OK,
                200,
                "read ok",
                json!({ "file_path": "/tmp/agent.txt", "content": "file contents" }),
            ),
            "/api/file/search-in-file" => (
                StatusCode::OK,
                200,
                "search ok",
                json!({
                    "file_path": "/tmp/agent.txt",
                    "matches": ["Agent"],
                    "line_numbers": [0]
                }),
            ),
            "/api/file/find-files" => (
                StatusCode::OK,
                200,
                "find ok",
                json!({ "dir_path": "/tmp", "files": ["/tmp/agent.txt"] }),
            ),
            "/api/file/check-file-exists" => (
                StatusCode::OK,
                200,
                "check ok",
                json!({ "file_path": "/tmp/agent.txt", "exists": true }),
            ),
            "/api/file/delete-file" => (StatusCode::BAD_REQUEST, 400, "delete denied", json!({})),
            _ => (StatusCode::OK, 200, "success", json!({ "endpoint": path })),
        };

        (
            status,
            Json(json!({
                "code": code,
                "msg": message,
                "data": data,
            })),
        )
            .into_response()
    }

    #[test]
    fn builds_urls_and_uses_container_name_as_id() {
        let sandbox = DockerSandbox::new(
            Ipv4Addr::new(172, 18, 0, 2),
            Some("mooc-manus-sandbox-a1b2c3d4".to_string()),
        )
        .unwrap();

        assert_eq!(sandbox.id(), "mooc-manus-sandbox-a1b2c3d4");
        assert_eq!(sandbox.ip(), Ipv4Addr::new(172, 18, 0, 2));
        assert_eq!(sandbox.base_url(), "http://172.18.0.2:3000");
        assert_eq!(sandbox.vnc_url(), "ws://172.18.0.2:5901");
        assert_eq!(sandbox.cdp_url(), "http://172.18.0.2:9222");
    }

    #[test]
    fn uses_default_id_for_an_existing_sandbox() {
        let sandbox = DockerSandbox::new(Ipv4Addr::LOCALHOST, None).unwrap();

        assert_eq!(sandbox.id(), "lenexus-sandbox");
    }

    #[tokio::test]
    async fn returns_an_ipv4_address_without_dns_lookup() {
        let ip = DockerSandbox::resolve_hostname_to_ip("127.0.0.1").await;

        assert_eq!(ip, Some(Ipv4Addr::LOCALHOST));
    }

    #[tokio::test]
    async fn reuses_an_existing_sandbox_from_settings() {
        let settings = SandboxSettings {
            address: Some("127.0.0.1".to_string()),
            name_prefix: Some("custom-container-prefix".to_string()),
            ..Default::default()
        };

        let sandbox = DockerSandbox::create(&settings).await.unwrap();

        assert_eq!(sandbox.id(), "lenexus-sandbox");
        assert_eq!(sandbox.base_url(), "http://127.0.0.1:3000");
    }

    #[tokio::test]
    async fn gets_and_destroys_an_unmanaged_sandbox_without_using_docker() {
        let settings = SandboxSettings {
            address: Some("127.0.0.1".to_string()),
            ..Default::default()
        };

        let sandbox = DockerSandbox::get(&settings, "shared-sandbox")
            .await
            .unwrap();

        assert_eq!(sandbox.id(), "shared-sandbox");
        assert_eq!(sandbox.container_id, None);
        assert!(sandbox.destroy().await.unwrap());
    }

    #[tokio::test]
    async fn confirms_sandbox_when_all_supervisor_services_are_running() {
        let (sandbox, server) = sandbox_with_supervisor_response(json!({
            "code": 200,
            "msg": "获取沙箱进程服务成功",
            "data": [
                { "name": "api", "statename": "RUNNING" },
                { "name": "chrome", "statename": "RUNNING" }
            ]
        }))
        .await;

        let result = sandbox.ensure_sandbox_with_policy(1, Duration::ZERO).await;
        server.abort();

        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn reports_services_that_are_not_running() {
        let (sandbox, server) = sandbox_with_supervisor_response(json!({
            "code": 200,
            "msg": "获取沙箱进程服务成功",
            "data": [
                { "name": "api", "statename": "RUNNING" },
                { "name": "chrome", "statename": "STARTING" }
            ]
        }))
        .await;

        let error = sandbox
            .ensure_sandbox_with_policy(1, Duration::ZERO)
            .await
            .unwrap_err();
        server.abort();

        assert!(error.to_string().contains("chrome(STARTING)"));
    }

    #[tokio::test]
    async fn bounds_the_total_supervisor_readiness_wait() {
        let app = Router::new().route(
            "/api/supervisor/status",
            get(|| async { pending::<Json<Value>>().await }),
        );
        let (sandbox, server) = sandbox_with_router(app).await;

        let error = sandbox
            .ensure_sandbox_with_timeout(30, Duration::from_secs(2), Duration::from_millis(20))
            .await
            .unwrap_err();
        server.abort();

        assert!(error
            .to_string()
            .contains("无法确认 Sandbox Supervisor 状态"));
    }

    #[test]
    fn treats_an_already_removed_container_as_destroyed() {
        let error = BollardError::DockerResponseServerError {
            status_code: 404,
            message: "No such container".to_string(),
        };

        assert!(is_container_not_found(&error));
    }

    #[test]
    fn rejects_a_container_outside_the_configured_name_prefix() {
        let error = validate_managed_container_id("postgres", "lenexus-sandbox").unwrap_err();

        assert!(error.to_string().contains("不属于配置的名称前缀"));
        assert!(
            validate_managed_container_id("lenexus-sandbox-a1b2c3d4", "lenexus-sandbox").is_ok()
        );
    }

    #[test]
    fn hostname_cache_evicts_the_least_recently_used_entry() {
        let mut cache = HostnameCache::default();
        for index in 0..HOSTNAME_CACHE_CAPACITY {
            cache.insert(
                format!("host-{index}"),
                Ipv4Addr::new(10, 0, 0, index as u8),
            );
        }

        assert!(cache.get("host-0").is_some());
        cache.insert("host-128".to_string(), Ipv4Addr::new(10, 0, 0, 128));

        assert!(cache.get("host-0").is_some());
        assert!(cache.get("host-1").is_none());
    }

    #[test]
    fn prefers_the_configured_network_ipv4() {
        let container = ContainerInspectResponse {
            network_settings: Some(NetworkSettings {
                networks: Some(HashMap::from([
                    (
                        "other".to_string(),
                        EndpointSettings {
                            ip_address: Some("172.19.0.4".to_string()),
                            ..Default::default()
                        },
                    ),
                    (
                        "sandbox".to_string(),
                        EndpointSettings {
                            ip_address: Some("172.20.0.3".to_string()),
                            ..Default::default()
                        },
                    ),
                ])),
                ..Default::default()
            }),
            ..Default::default()
        };

        assert_eq!(
            DockerSandbox::get_container_ip(&container, Some("sandbox")),
            Some(Ipv4Addr::new(172, 20, 0, 3))
        );
    }

    #[test]
    fn builds_python_equivalent_container_configuration() {
        let settings = SandboxSettings {
            image: Some("mooc-manus-sandbox".to_string()),
            name_prefix: Some("mooc-manus-sandbox".to_string()),
            ttl_minutes: Some(60),
            network: Some("mooc-manus-network".to_string()),
            chrome_args: Some("--headless=new".to_string()),
            https_proxy: Some("https://proxy.example".to_string()),
            http_proxy: Some("http://proxy.example".to_string()),
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            ..Default::default()
        };

        let name = build_container_name(settings.name_prefix.as_deref().unwrap());
        let config = build_container_config(&settings).unwrap();
        let host_config = config.host_config.unwrap();

        assert_eq!(name.len(), "mooc-manus-sandbox-".len() + 8);
        assert!(name.starts_with("mooc-manus-sandbox-"));
        assert!(name
            .rsplit_once('-')
            .unwrap()
            .1
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
        assert_eq!(config.image.as_deref(), Some("mooc-manus-sandbox"));
        assert_eq!(
            config.env.unwrap(),
            vec![
                "SERVER_TIMEOUT_MINUTES=60",
                "CHROME_ARGS=--headless=new",
                "HTTPS_PROXY=https://proxy.example",
                "HTTP_PROXY=http://proxy.example",
                "NO_PROXY=localhost,127.0.0.1",
            ]
        );
        assert_eq!(host_config.auto_remove, Some(true));
        assert_eq!(
            host_config.network_mode.as_deref(),
            Some("mooc-manus-network")
        );
    }

    #[tokio::test]
    async fn implements_file_and_shell_api_contracts() {
        let requests = RecordedRequests::default();
        let app = Router::new()
            .fallback(record_sandbox_request)
            .with_state(Arc::clone(&requests));
        let (sandbox, server) = sandbox_with_router(app).await;

        let read = Sandbox::read_file(&sandbox, "/tmp/agent.txt", None, None, None, None)
            .await
            .unwrap();
        let write =
            Sandbox::write_file(&sandbox, "/tmp/agent.txt", "Agent", None, None, None, None)
                .await
                .unwrap();
        let replace = Sandbox::replace_in_file(&sandbox, "/tmp/agent.txt", "Agent", "Manus", None)
            .await
            .unwrap();
        let search = Sandbox::search_in_file(&sandbox, "/tmp/agent.txt", "Agent", None)
            .await
            .unwrap();
        let found = Sandbox::find_files(&sandbox, "/tmp", "*.txt")
            .await
            .unwrap();
        let exists = Sandbox::check_file_exists(&sandbox, "/tmp/agent.txt")
            .await
            .unwrap();
        let deleted = Sandbox::delete_file(&sandbox, "/tmp/agent.txt")
            .await
            .unwrap();
        let uploaded = Sandbox::upload_file(
            &sandbox,
            b"upload-data".to_vec(),
            "/tmp/uploaded.bin",
            Some("agent.bin"),
        )
        .await
        .unwrap();
        let downloaded = Sandbox::download_file(&sandbox, "/tmp/download.txt")
            .await
            .unwrap();
        let executed = Sandbox::exec_command(&sandbox, "session-1", "/tmp", "pwd")
            .await
            .unwrap();
        let shell_output = Sandbox::read_shell_output(&sandbox, "session-1", None)
            .await
            .unwrap();
        let shell_input = Sandbox::write_shell_input(&sandbox, "session-1", "yes", None)
            .await
            .unwrap();
        let waited = Sandbox::wait_process(&sandbox, "session-1", None)
            .await
            .unwrap();
        let killed = Sandbox::kill_process(&sandbox, "session-1").await.unwrap();
        server.abort();

        assert_eq!(read.data.as_deref(), Some("file contents"));
        assert_eq!(search.data, Some(vec!["Agent".to_string()]));
        assert_eq!(found.data, Some(vec!["/tmp/agent.txt".to_string()]));
        assert_eq!(exists.data, Some(true));
        assert!(!deleted.success);
        assert_eq!(deleted.message.as_deref(), Some("delete denied"));
        assert_eq!(deleted.data, None);
        assert_eq!(downloaded, b"download-data");
        for result in [
            write,
            replace,
            uploaded,
            executed,
            shell_output,
            shell_input,
            waited,
            killed,
        ] {
            let data = result.data.expect("successful result should contain data");
            assert!(serde_json::from_str::<Value>(&data).unwrap().is_object());
        }

        let requests = requests.lock().unwrap();
        let paths = requests
            .iter()
            .map(|request| request.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "/api/file/read-file",
                "/api/file/write-file",
                "/api/file/replace-in-file",
                "/api/file/search-in-file",
                "/api/file/find-files",
                "/api/file/check-file-exists",
                "/api/file/delete-file",
                "/api/file/upload-file",
                "/api/file/download-file",
                "/api/shell/exec-command",
                "/api/shell/read-shell-output",
                "/api/shell/write-shell-input",
                "/api/shell/wait-process",
                "/api/shell/kill-process",
            ]
        );
        assert!(requests.iter().all(|request| {
            request.method == Method::POST || request.path == "/api/file/download-file"
        }));
        assert_eq!(requests[8].method, Method::GET);

        let read_body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(read_body["file_path"], "/tmp/agent.txt");
        assert_eq!(read_body["sudo"], false);
        assert_eq!(read_body["max_length"], 10_000);
        let upload_body = String::from_utf8_lossy(&requests[7].body);
        assert!(upload_body.contains("name=\"file\""));
        assert!(upload_body.contains("filename=\"agent.bin\""));
        assert!(upload_body.contains("name=\"file_path\""));
        assert!(upload_body.contains("/tmp/uploaded.bin"));
        assert!(upload_body.contains("upload-data"));
        assert!(requests[8]
            .query
            .as_deref()
            .is_some_and(|query| query.contains("file_path=")));
        let shell_input_body: Value = serde_json::from_slice(&requests[11].body).unwrap();
        assert_eq!(shell_input_body["press_enter"], true);
        let wait_body: Value = serde_json::from_slice(&requests[12].body).unwrap();
        assert!(wait_body["seconds"].is_null());
    }

    #[tokio::test]
    #[ignore = "requires a live Docker daemon"]
    async fn connects_to_the_live_docker_daemon() {
        let docker = Docker::connect_with_defaults().unwrap();

        docker.ping().await.unwrap();
    }

    async fn sandbox_with_supervisor_response(response: Value) -> (DockerSandbox, JoinHandle<()>) {
        let app = Router::new().route(
            "/api/supervisor/status",
            get(move || {
                let response = response.clone();
                async move { Json(response) }
            }),
        );
        sandbox_with_router(app).await
    }

    async fn sandbox_with_router(app: Router) -> (DockerSandbox, JoinHandle<()>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut sandbox = DockerSandbox::new(Ipv4Addr::LOCALHOST, None).unwrap();
        sandbox.base_url = format!("http://{address}");
        (sandbox, server)
    }
}
