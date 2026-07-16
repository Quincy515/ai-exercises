use std::{
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr},
    sync::{LazyLock, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use bollard::{
    models::{ContainerCreateBody, ContainerInspectResponse, EndpointSettings, HostConfig},
    query_parameters::{CreateContainerOptionsBuilder, RemoveContainerOptionsBuilder},
    Docker,
};
use reqwest::Client;
use tokio::net::lookup_host;
use tracing::error;
use uuid::Uuid;

use crate::infrastructure::settings::SandboxSettings;

const DEFAULT_SANDBOX_ID: &str = "lenexus-sandbox";
const SANDBOX_API_PORT: u16 = 3000;
const SANDBOX_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const HOSTNAME_CACHE_CAPACITY: usize = 128;

static HOSTNAME_CACHE: LazyLock<Mutex<HostnameCache>> =
    LazyLock::new(|| Mutex::new(HostnameCache::default()));

/// 基于 Docker 的沙箱服务。
#[derive(Debug, Clone)]
pub struct DockerSandbox {
    client: Client,
    ip: Ipv4Addr,
    id: String,
    base_url: String,
    vnc_url: String,
    cdp_url: String,
}

impl DockerSandbox {
    /// 构造函数，完成 Docker 沙箱扩展创建。
    pub fn new(ip: Ipv4Addr, container_name: Option<String>) -> Result<Self> {
        let client = build_http_client()?;

        Ok(Self::from_client(ip, container_name, client))
    }

    fn from_client(ip: Ipv4Addr, container_name: Option<String>, client: Client) -> Self {
        Self {
            client,
            ip,
            id: container_name.unwrap_or_else(|| DEFAULT_SANDBOX_ID.to_string()),
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

        Ok(Self::from_client(ip, Some(container_name), client))
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

#[derive(Debug, Default)]
struct HostnameCache {
    entries: VecDeque<(String, Ipv4Addr)>,
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

fn endpoint_ipv4(endpoint: &EndpointSettings) -> Option<Ipv4Addr> {
    endpoint.ip_address.as_deref()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, net::Ipv4Addr};

    use bollard::{
        models::{ContainerInspectResponse, EndpointSettings, NetworkSettings},
        Docker,
    };

    use super::{
        build_container_config, build_container_name, DockerSandbox, HostnameCache,
        HOSTNAME_CACHE_CAPACITY,
    };
    use crate::infrastructure::settings::SandboxSettings;

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
    #[ignore = "requires a live Docker daemon"]
    async fn connects_to_the_live_docker_daemon() {
        let docker = Docker::connect_with_defaults().unwrap();

        docker.ping().await.unwrap();
    }
}
