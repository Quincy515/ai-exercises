//! MCP 客户端管理器的开发思路:
//! 1. 在 Agent 执行的过程中，有可能需要调用多次工具,
//!    但是因为 MCP 工具的每次获取都需要调用客户端会话的 list_tools() 方法,
//!    非常耗时, 所以需要我们缓存工具的参数信息, 只有在初始化的时候才调用一次,
//!    并且在销毁 MCP 客户端管理器的时候一并清除;
//! 2. 在前端 UI 交互中, 无论 MCP 服务是否启动, 都会显示工具列表信息,
//!    但是在 Agent 执行的过程中, 我们只会传递已启动的 MCP 服务,
//!    所以 MCP 客户端管理器会根据接收的 MCP 配置筛选并加载已启用的服务器;
//! 3. MCP 客户端管理器会同时管理多个 MCP 服务, 支持 stdio、streamable_http 传输协议.
//!    需要根据传输协议的不同来创建客户端会话(ClientSession), 同时缓存会话;
//! 4. 另外有可能有一些环境变量是存储在我们整个系统中的, 在初始化 MCP 服务的时候，需要将传递进来的
//!    环境变量与系统的环境变量进行合并后传递给 MCP 服务;
//! 5. 使用 AsyncExitStack 异步上下文管理器来管理上下文，避免使用 with 多层嵌套;
//! 6. MCP 客户端管理器的初始化非常耗时, 所以需要有机制可以判断避免重复初始化;
//! 7. MCP 配置来自数据库或其他外部输入, 初始化连接时必须二次校验;
//! 8. 同时缓存 ClientSession+Tool-Schema, 一个是客户端会话, 一个是工具参数声明;
//! 9. MCP 客户端管理器在清除/停止使用的时候, 必须关闭异步上下文管理器、清除资源(ClientSession、Tool-Schema)、
//!    初始化标识等, 从而避免资源泄露;

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::header::{HeaderName, HeaderValue};
use rmcp::{
    model::{CallToolRequestParams, ClientInfo, Tool},
    service::RunningService,
    transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
        TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use serde_json::{json, Map, Value};
use tracing::{error, info};

use crate::domain::models::{McpConfig, McpServerConfig, McpTransport, ToolResult};

type ClientSession = RunningService<RoleClient, ClientInfo>;

/// MCP 客户端管理器
pub struct McpClientManager {
    /// mcp 配置信息
    mcp_config: Option<McpConfig>,
    /// 缓存的客户端会话
    clients: HashMap<String, ClientSession>,
    /// 缓存的 MCP 工具参数声明
    tools: HashMap<String, Vec<Tool>>,
    /// 是否初始化标识
    initialized: bool,
}

impl McpClientManager {
    /// 构造函数，完成 MCP 客户端管理器的初步初始化
    pub fn new(mcp_config: Option<McpConfig>) -> Self {
        Self {
            mcp_config,
            clients: HashMap::new(),
            tools: HashMap::new(),
            initialized: false,
        }
    }

    /// 初始化函数，用于连接所有配置的 MCP 服务器
    pub async fn initialize(&mut self) {
        // 1.检查下是否已经初始化成功
        if self.initialized {
            return;
        }

        // 2.记录日志并连接 MCP 服务器
        let server_count = self.mcp_config.as_ref().map_or(0, |config| {
            config
                .mcp_servers
                .values()
                .filter(|server| server.enabled)
                .count()
        });
        info!("从传入配置中加载了{server_count}个已启用的MCP服务器");
        self.connect_mcp_servers().await;
        self.initialized = true;
        info!("MCP客户端管理器加载成功");
    }

    /// 根据配置连接所有 MCP 服务
    async fn connect_mcp_servers(&mut self) {
        // 1.循环遍历传递进来的所有 MCP 服务器，只连接 enabled 为 true 的服务
        let servers = self
            .mcp_config
            .as_ref()
            .map(|config| {
                config
                    .mcp_servers
                    .iter()
                    .filter(|(_, config)| config.enabled)
                    .map(|(name, config)| (name.clone(), config.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for (server_name, server_config) in servers {
            // 2.根据服务名字+服务配置连接到 MCP 服务器
            if let Err(error) = self.connect_mcp_server(&server_name, &server_config).await {
                // 3.记录错误日志并跳过错误的 MCP 服务器
                error!("连接MCP服务器[{server_name}]出错: {error}");
            }
        }
    }

    /// 根据传递的服务名字+服务配置连接到单个 MCP 服务
    async fn connect_mcp_server(
        &mut self,
        server_name: &str,
        server_config: &McpServerConfig,
    ) -> Result<()> {
        // 1.获取 MCP 服务的传输协议
        let transport = server_config.transport;

        // 2.根据不同的传输协议调用不同的方法连接 MCP 服务器
        let result = match transport {
            McpTransport::Stdio => self.connect_stdio_server(server_name, server_config).await,
            McpTransport::StreamableHttp => {
                self.connect_streamable_http_server(server_name, server_config)
                    .await
            }
        };

        // 3.记录日志并抛出异常
        if let Err(error) = &result {
            error!("连接MCP服务器[{server_name}]出错: {error}");
        }
        result
    }

    /// 根据服务名字+配置连接stdio服务
    async fn connect_stdio_server(
        &mut self,
        server_name: &str,
        server_config: &McpServerConfig,
    ) -> Result<()> {
        let result: Result<()> = async {
            // 1.从配置中提取相关命令信息
            let command = server_config.command.as_deref().unwrap_or_default().trim();
            let args = server_config.args.as_deref().unwrap_or_default();
            let env = server_config.env.as_ref();

            // 2.检查command是否存在
            if command.is_empty() {
                bail!("连接stdio-mcp服务器需要配置command命令");
            }

            // 3.构建stdio连接参数
            let mut server_command = tokio::process::Command::new(command);
            server_command.args(args);
            if let Some(env) = env {
                server_command.envs(value_map_to_strings(env, "环境变量")?);
            }

            // 4.使用异步上下文管理器创建传输协议
            let stdio_transport =
                TokioChildProcess::new(server_command).context("创建stdio-mcp传输协议失败")?;

            // 5.根据读取与写入流构建会话
            let session = ClientInfo::default()
                .serve(stdio_transport)
                .await
                .context("创建stdio-mcp客户端会话失败")?;

            // 6.初始化MCP服务会话由rmcp的serve方法完成

            // 7.缓存对应的mcp连接客户端
            self.clients.insert(server_name.to_string(), session);

            // 8.缓存对应mcp服务的工具列表
            self.cache_mcp_server_tools(server_name).await;
            info!("连接stdio-mcp服务器成功: {server_name}");
            Ok(())
        }
        .await;

        if let Err(error) = &result {
            // 记录错误日志并直接抛出异常
            error!("连接stdio-mcp服务器失败: {error}");
        }
        result
    }

    /// 根据服务名字+配置连接streamable-http服务
    async fn connect_streamable_http_server(
        &mut self,
        server_name: &str,
        server_config: &McpServerConfig,
    ) -> Result<()> {
        let result: Result<()> = async {
            // 1.提取streamable-http服务器的连接url并判断是否存在
            let url = server_config.url.as_deref().unwrap_or_default().trim();
            if url.is_empty() {
                bail!("连接streamable-http-mcp服务器需要配置url");
            }

            // 2.连接streamable-http服务
            let headers = http_headers(server_config.headers.as_ref())?;
            let transport_config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
                .custom_headers(headers)
                .reinit_on_expired_session(true);
            let streamable_http_transport =
                StreamableHttpClientTransport::from_config(transport_config);

            // 3.streamable-http传输由rmcp统一管理输入与输出流

            // 4.创建客户端会话
            let session = ClientInfo::default()
                .serve(streamable_http_transport)
                .await
                .context("创建streamable-http-mcp客户端会话失败")?;

            // 5.初始化MCP服务会话由rmcp的serve方法完成

            // 6.缓存对应的mcp连接客户端
            self.clients.insert(server_name.to_string(), session);

            // 7.缓存对应mcp服务的工具列表
            self.cache_mcp_server_tools(server_name).await;
            info!("连接streamable-http-mcp服务器成功: {server_name}");
            Ok(())
        }
        .await;

        if let Err(error) = &result {
            // 7.记录错误日志并直接抛出异常
            error!("连接streamable-http-mcp服务器失败: {error}");
        }
        result
    }

    /// 根据传递的服务名字+会话缓存mcp服务工具列表
    async fn cache_mcp_server_tools(&mut self, server_name: &str) {
        let Some(session) = self.clients.get(server_name) else {
            self.tools.insert(server_name.to_string(), Vec::new());
            return;
        };

        match session.list_all_tools().await {
            Ok(tools) => {
                info!("MCP服务器[{server_name}]提供了{}个工具", tools.len());
                self.tools.insert(server_name.to_string(), tools);
            }
            Err(error) => {
                // 记录日志并将缓存设置为空
                error!("获取MCP服务器[{server_name}]工具列表失败: {error}");
                self.tools.insert(server_name.to_string(), Vec::new());
            }
        }
    }

    /// 获取所有 MCP 工具列表，返回 LLM 可以使用的工具参数声明列表并处理 MCP 的名字
    pub async fn get_all_tools(&self) -> Vec<Map<String, Value>> {
        // 1.定义一个变量存储所有结果
        let mut all_tools = Vec::new();

        // 2.循环遍历所有缓存的工具
        for (server_name, tools) in &self.tools {
            // 3.循环取出每个 MCP 服务的工具列表
            for tool in tools {
                // 4.修改工具名字加上mcp_前缀+服务名字
                let tool_name = prefixed_tool_name(server_name, tool.name.as_ref());

                // 5.生成OpenAI工具描述
                let tool_schema = json!({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": format!(
                            "[{server_name}] {}",
                            tool.description.as_deref().unwrap_or(tool.name.as_ref())
                        ),
                        "parameters": tool.input_schema.as_ref(),
                    }
                });
                if let Value::Object(tool_schema) = tool_schema {
                    all_tools.push(tool_schema);
                }
            }
        }

        all_tools
    }

    /// 根据传递的工具名字+参数调用MCP工具
    pub async fn invoke(
        &self,
        tool_name: &str,
        arguments: Map<String, Value>,
    ) -> ToolResult<String> {
        let result: Result<ToolResult<String>> = async {
            // 1.定义变量存储原始的服务名字+工具
            let mut original_server_name = None;
            let mut original_tool_name = None;

            // 2.循环遍历当前的所有mcp服务配置
            let mut server_names = self
                .mcp_config
                .as_ref()
                .map(|config| config.mcp_servers.keys().collect::<Vec<_>>())
                .unwrap_or_default();
            server_names.sort_by_key(|server_name| {
                std::cmp::Reverse(if server_name.starts_with("mcp_") {
                    server_name.len()
                } else {
                    server_name.len() + 4
                })
            });

            for server_name in server_names {
                // 3.为server_name组装前缀
                let expected_prefix = if server_name.starts_with("mcp_") {
                    server_name.clone()
                } else {
                    format!("mcp_{server_name}")
                };

                // 4.判断工具名字是否以该服务名字为开头
                if let Some(name) = tool_name.strip_prefix(&format!("{expected_prefix}_")) {
                    // 5.取出原始的服务名字+工具名字
                    original_server_name = Some(server_name.clone());
                    original_tool_name = (!name.is_empty()).then(|| name.to_string());
                    break;
                }
            }

            // 6.判断服务名字+工具是否都存在
            let (original_server_name, original_tool_name) = original_server_name
                .zip(original_tool_name)
                .ok_or_else(|| anyhow!("服务器解析MCP工具不存在: {tool_name}"))?;

            // 7.获取该工具所属的会话
            let Some(session) = self.clients.get(&original_server_name) else {
                return Ok(ToolResult {
                    success: false,
                    message: Some(format!("MCP服务器[{original_server_name}]未连接")),
                    data: None,
                });
            };

            // 8.使用会话调用工具
            let result = session
                .call_tool(CallToolRequestParams::new(original_tool_name).with_arguments(arguments))
                .await?;

            // 9.判断结果是否存在执行不同的操作
            // 10.处理MCP工具生成的content
            let content = result
                .content
                .iter()
                .map(|item| {
                    item.raw.as_text().map_or_else(
                        || {
                            serde_json::to_string(item)
                                .unwrap_or_else(|_| format!("{:?}", item.raw))
                        },
                        |text| text.text.clone(),
                    )
                })
                .collect::<Vec<_>>();

            // 11.返回工具结果
            Ok(ToolResult {
                success: true,
                message: Some(String::new()),
                data: Some(if content.is_empty() {
                    "工具执行成功".to_string()
                } else {
                    content.join("\n")
                }),
            })
        }
        .await;

        match result {
            Ok(result) => result,
            Err(error) => {
                // 记录错误日志并返回失败的工具结果
                error!("调用MCP工具[{tool_name}]失败: {error}");
                ToolResult {
                    success: false,
                    message: Some(format!("调用MCP工具[{tool_name}]失败: {error}")),
                    data: None,
                }
            }
        }
    }

    /// 当退出MCP服务时，清除对应资源
    pub async fn cleanup(&mut self) {
        let mut cleanup_failed = false;
        for (server_name, mut session) in self.clients.drain() {
            if let Err(error) = session.close().await {
                cleanup_failed = true;
                error!("清理MCP服务器[{server_name}]失败: {error}");
            }
        }

        self.tools.clear();
        self.initialized = false;

        if cleanup_failed {
            error!("清理MCP客户端管理器失败");
        } else {
            info!("清除MCP客户端管理器成功");
        }
    }
}

fn prefixed_tool_name(server_name: &str, tool_name: &str) -> String {
    if server_name.starts_with("mcp_") {
        format!("{server_name}_{tool_name}")
    } else {
        format!("mcp_{server_name}_{tool_name}")
    }
}

fn value_map_to_strings(
    values: &Map<String, Value>,
    field_name: &str,
) -> Result<HashMap<String, String>> {
    values
        .iter()
        .map(|(name, value)| {
            let value = value
                .as_str()
                .ok_or_else(|| anyhow!("{field_name}[{name}]必须是字符串"))?;
            Ok((name.clone(), value.to_string()))
        })
        .collect()
}

fn http_headers(headers: Option<&Map<String, Value>>) -> Result<HashMap<HeaderName, HeaderValue>> {
    value_map_to_strings(headers.unwrap_or(&Map::new()), "请求头")?
        .into_iter()
        .map(|(name, value)| {
            let header_name = HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("请求头名称无效: {name}"))?;
            let header_value =
                HeaderValue::from_str(&value).with_context(|| format!("请求头[{name}]的值无效"))?;
            Ok((header_name, header_value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager_with_servers(server_names: &[&str]) -> McpClientManager {
        let mut config = McpConfig::default();
        for server_name in server_names {
            config
                .mcp_servers
                .insert((*server_name).to_string(), McpServerConfig::default());
        }
        McpClientManager::new(Some(config))
    }

    #[tokio::test]
    async fn initializes_and_cleans_up_empty_config() {
        let mut manager = McpClientManager::new(Some(McpConfig::default()));

        manager.initialize().await;
        manager.initialize().await;

        assert!(manager.initialized);
        assert!(manager.clients.is_empty());
        assert!(manager.tools.is_empty());

        manager.cleanup().await;
        assert!(!manager.initialized);
    }

    #[tokio::test]
    async fn skips_disabled_mcp_servers_during_initialize() {
        let mut config = McpConfig::default();
        config.mcp_servers.insert(
            "disabled".to_string(),
            McpServerConfig {
                enabled: false,
                ..McpServerConfig::default()
            },
        );
        let mut manager = McpClientManager::new(Some(config));

        manager.initialize().await;

        assert!(manager.initialized);
        assert!(manager.clients.is_empty());
        assert!(manager.tools.is_empty());
    }

    #[tokio::test]
    async fn builds_openai_tool_schemas_with_mcp_prefix() {
        let mut manager = manager_with_servers(&["demo", "mcp_system"]);
        manager.tools.insert(
            "demo".to_string(),
            vec![Tool::new(
                "search",
                "搜索内容",
                Map::from_iter([
                    ("type".to_string(), json!("object")),
                    (
                        "properties".to_string(),
                        json!({"query": {"type": "string"}}),
                    ),
                ]),
            )],
        );
        manager.tools.insert(
            "mcp_system".to_string(),
            vec![Tool::new("status", "获取状态", Map::new())],
        );

        let tools = manager.get_all_tools().await;

        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|tool| {
            tool["function"]["name"] == "mcp_demo_search"
                && tool["function"]["description"] == "[demo] 搜索内容"
                && tool["function"]["parameters"]["type"] == "object"
        }));
        assert!(tools
            .iter()
            .any(|tool| tool["function"]["name"] == "mcp_system_status"));
    }

    #[tokio::test]
    async fn resolves_the_longest_server_name_prefix() {
        let manager = manager_with_servers(&["demo", "demo_search"]);

        let result = manager.invoke("mcp_demo_search_query", Map::new()).await;

        assert_eq!(
            result.message.as_deref(),
            Some("MCP服务器[demo_search]未连接")
        );
    }

    #[tokio::test]
    async fn returns_failure_when_mcp_server_is_disconnected() {
        let manager = manager_with_servers(&["demo"]);

        let result = manager.invoke("mcp_demo_search", Map::new()).await;

        assert!(!result.success);
        assert_eq!(result.message.as_deref(), Some("MCP服务器[demo]未连接"));
        assert!(result.data.is_none());
    }

    #[tokio::test]
    async fn returns_failure_when_tool_name_cannot_be_resolved() {
        let manager = manager_with_servers(&["demo"]);

        let result = manager.invoke("missing", Map::new()).await;

        assert!(!result.success);
        assert_eq!(
            result.message.as_deref(),
            Some("调用MCP工具[missing]失败: 服务器解析MCP工具不存在: missing")
        );
    }

    #[test]
    fn converts_string_headers_and_rejects_other_values() {
        let headers = Map::from_iter([
            ("Authorization".to_string(), json!("Bearer token")),
            ("X-Trace-Id".to_string(), json!("trace-id")),
        ]);

        let converted = http_headers(Some(&headers)).unwrap();
        assert_eq!(
            converted[&HeaderName::from_static("authorization")],
            "Bearer token"
        );
        assert_eq!(
            converted[&HeaderName::from_static("x-trace-id")],
            "trace-id"
        );

        let invalid = Map::from_iter([("X-Retry".to_string(), json!(3))]);
        assert_eq!(
            http_headers(Some(&invalid)).unwrap_err().to_string(),
            "请求头[X-Retry]必须是字符串"
        );
    }
}
