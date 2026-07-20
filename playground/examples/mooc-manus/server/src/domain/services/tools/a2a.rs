//! A2A客户端管理器的开发思路:
//! 1.在Agent执行过程中, 有可能需要多次调用Remote-Agent，
//!   但是a2a中的agent-card.json请求是网络io, 相对耗时，
//!   所以需要缓存agent-card的相关信息, 只有在初始化A2A客户端的时候才初始化一次,
//!   更新a2a服务器的时候更新, 清除a2a客户端管理器时删除;
//! 2.在前端UI交互中, 无论A2A服务器是否启动, 都会展示Card信息,
//!   但是呢, 在执行/规划Agent中, 我们只传递启用的A2A服务, 所以A2A客户端管理器必须动态接受配置;
//! 3.一个A2A客户端会同时管理多个Agent, 但是不同的A2A服务有可能他们的name是一样的，
//!   需要考虑传递给Agent信息时的唯一性, 会配置多一个唯一的id;
//! 4.由于使用httpx客户端, 这个客户端需要创建上下文/释放资源, 所以可以使用AsyncExitStack来管理
//!   异步上下文, 避免大量使用with..as的嵌套组合;
//! 5.A2AClientManager的初始化非常耗时, 一次请求中只初始化一次;
//! 6.A2A配置是写在config.yaml中的并直接暴露给开发者, 有可能开发者会手动修改config.yaml
//!   所以在使用的时候, 最多需要做多一次校验;
//! 7.A2A客户端管理器只实现两个方法, 一个是get_remote_agent_cards、call_remote_agent;
//! 8.A2A客户端管理器停止时必须清除对应资源, 涵盖了缓存, 异步上下文管理器避免资源泄露;

use std::{collections::HashMap, time::Duration};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Map, Value};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::domain::models::{A2aConfig, ToolResult};

use super::{
    arguments::required_str,
    base::{tool, BaseTool, ToolArguments, ToolDefinition},
};

///  A2A 客户端管理器
pub struct A2AClientManager {
    /// 配置
    a2a_config: A2aConfig,
    /// reqwest 客户端
    http_client: Option<Client>,
    /// agent 卡片
    agent_cards: HashMap<String, Map<String, Value>>,
    /// 是否初始化
    initialized: bool,
}

impl A2AClientManager {
    /// 构造函数，完成 A2A 客户端的初始化
    pub fn new(a2a_config: Option<A2aConfig>) -> Self {
        Self {
            a2a_config: a2a_config.unwrap_or_default(),
            http_client: None,
            agent_cards: HashMap::new(),
            initialized: false,
        }
    }

    /// 只读属性，返回 agent 卡片信息
    pub fn agent_cards(&self) -> &HashMap<String, Map<String, Value>> {
        &self.agent_cards
    }

    /// 异步初始化函数，用于初始化所有已配置的 a2a 服务
    pub async fn initialize(&mut self) -> Result<()> {
        // 1.检测是否已经初始化
        if self.initialized {
            return Ok(());
        }

        let result: Result<()> = async {
            // 3.初始化 reqwest 客户端
            self.http_client = Some(
                Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()?,
            );

            // 4.记录日志并连接所有配置的 a2a 服务获取卡片信息
            info!("加载{}个 A2A 服务", self.a2a_config.a2a_servers.len());
            self.get_a2a_agent_cards().await;
            self.initialized = true;
            info!("A2A 客户端加载成功");
            Ok(())
        }
        .await;

        if let Err(err) = result {
            error!("A2A 客户端管理器加载失败: {err}");
            return Err(anyhow!("A2A 客户端管理器加载失败"));
        }

        Ok(())
    }

    /// 根据配置连接所有 a2a 服务器获取 AgentCard 信息
    async fn get_a2a_agent_cards(&mut self) {
        // 1.循环遍历所有的 a2a 服务
        for a2a_server_config in &self.a2a_config.a2a_servers {
            let result: Result<Map<String, Value>> = async {
                // 2.调用 reqwest 客户端发起请求
                let agent_card_response = self
                    .http_client
                    .as_ref()
                    .ok_or_else(|| anyhow!("A2A 客户端未初始化"))?
                    .get(format!(
                        "{}/.well-known/agent-card.json",
                        a2a_server_config.base_url
                    ))
                    .send()
                    .await?
                    .error_for_status()?;
                let agent_card = agent_card_response.json().await?;

                Ok(agent_card)
            }
            .await;

            match result {
                Ok(agent_card) => {
                    // 3.存储到 agent_cards
                    self.agent_cards
                        .insert(a2a_server_config.id.clone(), agent_card);
                }
                Err(err) => {
                    warn!("加载 A2A 服务[{}]失败: {err}", a2a_server_config.id);
                }
            }
        }
    }

    /// 根据传递的智能体 id+query 调用 Remote-Agent
    pub async fn invoke(&self, agent_id: &str, query: &str) -> ToolResult<Value> {
        // 1.判断传递的 agent_id 是否存在
        let Some(agent_card) = self.agent_cards.get(agent_id) else {
            return ToolResult {
                success: false,
                message: Some("该远程 Agent 不存在".to_string()),
                data: None,
            };
        };

        // 2.Agent 存在，则取出端点信息
        let url = agent_card
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or_default();

        // 3.判断端点是否存在
        if url.is_empty() {
            return ToolResult {
                success: false,
                message: Some("该远程 Agent 调用端点不存在".to_string()),
                data: None,
            };
        }

        let result: Result<Value> = async {
            // 4.使用 reqwest 客户端发起 post 请求并传递数据
            let agent_response = self
                .http_client
                .as_ref()
                .ok_or_else(|| anyhow!("A2A 客户端未初始化"))?
                .post(url)
                .json(&json!({
                    "id": Uuid::new_v4().to_string(),
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": Uuid::new_v4().simple().to_string(),
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": query},
                            ],
                        },
                    },
                }))
                .send()
                .await?
                .error_for_status()?;
            let result = agent_response.json().await?;

            Ok(result)
        }
        .await;

        match result {
            Ok(result) => ToolResult {
                success: true,
                message: Some("调用远程 Agent 成功".to_string()),
                data: Some(result),
            },
            Err(err) => {
                error!("调用远程 Agent [{agent_id}:{url}] 出错: {err}");
                ToolResult {
                    success: false,
                    message: Some(format!("调用远程 Agent [{agent_id}:{url}] 出错: {err}")),
                    data: None,
                }
            }
        }
    }

    /// 当退出 A2A 客户端管理器时，清除对应资源
    pub async fn cleanup(&mut self) {
        // reqwest 客户端在句柄被丢弃时自动释放连接资源
        self.http_client = None;
        self.agent_cards.clear();
        self.initialized = false;
        info!("清除 A2A 客户端管理器成功");
    }
}

/// A2A 工具包，根据传递的配置完成 A2A 工具包的初始化
pub struct A2ATool {
    manager: Option<A2AClientManager>,
    definitions: Vec<ToolDefinition>,
    initialized: bool,
}

impl A2ATool {
    /// 构造函数，完成工具包初始化
    pub fn new() -> Self {
        Self {
            manager: None,
            definitions: vec![
                tool(
                    "get_remote_agent_cards",
                    "获取可远程调用的 Agent 卡片信息, 包含 Agent id、名称、描述、技能、请求端点等。",
                    ToolArguments::new(),
                    Vec::new(),
                ),
                tool(
                    "call_remote_agent",
                    "根据传递的 id+query(分配给远程 Agent 完成的任务 query) 调用远程 Agent 完成对应需求",
                    ToolArguments::from_iter([
                        (
                            "id".to_string(),
                            json!({
                                "type": "string",
                                "description": "需要调用远程 Agent 的 id, 格式参考 get_remote_agent_cards() 返回的数据结构",
                            }),
                        ),
                        (
                            "query".to_string(),
                            json!({
                                "type": "query",
                                "description": "需要分配给该远程 Agent 实现的任务/需求 query",
                            }),
                        ),
                    ]),
                    vec!["id".to_string(), "query".to_string()],
                ),
            ],
            initialized: false,
        }
    }

    /// 初始化 A2A 工具包
    pub async fn initialize(&mut self, a2a_config: Option<A2aConfig>) -> Result<()> {
        // 1.判断下是否已初始化
        if !self.initialized {
            // 2.初始化 A2A 客户端管理器
            let mut manager = A2AClientManager::new(a2a_config);
            manager.initialize().await?;
            self.manager = Some(manager);
            self.initialized = true;
        }

        Ok(())
    }

    /// 获取远程 Agent 卡片信息列表
    pub async fn get_remote_agent_cards(&self) -> Result<ToolResult<Vec<Value>>> {
        let manager = self
            .manager
            .as_ref()
            .ok_or_else(|| anyhow!("A2A工具包未初始化"))?;

        // 1.重组结构，将 id 填充到 agent_card 中
        let agent_cards = manager
            .agent_cards()
            .iter()
            .map(|(id, agent_card)| {
                let mut agent_card_with_id = Map::new();
                agent_card_with_id.insert("id".to_string(), Value::String(id.clone()));
                agent_card_with_id.extend(agent_card.clone());
                Value::Object(agent_card_with_id)
            })
            .collect();

        // 2.组装 ToolResult 响应
        Ok(ToolResult {
            success: true,
            message: Some("获取Agent卡片信息列表成功".to_string()),
            data: Some(agent_cards),
        })
    }

    /// 调用远程 Agent 并完成对应需求
    pub async fn call_remote_agent(&self, id: &str, query: &str) -> Result<ToolResult<Value>> {
        let manager = self
            .manager
            .as_ref()
            .ok_or_else(|| anyhow!("A2A工具包未初始化"))?;
        Ok(manager.invoke(id, query).await)
    }
}

impl Default for A2ATool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseTool for A2ATool {
    fn name(&self) -> &str {
        "a2a"
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        match tool_name {
            "get_remote_agent_cards" => {
                let result = self.get_remote_agent_cards().await?;
                Ok(ToolResult {
                    success: result.success,
                    message: result.message,
                    data: result.data.map(Value::Array),
                })
            }
            "call_remote_agent" => {
                let id = required_str(&kwargs, "id")?;
                let query = required_str(&kwargs, "query")?;
                self.call_remote_agent(id, query).await
            }
            _ => Err(anyhow!("工具[{tool_name}]未找到")),
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        if let Some(manager) = self.manager.as_mut() {
            manager.cleanup().await;
        }
        self.manager = None;
        self.initialized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        net::Ipv4Addr,
        sync::{Arc, Mutex},
    };

    use axum::{
        extract::State,
        routing::{get, post},
        Json, Router,
    };
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::*;
    use crate::domain::models::A2aServerConfig;

    #[derive(Clone)]
    struct TestState {
        invoke_url: String,
        request: Arc<Mutex<Option<Value>>>,
    }

    struct TestServer {
        base_url: String,
        request: Arc<Mutex<Option<Value>>>,
        task: JoinHandle<()>,
    }

    impl TestServer {
        async fn start() -> Self {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let address = listener.local_addr().unwrap();
            let base_url = format!("http://{address}");
            let request = Arc::new(Mutex::new(None));
            let state = TestState {
                invoke_url: format!("{base_url}/invoke"),
                request: request.clone(),
            };
            let app = Router::new()
                .route("/.well-known/agent-card.json", get(agent_card))
                .route("/invoke", post(invoke_agent))
                .with_state(state);
            let task = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });

            Self {
                base_url,
                request,
                task,
            }
        }

        fn config(&self, id: &str) -> A2aConfig {
            A2aConfig {
                a2a_servers: vec![A2aServerConfig {
                    id: id.to_string(),
                    base_url: self.base_url.clone(),
                    enabled: true,
                }],
            }
        }

        fn request(&self) -> Value {
            self.request.lock().unwrap().clone().unwrap()
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn agent_card(State(state): State<TestState>) -> Json<Value> {
        Json(json!({
            "name": "测试 Agent",
            "description": "用于 A2A 单元测试",
            "url": state.invoke_url,
        }))
    }

    async fn invoke_agent(
        State(state): State<TestState>,
        Json(payload): Json<Value>,
    ) -> Json<Value> {
        *state.request.lock().unwrap() = Some(payload);
        Json(json!({"jsonrpc": "2.0", "result": {"status": "completed"}}))
    }

    #[tokio::test]
    async fn manager_loads_invokes_and_cleans_up_remote_agents() {
        let server = TestServer::start().await;
        let mut config = server.config("remote");
        config.a2a_servers.push(A2aServerConfig {
            id: "unavailable".to_string(),
            base_url: "http://127.0.0.1:0".to_string(),
            enabled: true,
        });
        let mut manager = A2AClientManager::new(Some(config));

        manager.initialize().await.unwrap();
        manager.initialize().await.unwrap();

        assert!(manager.initialized);
        assert_eq!(manager.agent_cards().len(), 1);
        assert_eq!(manager.agent_cards()["remote"]["name"], "测试 Agent");

        let result = manager.invoke("remote", "分析这段代码").await;
        assert!(result.success);
        assert_eq!(result.message.as_deref(), Some("调用远程 Agent 成功"));
        assert_eq!(result.data.unwrap()["result"]["status"], "completed");

        let request = server.request();
        assert_eq!(request["jsonrpc"], "2.0");
        assert_eq!(request["method"], "message/send");
        assert_eq!(
            request["params"]["message"]["parts"],
            json!([{"kind": "text", "text": "分析这段代码"}])
        );

        manager.cleanup().await;
        assert!(!manager.initialized);
        assert!(manager.http_client.is_none());
        assert!(manager.agent_cards().is_empty());
    }

    #[tokio::test]
    async fn manager_reports_missing_agents_endpoints_and_request_errors() {
        let mut manager = A2AClientManager::new(None);
        manager.initialize().await.unwrap();

        let result = manager.invoke("missing", "任务").await;
        assert!(!result.success);
        assert_eq!(result.message.as_deref(), Some("该远程 Agent 不存在"));

        manager
            .agent_cards
            .insert("without-url".to_string(), Map::new());
        let result = manager.invoke("without-url", "任务").await;
        assert_eq!(
            result.message.as_deref(),
            Some("该远程 Agent 调用端点不存在")
        );

        manager.agent_cards.insert(
            "unavailable".to_string(),
            Map::from_iter([(
                "url".to_string(),
                Value::String("http://127.0.0.1:0".to_string()),
            )]),
        );
        let result = manager.invoke("unavailable", "任务").await;
        assert!(!result.success);
        assert!(result
            .message
            .as_deref()
            .unwrap()
            .starts_with("调用远程 Agent [unavailable:http://127.0.0.1:0] 出错:"));
    }

    #[test]
    fn exposes_a2a_tool_schemas_like_python_tool_decorators() {
        let tool = A2ATool::new();

        assert_eq!(tool.name(), "a2a");
        assert!(tool.has_tool("get_remote_agent_cards"));
        assert!(tool.has_tool("call_remote_agent"));
        assert!(!tool.has_tool("missing"));

        let schemas = tool.get_tools();
        let call_agent = schemas
            .iter()
            .find(|schema| schema["function"]["name"] == "call_remote_agent")
            .unwrap();
        assert_eq!(
            call_agent["function"]["parameters"]["properties"]["id"]["type"],
            "string"
        );
        assert_eq!(
            call_agent["function"]["parameters"]["properties"]["query"]["type"],
            "query"
        );
        assert_eq!(
            call_agent["function"]["parameters"]["required"],
            json!(["id", "query"])
        );
    }

    #[tokio::test]
    async fn initializes_dispatches_validates_and_cleans_up_a2a_tool() {
        let server = TestServer::start().await;
        let mut tool = A2ATool::new();
        tool.initialize(Some(server.config("remote")))
            .await
            .unwrap();

        let cards = tool
            .invoke("get_remote_agent_cards", Map::new())
            .await
            .unwrap();
        assert_eq!(cards.data.unwrap()[0]["id"], "remote");

        let result = tool
            .invoke(
                "call_remote_agent",
                Map::from_iter([
                    ("id".to_string(), json!("remote")),
                    ("query".to_string(), json!("完成远程任务")),
                    ("extra".to_string(), json!("ignored")),
                ]),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(
            server.request()["params"]["message"]["parts"][0]["text"],
            "完成远程任务"
        );

        let error = tool
            .invoke(
                "call_remote_agent",
                Map::from_iter([("id".to_string(), json!("remote"))]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "工具参数[query]缺失");

        tool.cleanup().await.unwrap();
        assert!(!tool.initialized);
        assert!(tool.manager.is_none());
    }
}
