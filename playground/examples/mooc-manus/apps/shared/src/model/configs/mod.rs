//! 设置根模块：聚合各资源状态，通过子事件委派业务更新。

pub mod a2a;
pub mod agent;
pub mod llm;
pub mod mcp;

use crux_core::Command;
use crux_http::{HttpError, Response};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::effects::Effect;
use a2a::{A2aConfigEvent, A2aConfigModel};
use agent::{AgentConfigEvent, AgentConfigModel};
use llm::{LlmConfigEvent, LlmConfigModel};
use mcp::{McpConfigEvent, McpConfigModel};

#[derive(Facet, Serialize, Deserialize, Debug, PartialEq)]
#[repr(C)]
pub enum ConfigsEvent {
    Agent(AgentConfigEvent),
    Llm(LlmConfigEvent),
    A2a(A2aConfigEvent),
    Mcp(McpConfigEvent),
}

#[derive(Default)]
pub struct ConfigsModel {
    pub(crate) agent: AgentConfigModel,
    pub(crate) llm: LlmConfigModel,
    pub(crate) a2a: A2aConfigModel,
    pub(crate) mcp: McpConfigModel,
}

impl ConfigsModel {
    pub fn update(&mut self, event: ConfigsEvent) -> Command<Effect, ConfigsEvent> {
        match event {
            ConfigsEvent::Agent(event) => self.agent.update(event).map_event(ConfigsEvent::Agent),
            ConfigsEvent::Llm(event) => self.llm.update(event).map_event(ConfigsEvent::Llm),
            ConfigsEvent::A2a(event) => self.a2a.update(event).map_event(ConfigsEvent::A2a),
            ConfigsEvent::Mcp(event) => self.mcp.update(event).map_event(ConfigsEvent::Mcp),
        }
    }
}

// 设置接口共用固定错误分类，避免将后端响应或传输错误中的敏感内容展示出来。
fn receive_config<T>(
    result: crux_http::Result<Response<T>>,
    saved: bool,
    resource: &str,
) -> Result<T, String> {
    let action = if saved { "保存" } else { "读取" };
    match result {
        Ok(mut response) => response
            .take_body()
            .ok_or_else(|| "服务返回了空响应，请重试。".to_string()),
        Err(HttpError::Http { code: 422, .. }) if saved => {
            Err("配置未通过服务端校验，请检查输入范围后重试。".to_string())
        }
        Err(HttpError::Http { code, .. }) => Err(format!(
            "{action} {resource} 配置失败（HTTP {code}），请重试。"
        )),
        Err(HttpError::Json(_)) => Err("配置响应格式有误，请检查接口字段。".to_string()),
        Err(HttpError::Timeout) if saved => {
            Err("保存请求超时，结果尚未确认，请稍后重试或重新打开设置核对。".to_string())
        }
        Err(HttpError::Timeout) => Err("请求超时，请重试。".to_string()),
        Err(HttpError::Io(_)) if saved => {
            Err("保存时网络异常，结果尚未确认，请稍后重试或重新打开设置核对。".to_string())
        }
        Err(HttpError::Io(_)) => Err("网络请求失败，请检查后端服务和网络连接。".to_string()),
        Err(HttpError::Url(_)) => Err("服务地址无效，请检查 API 地址配置。".to_string()),
        Err(_) => Err(format!("{action} {resource} 配置失败，请重试。")),
    }
}
