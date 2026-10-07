//! MCP 安全视图：完整 JSON、环境变量和认证请求头保持在私有草稿中。

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{api::configs::mcp::McpServer, model::configs::mcp::McpConfigModel};

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct McpConfigViewModel {
    pub servers: Vec<McpServer>,
    pub draft_present: bool,
    pub loaded: bool,
    pub loading: bool,
    pub saving: bool,
    pub error: Option<String>,
    pub notice: Option<String>,
    pub created: bool,
    pub write_uncertain: bool,
}

impl From<&McpConfigModel> for McpConfigViewModel {
    fn from(model: &McpConfigModel) -> Self {
        Self {
            servers: model.servers.clone(),
            draft_present: model.draft_present(),
            loaded: model.loaded,
            loading: model.loading,
            saving: model.saving,
            error: model.error.clone(),
            notice: model.notice.clone(),
            created: model.created,
            write_uncertain: model.write_uncertain,
        }
    }
}
