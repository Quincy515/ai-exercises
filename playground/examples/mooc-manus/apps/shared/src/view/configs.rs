//! 设置页面的展示数据。

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{api::configs::AgentConfig, model::configs::ConfigsModel};

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentConfigViewModel {
    pub data: Option<AgentConfig>,
    pub loading: bool,
    pub error: Option<String>,
}

impl From<&ConfigsModel> for AgentConfigViewModel {
    fn from(model: &ConfigsModel) -> Self {
        Self {
            data: model.data.clone(),
            loading: model.loading,
            error: model.error.clone(),
        }
    }
}
