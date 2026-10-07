//! 设置页面的展示数据。

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::agent::AgentConfig,
    model::configs::agent::{AgentConfigDraft, AgentConfigModel},
};

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentConfigViewModel {
    pub data: Option<AgentConfig>,
    pub draft: AgentConfigDraft,
    pub loading: bool,
    pub saving: bool,
    pub error: Option<String>,
    pub saved: bool,
    pub dirty: bool,
    pub can_save: bool,
}

impl From<&AgentConfigModel> for AgentConfigViewModel {
    fn from(model: &AgentConfigModel) -> Self {
        Self {
            data: model.data.clone(),
            draft: model.draft.clone(),
            loading: model.loading,
            saving: model.saving,
            error: model.error.clone(),
            saved: model.saved,
            dirty: model.dirty(),
            can_save: model.can_save(),
        }
    }
}
